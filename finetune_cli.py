#!/usr/bin/env python
import argparse
import json
import os
import random
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio
from tqdm import tqdm
import whisper
import wandb  # <--- NEW: Import WandB

from inference import (
    load_model_from_hf,
    load_fish_ae_from_hf,
    load_pca_state_from_hf,
    load_audio,
    ae_encode,
    ae_decode,
    get_speaker_latent_and_mask,
    get_text_input_ids_and_mask,
    sample_euler_cfg_independent_guidances,
    crop_audio_to_flattening_point,
)

# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------

AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".ogg", ".flac", ".webm", ".aac", ".opus"}
AE_DOWNSAMPLE_FACTOR = 2048
MAX_LATENT_LENGTH = 640
MAX_AUDIO_SAMPLES = AE_DOWNSAMPLE_FACTOR * MAX_LATENT_LENGTH


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def list_audio_files(root: Path) -> List[Path]:
    root = root.expanduser()
    if not root.exists():
        return []
    return sorted(
        [
            p
            for p in root.rglob("*")
            if p.is_file() and p.suffix.lower() in AUDIO_EXTS
        ]
    )


# --------------------------------------------------------------------------------------
# 1️⃣ PREPARE
# --------------------------------------------------------------------------------------

def cmd_prepare(raw_dir: Path, segments_dir: Path, max_duration: int = 30) -> None:
    raw_dir = raw_dir.expanduser()
    segments_dir = segments_dir.expanduser()
    ensure_dir(segments_dir)

    files = list_audio_files(raw_dir)
    print(f"[Prepare] Found {len(files)} raw files in {raw_dir}")
    if not files:
        return

    max_samples = int(44_100 * max_duration)

    for in_path in tqdm(files, desc="[Prepare] slicing", ncols=100):
        wav, sr = torchaudio.load(str(in_path))
        if sr != 44_100:
            wav = torchaudio.functional.resample(wav, sr, 44_100)
            sr = 44_100

        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)

        total = wav.shape[-1]
        base = in_path.stem

        idx = 0
        for start in range(0, total, max_samples):
            end = min(start + max_samples, total)
            chunk = wav[:, start:end]
            if chunk.shape[-1] < int(0.5 * 44_100):
                continue
            out_name = f"{base}_{idx:03d}.wav"
            out_path = segments_dir / out_name
            torchaudio.save(str(out_path), chunk, sr)
            idx += 1

    print(f"[Prepare] Done. Segments in: {segments_dir}")


# --------------------------------------------------------------------------------------
# 2️⃣ TRANSCRIBE
# --------------------------------------------------------------------------------------

def load_existing_metadata_for_skip(meta_path: Path) -> Dict[str, Dict[str, Any]]:
    existing: Dict[str, Dict[str, Any]] = {}
    if not meta_path.exists():
        return existing

    with meta_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            audio_val = obj.get("audio")
            text_val = obj.get("text")
            if not audio_val or not text_val:
                continue
            name = Path(audio_val).name
            existing[name] = obj
    return existing


def cmd_transcribe(segments_dir: Path, metadata: Path, lang: str | None) -> None:
    segments_dir = segments_dir.expanduser()
    metadata = metadata.expanduser()
    ensure_dir(metadata.parent)

    files = list_audio_files(segments_dir)
    print(f"[Transcribe] Scanning {len(files)} files in {segments_dir}")

    existing = load_existing_metadata_for_skip(metadata)
    print(f"[Transcribe] Found {len(existing)} existing entries.")

    to_process: List[Path] = []
    for p in files:
        name = p.name
        if name in existing:
            continue
        to_process.append(p)

    if not to_process:
        print("[Transcribe] Nothing new to transcribe. Done.")
        return

    print(f"[Transcribe] Loading Whisper `large-v3` …")
    model = whisper.load_model("large-v3")

    new_count = 0
    with metadata.open("a", encoding="utf-8") as fout:
        for p in tqdm(to_process, desc="[Transcribe] files", ncols=100):
            abs_path = p.resolve()
            result = model.transcribe(
                str(abs_path),
                language=lang if lang else None,
                task="transcribe",
                verbose=False,
            )
            text = result.get("text", "").strip()
            if not text:
                continue

            obj = {
                "audio": str(abs_path),
                "text": text,
                "language": lang or result.get("language", None),
            }
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            new_count += 1

    print(f"[Transcribe] Added {new_count} new segments.")


# --------------------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------------------

class EchoDataset(Dataset):
    def __init__(self, metadata_path: Path):
        self.items: List[Dict[str, Any]] = []
        metadata_path = metadata_path.expanduser()
        if not metadata_path.exists():
            raise FileNotFoundError(metadata_path)

        with metadata_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                audio = obj.get("audio")
                text = obj.get("text", "").strip()
                if not audio or not text:
                    continue
                self.items.append({"audio": audio, "text": text})

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        target_item = self.items[idx]
        if len(self.items) > 1:
            ref_idx = idx
            while ref_idx == idx:
                ref_idx = random.randint(0, len(self.items) - 1)
            ref_item = self.items[ref_idx]
        else:
            ref_item = target_item
        return {"target": target_item, "reference": ref_item}


def build_batch_latents(
    batch: List[Dict[str, Any]],
    fish_ae,
    pca_state,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    B = len(batch)
    target_audio_list: List[torch.Tensor] = []
    ref_audio_list: List[torch.Tensor] = []
    texts: List[str] = []

    for item in batch:
        t_path = item["target"]["audio"]
        t_text = item["target"]["text"]
        t_audio = load_audio(t_path)
        target_audio_list.append(t_audio)
        texts.append(t_text)

        r_path = item["reference"]["audio"]
        r_audio = load_audio(r_path)
        ref_audio_list.append(r_audio)

    # Targets
    target_batch = torch.zeros((B, 1, MAX_AUDIO_SAMPLES), dtype=torch.float32, device=device)
    latent_mask = torch.zeros((B, MAX_LATENT_LENGTH), dtype=torch.bool, device=device)

    for i, a in enumerate(target_audio_list):
        a = a.to(device)
        L = a.shape[-1]
        L_clamped = min(L, MAX_AUDIO_SAMPLES)
        target_batch[i, 0, :L_clamped] = a[0, :L_clamped]
        latent_len = L_clamped // AE_DOWNSAMPLE_FACTOR
        latent_len = min(latent_len, MAX_LATENT_LENGTH)
        if latent_len > 0:
            latent_mask[i, :latent_len] = True

    latent = ae_encode(fish_ae, pca_state, target_batch.to(fish_ae.dtype))

    # References
    spk_latents: List[torch.Tensor] = []
    spk_masks: List[torch.Tensor] = []

    for i, r_audio in enumerate(ref_audio_list):
        r_audio = r_audio.to(device)
        spk_latent, spk_mask = get_speaker_latent_and_mask(
            fish_ae, pca_state, r_audio.to(fish_ae.dtype),
            max_speaker_latent_length=MAX_LATENT_LENGTH, pad_to_max=False,
        )
        spk_latents.append(spk_latent[0])
        spk_masks.append(spk_mask[0])

    max_spk_len = max(t.shape[0] for t in spk_latents)
    speaker_latent = torch.zeros((B, max_spk_len, 80), dtype=fish_ae.dtype, device=device)
    speaker_mask = torch.zeros((B, max_spk_len), dtype=torch.bool, device=device)
    for i, (sl, sm) in enumerate(zip(spk_latents, spk_masks)):
        T = sl.shape[0]
        speaker_latent[i, :T] = sl
        speaker_mask[i, :T] = sm

    # Text
    text_input_ids, text_mask = get_text_input_ids_and_mask(
        texts, max_length=None, device=str(device), normalize=True,
        return_normalized_text=False, pad_to_max=True,
    )

    return {
        "latent": latent.to(device),
        "latent_mask": latent_mask,
        "speaker_latent": speaker_latent,
        "speaker_mask": speaker_mask,
        "text_input_ids": text_input_ids,
        "text_mask": text_mask,
    }


# --------------------------------------------------------------------------------------
# Validation & Training
# --------------------------------------------------------------------------------------

def run_inference_test(
    model, fish_ae, pca_state, device, step,
    speaker_path, text_prompt, 
    cfg_text, cfg_speaker, kv_scale, kv_min_t
):
    print(f"\n[Val] 🧪 Validation Step {step}...")
    out_dir = Path("outputs")
    ensure_dir(out_dir)
    out_path = out_dir / f"test_step{step}.wav"

    try:
        speaker_audio = load_audio(str(speaker_path)).to(device)
        speaker_latent, speaker_mask = get_speaker_latent_and_mask(
            fish_ae, pca_state, speaker_audio.to(fish_ae.dtype),
            max_speaker_latent_length=6400, pad_to_max=False
        )
        text_input_ids, text_mask = get_text_input_ids_and_mask(
            [text_prompt], max_length=None, device=device, normalize=True
        )

        model.eval()
        with torch.no_grad():
            latent_out = sample_euler_cfg_independent_guidances(
                model=model,
                speaker_latent=speaker_latent,
                speaker_mask=speaker_mask,
                text_input_ids=text_input_ids,
                text_mask=text_mask,
                rng_seed=42, 
                num_steps=40,
                cfg_scale_text=cfg_text,
                cfg_scale_speaker=cfg_speaker,
                cfg_min_t=0.5,
                cfg_max_t=1.0,
                truncation_factor=0.9,
                rescale_k=1.2,
                rescale_sigma=3.0,
                speaker_kv_scale=kv_scale,
                speaker_kv_max_layers=None,
                speaker_kv_min_t=kv_min_t,
                sequence_length=640
            )
            audio_out = ae_decode(fish_ae, pca_state, latent_out)
            audio_out = crop_audio_to_flattening_point(audio_out, latent_out[0])
            
            # Save locally
            torchaudio.save(str(out_path), audio_out[0].cpu(), 44100)
            print(f"[Val] ✅ Saved: {out_path}")
            
            # Log to WandB
            wandb.log({
                "val/audio": wandb.Audio(str(out_path), caption=f"Step {step}"),
                "val/step": step
            })

    except Exception as e:
        print(f"[Val] ❌ Error: {e}")
    
    model.train()

def train_step(model, batch_latents, optimizer, device):
    model.train()
    latent = batch_latents["latent"]
    latent_mask = batch_latents["latent_mask"]
    speaker_latent = batch_latents["speaker_latent"]
    speaker_mask = batch_latents["speaker_mask"]
    text_input_ids = batch_latents["text_input_ids"]
    text_mask = batch_latents["text_mask"]

    B, T, _ = latent.shape
    t = torch.rand(B, device=device)
    noise = torch.randn_like(latent)
    t_broadcast = t.view(B, 1, 1)
    x_noised = latent * (1.0 - t_broadcast) + noise * t_broadcast

    kv_text = model.get_kv_cache_text(text_input_ids, text_mask)
    kv_speaker = model.get_kv_cache_speaker(speaker_latent.to(model.dtype))

    optimizer.zero_grad(set_to_none=True)

    v_pred = model(
        x=x_noised.to(model.dtype),
        t=t.to(model.dtype),
        text_mask=text_mask,
        speaker_mask=speaker_mask,
        kv_cache_text=kv_text,
        kv_cache_speaker=kv_speaker,
    ).float()

    loss = ((v_pred - (noise - latent)) ** 2).mean(dim=-1)
    if latent_mask is not None:
        lm = latent_mask.float()
        loss = (loss * lm).sum() / (lm.sum() + 1e-6)
    else:
        loss = loss.mean()

    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return loss.detach().item(), grad_norm.item()


def cmd_train(
    segments_dir: Path, metadata: Path, out_dir: Path, lang: str | None,
    batch_size: int, steps: int, lr: float, device_str: str,
    resume_path: Path | None, 
    val_speaker: Path | None, val_text: str | None,
    use_wandb: bool
) -> None:
    
    device = torch.device(device_str)
    out_dir = out_dir.expanduser()
    ensure_dir(out_dir)

    # Initialize WandB
    if use_wandb:
        wandb.init(project="echo-tts-finetune", name="abu-8hr-run")
        wandb.config.update({
            "lr": lr, "batch_size": batch_size, "steps": steps
        })

    dataset = EchoDataset(metadata)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=lambda x: x)

    print(f"[Train] Loading Model...")
    model = load_model_from_hf(device=device_str, dtype=torch.bfloat16, delete_blockwise_modules=True)
    fish_ae = load_fish_ae_from_hf(device=device_str, dtype=torch.float32)
    pca_state = load_pca_state_from_hf()

    global_step = 0
    if resume_path and resume_path.exists():
        print(f"[Train] 🔄 Resuming from {resume_path}")
        model.load_state_dict(torch.load(resume_path, map_location=device), strict=False)
        match = re.search(r"step(\d+)", resume_path.stem)
        if match:
            global_step = int(match.group(1))
    else:
        print("[Train] ⭐ Fresh Start")

    # Freeze Logic
    print("[Train] Freezing Encoders...")
    freeze_keywords = ["text", "speaker", "encoder", "embed", "condition"]
    frozen_c, trainable_c = 0, 0
    for name, param in model.named_parameters():
        should_freeze = any(k in name for k in freeze_keywords) and "latent" not in name
        if should_freeze:
            param.requires_grad = False
            frozen_c += param.numel()
        else:
            param.requires_grad = True
            trainable_c += param.numel()
            
    print(f"[Train] Trainable: {trainable_c/1e6:.1f}M | Frozen: {frozen_c/1e6:.1f}M")
    
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    print(f"[Train] Go for {steps} steps...")
    
    while global_step < steps:
        for batch in loader:
            if global_step >= steps: break
            
            # Validation
            if val_speaker and val_text and global_step % 500 == 0 and global_step > 0:
                run_inference_test(
                    model, fish_ae, pca_state, device, global_step,
                    val_speaker, val_text, 
                    cfg_text=4.0, cfg_speaker=2.0, kv_scale=1.2, kv_min_t=0.2
                )

            batch_latents = build_batch_latents(batch, fish_ae, pca_state, device)
            loss, grad_norm = train_step(model, batch_latents, optimizer, device)
            global_step += 1

            if use_wandb:
                wandb.log({
                    "train/loss": loss,
                    "train/grad_norm": grad_norm,
                    "train/step": global_step
                })

            if global_step % 10 == 0:
                print(f"[Train] step {global_step}/{steps} | loss={loss:.4f} | grad={grad_norm:.4f}")

            if global_step % 1000 == 0 or global_step == steps:
                ckpt = out_dir / f"echo_abu_step{global_step}.pt"
                torch.save(model.state_dict(), ckpt)
                print(f"[Train] Saved checkpoint → {ckpt}")

    torch.save(model.state_dict(), out_dir / "echo_abu_final.pt")
    if use_wandb: wandb.finish()
    print("[Train] Done.")


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Echo-TTS Finetune CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # Prepare
    p_prep = sub.add_parser("prepare")
    p_prep.add_argument("--raw_dir", type=Path, required=True)
    p_prep.add_argument("--segments_dir", type=Path, required=True)
    p_prep.add_argument("--max_duration", type=int, default=30)

    # Transcribe
    p_tr = sub.add_parser("transcribe")
    p_tr.add_argument("--segments_dir", type=Path, required=True)
    p_tr.add_argument("--metadata", type=Path, required=True)
    p_tr.add_argument("--lang", type=str, default=None)

    # Train
    p_train = sub.add_parser("train")
    p_train.add_argument("--segments_dir", type=Path, required=True)
    p_train.add_argument("--metadata", type=Path, required=True)
    p_train.add_argument("--out_dir", type=Path, required=True)
    p_train.add_argument("--lang", type=str, default=None)
    p_train.add_argument("--batch_size", type=int, default=4)
    p_train.add_argument("--steps", type=int, default=1000)
    p_train.add_argument("--lr", type=float, default=2e-5)
    p_train.add_argument("--device", type=str, default="cuda")
    p_train.add_argument("--resume", type=Path, default=None)
    
    # Validation Args
    p_train.add_argument("--val_speaker", type=Path, default=None)
    p_train.add_argument("--val_text", type=str, default=None)
    p_train.add_argument("--no_wandb", action="store_true", help="Disable WandB logging")

    args = parser.parse_args()

    if args.cmd == "prepare":
        cmd_prepare(args.raw_dir, args.segments_dir, args.max_duration)
    elif args.cmd == "transcribe":
        cmd_transcribe(args.segments_dir, args.metadata, args.lang)
    elif args.cmd == "train":
        cmd_train(
            args.segments_dir, args.metadata, args.out_dir, args.lang,
            args.batch_size, args.steps, args.lr, args.device, 
            args.resume, args.val_speaker, args.val_text,
            use_wandb=not args.no_wandb
        )

if __name__ == "__main__":
    main()
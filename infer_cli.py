# infer_cli.py
import argparse
import os
import time
from functools import partial
from pathlib import Path

import torch
import torchaudio

from inference import (
    load_model_from_hf,
    load_fish_ae_from_hf,
    load_pca_state_from_hf,
    load_audio,
    sample_pipeline,
    sample_euler_cfg_independent_guidances,
)


def parse_args():
    p = argparse.ArgumentParser(
        description="Echo-TTS CLI inference (base or finetuned checkpoint)."
    )

    p.add_argument(
        "--checkpoint",
        type=str,
        default="",
        help="Path to finetuned checkpoint (.pt). "
             "E.g. checkpoints/echo_abu/echo_abu_step700.pt. "
             "If empty, uses base Echo-TTS.",
    )
    p.add_argument(
        "--speaker",
        type=str,
        required=True,
        help="Path to reference audio (wav/mp3/etc).",
    )
    p.add_argument(
        "--text",
        type=str,
        required=True,
        help="Text to synthesize. Use [S1] / [S2] tags if needed.",
    )
    p.add_argument(
        "--out",
        type=str,
        default="outputs/echo_out.wav",
        help="Output wav path.",
    )
    p.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to use. 'auto' = cuda if available else cpu.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="RNG seed.",
    )
    p.add_argument(
        "--num_steps",
        type=int,
        default=40,
        help="Diffusion steps (20–60 is typical).",
    )
    p.add_argument(
        "--cfg_text",
        type=float,
        default=3.0,
        help="Text CFG scale.",
    )
    p.add_argument(
        "--cfg_speaker",
        type=float,
        default=8.0,
        help="Speaker CFG scale.",
    )
    p.add_argument(
        "--cfg_min_t",
        type=float,
        default=0.5,
        help="CFG min t (0–1).",
    )
    p.add_argument(
        "--cfg_max_t",
        type=float,
        default=1.0,
        help="CFG max t (0–1).",
    )
    p.add_argument(
        "--truncation",
        type=float,
        default=0.8,
        help="Initial noise truncation factor (0.8 or 0.9).",
    )
    p.add_argument(
        "--rescale_k",
        type=float,
        default=1.2,
        help="Temporal score rescale k (1.2 = flatter, 0.96 = sharper, 1.0 = off).",
    )
    p.add_argument(
        "--rescale_sigma",
        type=float,
        default=3.0,
        help="Temporal score rescale sigma.",
    )
    p.add_argument(
        "--sequence_length",
        type=int,
        default=640,
        help="Latent sequence length (640 ≈ 30s max).",
    )

    return p.parse_args()


def main():
    args = parse_args()

    # -------- device ----------
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print(f"[Infer] Using device: {device}")

    # -------- load base model + AE ----------
    t0 = time.time()
    print("[Infer] Loading base Echo-TTS model...")
    model = load_model_from_hf(
        delete_blockwise_modules=True,
        dtype=torch.bfloat16,
        device=device,
    )

    print("[Infer] Loading Fish AE...")
    fish_ae = load_fish_ae_from_hf(
        dtype=torch.float32,
        device=device,
    )
    pca_state = load_pca_state_from_hf(device=device)
    print(f"[Infer] Base models loaded in {time.time() - t0:.1f}s")

    # -------- load finetuned checkpoint (if any) ----------
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        print(f"[Infer] Loading finetuned checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)

        # Our finetune script saved {'step': ..., 'model': state_dict, 'optimizer': ...}
        state_dict = ckpt.get("model", ckpt)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"[Infer]   loaded; missing={len(missing)}, unexpected={len(unexpected)}")

    # -------- load speaker audio ----------
    speaker_path = Path(args.speaker)
    if not speaker_path.exists():
        raise FileNotFoundError(f"Speaker audio not found: {speaker_path}")

    print(f"[Infer] Loading speaker audio from {speaker_path}")
    speaker_audio = load_audio(str(speaker_path)).to(device)

    # -------- build sampler fn ----------
    sample_fn = partial(
        sample_euler_cfg_independent_guidances,
        num_steps=int(args.num_steps),
        cfg_scale_text=float(args.cfg_text),
        cfg_scale_speaker=float(args.cfg_speaker),
        cfg_min_t=float(args.cfg_min_t),
        cfg_max_t=float(args.cfg_max_t),
        truncation_factor=float(args.truncation),
        rescale_k=float(args.rescale_k),
        rescale_sigma=float(args.rescale_sigma),
        speaker_kv_scale=None,
        speaker_kv_max_layers=None,
        speaker_kv_min_t=None,
        sequence_length=int(args.sequence_length),
    )

    # -------- run generation ----------
    print("[Infer] Generating audio...")
    gen_t0 = time.time()
    with torch.inference_mode():
        audio_out, normalized_text = sample_pipeline(
            model=model,
            fish_ae=fish_ae,
            pca_state=pca_state,
            sample_fn=sample_fn,
            text_prompt=args.text,
            speaker_audio=speaker_audio,
            rng_seed=int(args.seed),
            pad_to_max_speaker_latent_length=None,
            pad_to_max_text_length=None,
            normalize_text=True,
        )
    torch.cuda.synchronize() if device.startswith("cuda") else None
    print(f"[Infer] Generation done in {time.time() - gen_t0:.1f}s")
    print(f"[Infer] Normalized text: {normalized_text}")

    # -------- save wav ----------
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(out_path), audio_out[0].cpu(), 44100)
    print(f"[Infer] Saved output to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
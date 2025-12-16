#!/usr/bin/env python
import os
import math
import json
import argparse
from pathlib import Path
from typing import List

import torchaudio
import whisper

# ------------ Config ------------

AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".ogg", ".flac", ".webm", ".aac", ".opus"}


def list_audio_files(root: Path) -> List[Path]:
    files: List[Path] = []
    for ext in AUDIO_EXTS:
        files.extend(root.rglob(f"*{ext}"))
    return sorted(files)


# ------------ 1) Split into 30s segments ------------

def prepare_segments(raw_dir: Path, segments_dir: Path, max_duration: float = 30.0) -> None:
    """
    raw_dir: folder with your long recordings
    segments_dir: where to save 30s chunks
    """
    segments_dir.mkdir(parents=True, exist_ok=True)

    files = list_audio_files(raw_dir)
    print(f"[Prepare] Found {len(files)} raw files in {raw_dir}")

    if not files:
        print("[Prepare] No files found, nothing to do.")
        return

    for idx, f in enumerate(files, 1):
        print(f"[Prepare] ({idx}/{len(files)}) {f.name}")

        audio, sr = torchaudio.load(str(f))

        # Convert to mono
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)

        # Resample to 44.1k (Echo / Fish AE default)
        if sr != 44100:
            audio = torchaudio.functional.resample(audio, sr, 44100)
            sr = 44100

        total_samples = audio.shape[1]
        chunk_samples = int(max_duration * sr)
        num_chunks = math.ceil(total_samples / chunk_samples)

        base = f.stem.replace(" ", "_")

        for ci in range(num_chunks):
            start = ci * chunk_samples
            end = min((ci + 1) * chunk_samples, total_samples)
            if end <= start:
                continue

            chunk = audio[:, start:end]
            out_name = f"{base}_{ci:04d}.wav"
            out_path = segments_dir / out_name
            torchaudio.save(str(out_path), chunk, sr)

    print(f"[Prepare] Done. Segments written to {segments_dir}")


# ------------ 2) Transcribe with Whisper large-v3 ------------

def transcribe_segments(
    segments_dir: Path,
    metadata_path: Path,
    lang_code: str | None = None,
    force: bool = False,
) -> None:
    """
    segments_dir: folder with 30s chunks
    metadata_path: output metadata.jsonl
    lang_code: e.g. "ar" or "en" (None = auto detect)
    force: if False and metadata exists, skip
    """
    if metadata_path.exists() and not force:
        print(f"[Transcribe] {metadata_path} already exists, skipping (use --force to redo).")
        return

    files = list_audio_files(segments_dir)
    print(f"[Transcribe] Found {len(files)} segments in {segments_dir}")

    if not files:
        print("[Transcribe] No segments found, nothing to do.")
        return

    print("[Transcribe] Loading Whisper model 'large-v3'...")
    model = whisper.load_model("large-v3")

    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    with metadata_path.open("w", encoding="utf-8") as fout:
        for i, f in enumerate(files, 1):
            print(f"[Transcribe] ({i}/{len(files)}) {f.name}")

            # IMPORTANT: do NOT pass batch_size → avoids DecodingOptions error
            result = model.transcribe(
                str(f),
                language=lang_code if lang_code else None,
                task="transcribe",
                verbose=False,
            )

            text = result.get("text", "").strip()
            if not text:
                continue

            # audio_relpath is relative to parent of "segments" folder
            # e.g. if segments_dir = runs/abu/ar/segments
            # audio_relpath = "segments/file.wav"
            audio_rel = os.path.relpath(str(f), start=str(segments_dir.parent))

            row = {
                "audio_relpath": audio_rel,
                "text": text,
                "language": result.get("language", lang_code or "unknown"),
            }
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"[Transcribe] Saved metadata to {metadata_path}")


# ------------ CLI entrypoint ------------

def main():
    parser = argparse.ArgumentParser(description="Echo-TTS data prep (segments + Whisper metadata)")

    parser.add_argument("--raw_dir", type=str, required=True,
                        help="Folder with original long audio (Windows H: becomes /mnt/h in WSL)")
    parser.add_argument("--out_root", type=str, required=True,
                        help="Root folder for this run (e.g. runs/echo_abu/ar)")

    parser.add_argument("--lang", type=str, default=None,
                        help="Language code for Whisper (e.g. 'ar', 'en'; leave empty for auto)")
    parser.add_argument("--max_duration", type=float, default=30.0,
                        help="Max chunk length in seconds (default 30)")
    parser.add_argument("--force", action="store_true",
                        help="Force re-transcription even if metadata.jsonl exists")
    parser.add_argument("--skip_prepare", action="store_true",
                        help="Skip segment creation (if already done)")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir).expanduser()
    out_root = Path(args.out_root).expanduser()

    segments_dir = out_root / "segments"
    metadata_path = out_root / "metadata.jsonl"

    print(f"[CLI] Raw dir:      {raw_dir}")
    print(f"[CLI] Segments dir: {segments_dir}")
    print(f"[CLI] Metadata:     {metadata_path}")
    print(f"[CLI] Language:     {args.lang or 'auto-detect'}")

    if not args.skip_prepare:
        print("\n[CLI] Step 1/2: preparing segments…")
        prepare_segments(raw_dir, segments_dir, max_duration=args.max_duration)
    else:
        print("\n[CLI] Skipping segment preparation (--skip_prepare)")

    print("\n[CLI] Step 2/2: transcribing with Whisper large-v3…")
    transcribe_segments(segments_dir, metadata_path, lang_code=args.lang, force=args.force)


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
from pathlib import Path

import datasets
import numpy as np
import soundfile as sf


def save_split_to_manifest(
    hf_dataset,
    out_audio_dir: Path,
    manifest_path: Path,
    max_samples: int | None = None,
):
    out_audio_dir.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as mf:
        saved = 0
        for ex in hf_dataset:
            if max_samples is not None and saved >= max_samples:
                break
            audio = ex["audio"]
            text = ex.get("text") or ""
            if not text:
                continue
            arr = np.asarray(audio["array"], dtype="float32")
            sr = int(audio["sampling_rate"])
            wav_path = out_audio_dir / f"ls_clean100_{saved:07d}.wav"
            sf.write(str(wav_path), arr, sr)
            duration = len(arr) / sr
            entry = {
                "audio_filepath": str(wav_path.resolve()),
                "text": text.strip(),
                "duration": float(duration),
                "language": "en",
            }
            mf.write(json.dumps(entry, ensure_ascii=False) + "\n")
            saved += 1


def main():
    out_root = Path("data/librispeech_clean_100")
    out_root.mkdir(parents=True, exist_ok=True)
    ds_train = datasets.load_dataset(
        "openslr/librispeech_asr",
        "clean",
        split="train.100",
    )
    ds_val = datasets.load_dataset(
        "openslr/librispeech_asr",
        "clean",
        split="validation",
    )
    save_split_to_manifest(
        ds_train,
        out_audio_dir=out_root / "wav_train",
        manifest_path=out_root / "train.jsonl",
        max_samples=None,
    )
    save_split_to_manifest(
        ds_val,
        out_audio_dir=out_root / "wav_val",
        manifest_path=out_root / "val.jsonl",
        max_samples=None,
    )


if __name__ == "__main__":
    main()

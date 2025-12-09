from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import tyro
from datasets import load_dataset


@dataclass
class EnBalanceConfig:
    base_train_manifest: Path = Path("data/multilingual_asr/train_70_30.jsonl")
    base_val_manifest: Path = Path("data/multilingual_asr/val_70_30.jsonl")

    out_train_manifest: Path = Path("data/multilingual_asr/train_70_30_en_balanced.jsonl")
    out_val_manifest: Path = Path("data/multilingual_asr/val_70_30_en_balanced.jsonl")

    audio_root: Path = Path("data/multilingual_asr/audio_en_extra")
    max_duration: float = 60.0
    min_duration: float = 0.1


def count_lang_examples(manifest: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    with manifest.open("r", encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line)
            lang = ex.get("language", "unknown")
            counts[lang] = counts.get(lang, 0) + 1
    return counts


def copy_manifest(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with src.open("r", encoding="utf-8") as fin, dst.open("w", encoding="utf-8") as fout:
        for line in fin:
            fout.write(line)


def add_librispeech_split(
    split: str,
    needed_samples: int,
    audio_dir: Path,
    out_manifest: Path,
    min_duration: float,
    max_duration: float,
) -> int:
    if needed_samples <= 0:
        return 0

    audio_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(
        "openslr/librispeech_asr",
        "clean",
        split=split,
        streaming=True,
    )

    added = 0
    with out_manifest.open("a", encoding="utf-8") as mf:
        for ex in ds:
            if added >= needed_samples:
                break

            text = (ex.get("text") or "").strip()
            if len(text) < 3:
                continue

            audio = ex["audio"]
            arr = np.asarray(audio["array"], dtype="float32")
            sr = int(audio["sampling_rate"])
            dur = float(len(arr) / sr)
            if dur < min_duration or dur > max_duration:
                continue

            wav_name = f"librispeech_{split}_{added:07d}.wav"
            wav_path = audio_dir / wav_name
            sf.write(str(wav_path), arr, sr)

            entry = {
                "audio_filepath": str(wav_path.resolve()),
                "text": text.lower(),
                "duration": dur,
                "language": "en",
            }
            mf.write(json.dumps(entry, ensure_ascii=False) + "\n")
            added += 1

    return added


def main(cfg: EnBalanceConfig) -> None:
    train_counts = count_lang_examples(cfg.base_train_manifest)
    val_counts = count_lang_examples(cfg.base_val_manifest)

    en_train = train_counts.get("en", 0)
    pl_train = train_counts.get("pl", 0)
    en_val = val_counts.get("en", 0)
    pl_val = val_counts.get("pl", 0)

    need_en_train = max(0, pl_train - en_train)
    need_en_val = max(0, pl_val - en_val)

    print(
        f"[BEFORE] train: en={en_train}, pl={pl_train} "
        f"(need +{need_en_train} en); "
        f"val: en={en_val}, pl={pl_val} (need +{need_en_val} en)"
    )

    copy_manifest(cfg.base_train_manifest, cfg.out_train_manifest)
    copy_manifest(cfg.base_val_manifest, cfg.out_val_manifest)

    added_train = add_librispeech_split(
        split="train.360",
        needed_samples=need_en_train,
        audio_dir=cfg.audio_root / "train",
        out_manifest=cfg.out_train_manifest,
        min_duration=cfg.min_duration,
        max_duration=cfg.max_duration,
    )

    added_val = add_librispeech_split(
        split="test",
        needed_samples=need_en_val,
        audio_dir=cfg.audio_root / "val",
        out_manifest=cfg.out_val_manifest,
        min_duration=cfg.min_duration,
        max_duration=cfg.max_duration,
    )

    new_train_counts = count_lang_examples(cfg.out_train_manifest)
    new_val_counts = count_lang_examples(cfg.out_val_manifest)

    print(f"[ADDED] librispeech train={added_train}, val={added_val}")
    print(
        f"[AFTER] train_en_balanced: en={new_train_counts.get('en', 0)}, "
        f"pl={new_train_counts.get('pl', 0)}"
    )
    print(
        f"[AFTER] val_en_balanced: en={new_val_counts.get('en', 0)}, "
        f"pl={new_val_counts.get('pl', 0)}"
    )


if __name__ == "__main__":
    cfg = tyro.cli(EnBalanceConfig)
    main(cfg)

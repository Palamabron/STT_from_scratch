from __future__ import annotations

import gc
import json
from dataclasses import dataclass
from pathlib import Path

import datasets
import numpy as np
import soundfile as sf
import tyro


def save_split_to_manifest(
    hf_dataset,
    out_audio_dir: Path,
    manifest_path: Path,
    language: str,
    max_samples: int | None = None,
    max_duration: float = 60.0,
):
    out_audio_dir.mkdir(parents=True, exist_ok=True)

    with manifest_path.open("w", encoding="utf-8") as mf:
        saved = 0
        skipped = 0
        for ex in hf_dataset:
            if max_samples is not None and saved >= max_samples:
                break
            audio = ex["audio"]
            text = ex.get("sentence") or ex.get("text") or ex.get("transcript") or ""
            if not text or len(text.strip()) < 3:
                skipped += 1
                continue
            arr = np.asarray(audio["array"], dtype="float32")
            sr = int(audio["sampling_rate"])
            duration = len(arr) / sr
            if duration < 0.1 or duration > max_duration:
                skipped += 1
                continue
            wav_path = out_audio_dir / f"{language}_{saved:07d}.wav"
            sf.write(str(wav_path), arr, sr)
            entry = {
                "audio_filepath": str(wav_path.resolve()),
                "text": text.strip().lower(),
                "duration": float(duration),
                "language": language,
            }
            mf.write(json.dumps(entry, ensure_ascii=False) + "\n")
            saved += 1
        print(f"[{language}] saved={saved}, skipped={skipped}, manifest={manifest_path}")


def merge_manifests(paths: list[Path], out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with out_path.open("w", encoding="utf-8") as out_f:
        for p in paths:
            if not p.exists():
                continue
            with p.open("r", encoding="utf-8") as in_f:
                for line in in_f:
                    out_f.write(line)
                    total += 1
    print(f"Merged {len(paths)} manifests into {out_path}, total_samples={total}")


def compute_stats(manifest_path: Path):
    total_dur = 0.0
    total = 0
    langs: dict[str, int] = {}
    if not manifest_path.exists():
        print(f"[STATS] {manifest_path}: file does not exist")
        return
    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line)
            total += 1
            total_dur += float(ex.get("duration", 0.0))
            lang = ex.get("language", "unknown")
            langs[lang] = langs.get(lang, 0) + 1
    print(f"[STATS] {manifest_path}: samples={total}, hours={total_dur / 3600:.1f}, langs={langs}")


def subsample_dataset(
    hf_dataset,
    frac: float | None,
    max_samples: int | None,
    seed: int = 42,
):
    ds = hf_dataset
    if frac is not None:
        n_total = len(ds)
        n_keep = int(n_total * frac)
        ds = ds.shuffle(seed=seed).select(range(n_keep))
    if max_samples is not None:
        n_total = len(ds)
        if n_total > max_samples:
            ds = ds.shuffle(seed=seed).select(range(max_samples))
    return ds


@dataclass
class PrepareConfig:
    rebuild_en: bool = True
    rebuild_pl: bool = True
    max_cv_pl: int = 50000
    max_duration: float = 60.0
    seed: int = 42


def main(cfg: PrepareConfig):
    root = Path("data/multilingual_asr")
    audio_root = root / "audio"
    root.mkdir(parents=True, exist_ok=True)

    en_train_manifest = root / "en_train.jsonl"
    en_val_manifest = root / "en_val.jsonl"
    pl_mls_train_manifest = root / "pl_mls_train.jsonl"
    pl_mls_val_manifest = root / "pl_mls_val.jsonl"
    pl_cv_manifest = root / "pl_cv_train.jsonl"

    if cfg.rebuild_en:
        ds_en_train = datasets.load_dataset(
            "openslr/librispeech_asr",
            "clean",
            split="train.100",
        )
        ds_en_val = datasets.load_dataset(
            "openslr/librispeech_asr",
            "clean",
            split="validation",
        )

        save_split_to_manifest(
            ds_en_train,
            out_audio_dir=audio_root / "en_train",
            manifest_path=en_train_manifest,
            language="en",
            max_samples=None,
            max_duration=cfg.max_duration,
        )
        save_split_to_manifest(
            ds_en_val,
            out_audio_dir=audio_root / "en_val",
            manifest_path=en_val_manifest,
            language="en",
            max_samples=None,
            max_duration=cfg.max_duration,
        )

        del ds_en_train, ds_en_val
        gc.collect()

    if cfg.rebuild_pl:
        ds_pl_mls_train = datasets.load_dataset(
            "facebook/multilingual_librispeech",
            "polish",
            split="train",
        )
        ds_pl_mls_dev = datasets.load_dataset(
            "facebook/multilingual_librispeech",
            "polish",
            split="dev",
        )

        save_split_to_manifest(
            ds_pl_mls_train,
            out_audio_dir=audio_root / "pl_mls_train",
            manifest_path=pl_mls_train_manifest,
            language="pl",
            max_samples=None,
            max_duration=cfg.max_duration,
        )
        save_split_to_manifest(
            ds_pl_mls_dev,
            out_audio_dir=audio_root / "pl_mls_val",
            manifest_path=pl_mls_val_manifest,
            language="pl",
            max_samples=None,
            max_duration=cfg.max_duration,
        )

        del ds_pl_mls_train, ds_pl_mls_dev
        gc.collect()

        ds_pl_cv_train = datasets.load_dataset(
            "fsicoli/common_voice_21_0",
            "pl",
            split="train",
        )
        ds_pl_cv_train = subsample_dataset(
            ds_pl_cv_train,
            frac=None,
            max_samples=cfg.max_cv_pl,
            seed=cfg.seed,
        )

        save_split_to_manifest(
            ds_pl_cv_train,
            out_audio_dir=audio_root / "pl_cv_train",
            manifest_path=pl_cv_manifest,
            language="pl",
            max_samples=None,
            max_duration=cfg.max_duration,
        )

        del ds_pl_cv_train
        gc.collect()

    train_manifest = root / "train.jsonl"
    val_manifest = root / "val.jsonl"

    merge_manifests(
        [en_train_manifest, pl_mls_train_manifest, pl_cv_manifest],
        train_manifest,
    )
    merge_manifests(
        [en_val_manifest, pl_mls_val_manifest],
        val_manifest,
    )

    compute_stats(train_manifest)
    compute_stats(val_manifest)


if __name__ == "__main__":
    cfg = tyro.cli(PrepareConfig)
    main(cfg)

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import soundfile as sf
import torchaudio
from loguru import logger


def duration_seconds(path: Path) -> float:
    try:
        meta = torchaudio.info(str(path))
        sr = getattr(meta, "sample_rate", 0) or 0
        nf = getattr(meta, "num_frames", 0) or 0
        if sr > 0 and nf > 0:
            return float(nf / sr)
    except Exception:
        pass

    with sf.SoundFile(str(path)) as f:
        if f.samplerate <= 0:
            return 0.0
        return float(f.frames / f.samplerate)


def _is_bad_duration(value) -> bool:
    if value is None:
        return True
    try:
        d = float(value)
    except Exception:
        return True
    return d <= 0.0


def check_manifest(path: Path) -> dict[str, int]:
    if not path.exists():
        raise FileNotFoundError(str(path))

    total = 0
    bad_duration = 0
    missing_duration_field = 0
    missing_files = 0

    with path.open("r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            total += 1
            item = json.loads(line)

            if "duration" not in item:
                missing_duration_field += 1
                bad_duration += 1
            else:
                if _is_bad_duration(item.get("duration")):
                    bad_duration += 1

            audio_fp = Path(item["audio_filepath"])
            if not audio_fp.exists():
                missing_files += 1

    return {
        "total": total,
        "bad_duration": bad_duration,
        "missing_duration_field": missing_duration_field,
        "missing_files": missing_files,
    }


def fix_manifest(path: Path, rewrite_all: bool = True) -> dict[str, int]:
    if not path.exists():
        raise FileNotFoundError(str(path))

    tmp_path = path.with_suffix(path.suffix + ".tmp")

    total = 0
    fixed = 0
    missing_files = 0
    errors = 0

    with path.open("r", encoding="utf-8") as fin, tmp_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue

            total += 1
            item = json.loads(line)

            audio_fp = Path(item["audio_filepath"])
            if not audio_fp.exists():
                missing_files += 1
                fout.write(json.dumps(item, ensure_ascii=False) + "\n")
                continue

            old = item.get("duration", None)
            should_fix = rewrite_all or _is_bad_duration(old)

            if should_fix:
                try:
                    item["duration"] = float(duration_seconds(audio_fp))
                    fixed += 1
                except Exception:
                    errors += 1

            fout.write(json.dumps(item, ensure_ascii=False) + "\n")

    os.replace(tmp_path, path)
    return {"total": total, "fixed": fixed, "missing_files": missing_files, "errors": errors}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_manifest", type=str, default="data/manifests/final/train_final.jsonl")
    ap.add_argument("--val_manifest", type=str, default="data/manifests/final/val_final.jsonl")

    ap.add_argument(
        "--check_only",
        action="store_true",
        help="Only check for bad durations (<=0 or not float). Do not rewrite manifests.",
    )

    ap.add_argument(
        "--rewrite_all",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Default: True. If False, fixes only bad durations (<=0 or not float).",
    )
    args = ap.parse_args()

    train_path = Path(args.train_manifest)
    val_path = Path(args.val_manifest)

    if args.check_only:
        train_stats = check_manifest(train_path)
        val_stats = check_manifest(val_path)
        logger.info("TRAIN CHECK: {}  path={}", train_stats, train_path)
        logger.info("VAL CHECK:   {}  path={}", val_stats, val_path)
        return

    train_stats = fix_manifest(train_path, rewrite_all=args.rewrite_all)
    val_stats = fix_manifest(val_path, rewrite_all=args.rewrite_all)
    logger.info("TRAIN FIX: {}  path={}", train_stats, train_path)
    logger.info("VAL FIX:   {}  path={}", val_stats, val_path)


if __name__ == "__main__":
    main()

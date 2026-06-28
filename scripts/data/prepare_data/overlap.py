from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


def entry_key(row: dict) -> str:
    uid = row.get("uid")
    if isinstance(uid, str) and uid:
        return uid
    audio_path = row.get("audio_filepath")
    if isinstance(audio_path, str) and audio_path:
        return audio_path
    raise ValueError(f"Manifest row missing uid/audio_filepath: {row}")


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def manifest_stats(path: Path) -> dict[str, Counter[str]]:
    langs: Counter[str] = Counter()
    datasets: Counter[str] = Counter()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            langs[str(row.get("language", "unknown"))] += 1
            datasets[str(row.get("dataset", "unknown"))] += 1
    return {"language": langs, "dataset": datasets}


def assert_no_overlap(train_path: Path, val_path: Path) -> None:
    train_keys = {entry_key(row) for row in read_jsonl(train_path)}
    val_keys = {entry_key(row) for row in read_jsonl(val_path)}
    overlap = train_keys & val_keys
    if overlap:
        raise RuntimeError(f"Train/val overlap detected: {len(overlap)} shared utterances")


def assert_val_has_no_train_bucket_names(val_path: Path, train_bucket_names: set[str]) -> None:
    stats = manifest_stats(val_path)
    bad = {name: count for name, count in stats["dataset"].items() if name in train_bucket_names}
    if bad:
        raise RuntimeError(f"Val manifest contains train bucket names: {bad}")

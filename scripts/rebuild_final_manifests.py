"""Rebuild train_final.jsonl and val_final.jsonl from individual bucket manifests.

Fixes mixed-source val buckets, caps oversampled buckets to data.yaml targets,
and guarantees no overlap between train and val finals.
"""

from __future__ import annotations

import json
import shutil
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import tyro
import yaml
from loguru import logger

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from prepare_data.buckets import UnderdeliveryPolicy, group_by_name  # noqa: E402
from prepare_data.config import AppConfig, DatasetSpec  # noqa: E402
from prepare_data.pipeline import (  # noqa: E402
    _build_final_from_buckets,
    build_train_final_from_buckets,
)


def _load_app_config(config_path: Path) -> AppConfig:
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    paths = raw["paths"]
    run = raw["run"]
    hf = raw.get("hf", {"token_env": "HF_TOKEN"})
    datasets_section = raw["datasets"]

    train_specs = [DatasetSpec(**item) for item in datasets_section.get("train", [])]
    val_specs = [DatasetSpec(**item) for item in datasets_section.get("val", [])]

    return AppConfig(
        paths=type(AppConfig().paths)(**{k: Path(v) for k, v in paths.items()}),
        run=type(AppConfig().run)(**run),
        hf=type(AppConfig().hf)(**hf),
        datasets=type(AppConfig().datasets)(train=train_specs, val=val_specs),
    )


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _entry_key(row: dict) -> str:
    uid = row.get("uid")
    if isinstance(uid, str) and uid:
        return uid
    audio_path = row.get("audio_filepath")
    if isinstance(audio_path, str) and audio_path:
        return audio_path
    raise ValueError(f"Manifest row missing uid/audio_filepath: {row}")


def split_mixed_noisy_val_bucket(individual_dir: Path) -> None:
    """Move CV15 validation rows out of bigos_pl_noisy_val into their own bucket."""
    noisy_path = individual_dir / "bigos_pl_noisy_val.jsonl"
    cv15_path = individual_dir / "bigos_pl_noisy_val_cv15.jsonl"
    if not noisy_path.exists():
        logger.warning("Missing {}, skipping noisy-val split", noisy_path)
        return

    rows = _read_jsonl(noisy_path)
    spont_rows: list[dict] = []
    cv15_rows: list[dict] = []

    for row in rows:
        audio_path = str(row.get("audio_filepath", ""))
        if "mozilla-common_voice_15-23__validation" in audio_path:
            fixed = dict(row)
            fixed["dataset"] = "bigos_pl_noisy_val_cv15"
            cv15_rows.append(fixed)
        else:
            fixed = dict(row)
            fixed["dataset"] = "bigos_pl_noisy_val"
            spont_rows.append(fixed)

    if not cv15_rows:
        logger.info("No CV15 rows found inside {}, leaving bucket unchanged", noisy_path.name)
        return

    _write_jsonl(noisy_path, spont_rows)
    existing_cv15 = _read_jsonl(cv15_path)
    merged_cv15 = existing_cv15 + cv15_rows
    deduped: dict[str, dict] = {}
    for row in merged_cv15:
        deduped[_entry_key(row)] = row
    _write_jsonl(cv15_path, list(deduped.values()))

    logger.info(
        "Split {} -> spont={} cv15={} (cv15 manifest total={})",
        noisy_path.name,
        len(spont_rows),
        len(cv15_rows),
        len(deduped),
    )


def _backup(path: Path, backup_dir: Path) -> None:
    if not path.exists():
        return
    backup_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, backup_dir / path.name)
    logger.info("Backed up {} -> {}", path, backup_dir / path.name)


def _manifest_stats(path: Path) -> dict[str, Counter[str]]:
    langs: Counter[str] = Counter()
    datasets: Counter[str] = Counter()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            langs[str(row.get("language", "unknown"))] += 1
            datasets[str(row.get("dataset", "unknown"))] += 1
    return {"language": langs, "dataset": datasets}


def _assert_no_overlap(train_path: Path, val_path: Path) -> None:
    train_keys = {_entry_key(row) for row in _read_jsonl(train_path)}
    val_keys = {_entry_key(row) for row in _read_jsonl(val_path)}
    overlap = train_keys & val_keys
    if overlap:
        raise RuntimeError(f"Train/val overlap detected: {len(overlap)} shared utterances")


def _assert_val_has_no_train_bucket_names(val_path: Path, train_bucket_names: set[str]) -> None:
    stats = _manifest_stats(val_path)
    bad = {name: count for name, count in stats["dataset"].items() if name in train_bucket_names}
    if bad:
        raise RuntimeError(f"Val manifest contains train bucket names: {bad}")


@dataclass
class RebuildConfig:
    config: Path = Path("configs/data.yaml")
    backup: bool = True


def main(cfg: RebuildConfig) -> None:
    app_config = _load_app_config(cfg.config.resolve())
    root_dir = app_config.paths.root_dir.resolve()
    individual_dir = (root_dir / app_config.paths.individual_manifests_dir).resolve()
    train_out = (root_dir / app_config.paths.final_train_manifest).resolve()
    val_out = (root_dir / app_config.paths.final_val_manifest).resolve()

    if cfg.backup:
        stamp = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
        backup_dir = train_out.parent / f"backup_{stamp}"
        _backup(train_out, backup_dir)
        _backup(val_out, backup_dir)

    split_mixed_noisy_val_bucket(individual_dir)

    build_train_final_from_buckets(
        bucket_specs=app_config.datasets.train,
        individual_manifests_dir=individual_dir,
        output_manifest=train_out,
    )
    _build_final_from_buckets(
        bucket_specs=app_config.datasets.val,
        individual_manifests_dir=individual_dir,
        output_manifest=val_out,
        underdelivery_policy=UnderdeliveryPolicy.WARN,
    )

    train_bucket_names = set(group_by_name(app_config.datasets.train))
    _assert_no_overlap(train_out, val_out)
    _assert_val_has_no_train_bucket_names(val_out, train_bucket_names)

    from fix_manifest_durations import fix_manifest

    for path in (train_out, val_out):
        stats = fix_manifest(path, rewrite_all=True)
        logger.info("Fixed durations in {}: {}", path.name, stats)

    for label, path in [("train", train_out), ("val", val_out)]:
        stats = _manifest_stats(path)
        total = sum(stats["language"].values())
        logger.info("[{}] {} utterances", label, total)
        logger.info("[{}] languages: {}", label, dict(stats["language"]))
        logger.info("[{}] datasets: {}", label, dict(stats["dataset"]))


if __name__ == "__main__":
    main(tyro.cli(RebuildConfig))

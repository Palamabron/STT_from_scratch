from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tyro
from loguru import logger


def _require_duration(item: dict[str, Any], line_no: int, manifest_path: Path) -> float:
    if "duration" not in item:
        raise KeyError(f"Missing 'duration' at {manifest_path}:{line_no}")
    try:
        d = float(item["duration"])
    except (TypeError, ValueError) as e:
        raise ValueError(
            f"Bad 'duration' (not float) at {manifest_path}:{line_no}: {item['duration']!r}"
        ) from e
    if d <= 0.0:
        raise ValueError(f"Bad 'duration' (<=0) at {manifest_path}:{line_no}: {d}")
    return d


def _require_dataset(item: dict[str, Any]) -> str:
    ds = item.get("dataset", "unknown")
    if not isinstance(ds, str) or not ds:
        return "unknown"
    return ds


@dataclass
class Summary:
    n: int = 0
    total_sec: float = 0.0

    @property
    def hours(self) -> float:
        return self.total_sec / 3600.0

    @property
    def avg_sec(self) -> float:
        return self.total_sec / max(1, self.n)


def summarize_manifest(path: Path) -> tuple[Summary, dict[str, Summary]]:
    if not path.exists():
        raise FileNotFoundError(str(path))

    total = Summary()
    by_dataset: dict[str, Summary] = {}

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            d = _require_duration(item, line_no, path)
            ds = _require_dataset(item)

            total.n += 1
            total.total_sec += d

            s = by_dataset.get(ds)
            if s is None:
                s = Summary()
                by_dataset[ds] = s
            s.n += 1
            s.total_sec += d

    return total, by_dataset


def _fmt(x: float, nd: int = 2) -> str:
    return f"{x:.{nd}f}"


@dataclass
class AnalyzeConfig:
    train_manifest: Path = Path("data/manifests/final/train_final.jsonl")
    val_manifest: Path = Path("data/manifests/final/val_final.jsonl")
    show_datasets: bool = True


def main(cfg: AnalyzeConfig) -> None:
    train_total, train_by_ds = summarize_manifest(cfg.train_manifest)
    val_total, val_by_ds = summarize_manifest(cfg.val_manifest)

    overall = Summary(
        n=train_total.n + val_total.n,
        total_sec=train_total.total_sec + val_total.total_sec,
    )

    logger.info(
        "TRAIN:   n={}  seconds={}  hours={}  avg_s={}  path={}",
        train_total.n,
        _fmt(train_total.total_sec),
        _fmt(train_total.hours),
        _fmt(train_total.avg_sec),
        cfg.train_manifest,
    )
    logger.info(
        "VAL:     n={}  seconds={}  hours={}  avg_s={}  path={}",
        val_total.n,
        _fmt(val_total.total_sec),
        _fmt(val_total.hours),
        _fmt(val_total.avg_sec),
        cfg.val_manifest,
    )
    logger.info(
        "OVERALL: n={}  seconds={}  hours={}  avg_s={}  manifests=train+val",
        overall.n,
        _fmt(overall.total_sec),
        _fmt(overall.hours),
        _fmt(overall.avg_sec),
    )

    if not cfg.show_datasets:
        return

    all_ds = sorted(set(train_by_ds.keys()) | set(val_by_ds.keys()))

    def get(d: dict[str, Summary], k: str) -> Summary:
        return d.get(k, Summary())

    logger.info("")
    logger.info("Per-dataset counts/hours:")
    logger.info("dataset | train_n | val_n | total_n | train_h | val_h | total_h")
    logger.info("------- | ------: | ----: | ------: | ------: | ----: | ------:")

    for ds in all_ds:
        tr = get(train_by_ds, ds)
        va = get(val_by_ds, ds)
        tot_n = tr.n + va.n
        tot_h = tr.hours + va.hours
        logger.info(
            "{} | {} | {} | {} | {} | {} | {}",
            ds,
            tr.n,
            va.n,
            tot_n,
            _fmt(tr.hours),
            _fmt(va.hours),
            _fmt(tot_h),
        )


if __name__ == "__main__":
    main(tyro.cli(AnalyzeConfig))

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import sentencepiece as spm
import tyro
from loguru import logger


@dataclass(slots=True)
class DatasetStatsConfig:
    train: str = "data/manifests/final/train_final.jsonl"
    val: str = "data/manifests/final/val_final.jsonl"
    tokenizer_model: str = "models/spm_unigram_2k_trainval.model"
    max_duration: float = 16.0
    output: str = "results/demo/dataset_stats.json"


def compute_stats_for_manifest(
    manifest_path: Path,
    processor: spm.SentencePieceProcessor,
    max_duration: float,
) -> dict[str, Any]:
    logger.info("Processing manifest: {}", manifest_path)

    # Raw counters
    total_entries = 0
    dropped_bad_duration = 0
    dropped_too_long = 0
    dropped_empty_text = 0

    # Valid entries lists for histograms and stats
    durations: list[float] = []
    char_lens: list[int] = []
    token_lens: list[int] = []

    # Language distribution
    lang_durations: dict[str, float] = {}
    lang_counts: dict[str, int] = {}

    # Dataset distribution
    dataset_counts: dict[str, int] = {}

    # For finding top/bottom 20 longest/shortest (by duration)
    # We will store a list of dicts: {'text': text, 'duration': duration, 'language': lang, 'audio_filepath': path}
    valid_entries: list[dict[str, Any]] = []

    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total_entries += 1
            entry = json.loads(line)

            # Duration parse
            duration_val = entry.get("duration")
            try:
                duration = float(duration_val) if duration_val is not None else None
            except (ValueError, TypeError):
                duration = None

            if duration is None or duration <= 0.0:
                dropped_bad_duration += 1
                continue

            if duration > max_duration:
                dropped_too_long += 1
                continue

            text = (entry.get("text") or "").strip()
            if not text:
                dropped_empty_text += 1
                continue

            token_ids = processor.encode(text, out_type=int)
            if not token_ids:
                dropped_empty_text += 1
                continue

            # Valid entry!
            durations.append(duration)
            char_lens.append(len(text))
            token_lens.append(len(token_ids))

            lang = str(entry.get("language", "unknown"))
            lang_durations[lang] = lang_durations.get(lang, 0.0) + duration
            lang_counts[lang] = lang_counts.get(lang, 0) + 1

            dataset_name = str(entry.get("dataset", "unknown"))
            dataset_counts[dataset_name] = dataset_counts.get(dataset_name, 0) + 1

            valid_entries.append(
                {
                    "audio_filepath": str(entry.get("audio_filepath", "")),
                    "text": text,
                    "duration": duration,
                    "language": lang,
                    "dataset": dataset_name,
                    "token_len": len(token_ids),
                }
            )

    # Sort valid entries to find top/bottom 20
    sorted_by_duration = sorted(valid_entries, key=lambda x: x["duration"])
    top_20_shortest = sorted_by_duration[:20]
    top_20_longest = sorted_by_duration[-20:][::-1]  # descending

    # Compute duration histogram
    # Bins from 0 to max_duration with step 0.5s
    duration_bins = np.arange(0, max_duration + 0.51, 0.5).tolist()
    duration_counts, _ = np.histogram(durations, bins=duration_bins)

    # Compute char length histogram (bins from 0 to 500, step 10)
    char_bins = np.arange(0, 501, 10).tolist()
    char_counts, _ = np.histogram(char_lens, bins=char_bins)

    # Compute token length histogram (bins from 0 to 100, step 2)
    token_bins = np.arange(0, 101, 2).tolist()
    token_counts, _ = np.histogram(token_lens, bins=token_bins)

    # Basic stats
    avg_duration = float(np.mean(durations)) if durations else 0.0
    total_duration_hours = float(sum(durations) / 3600.0)

    # Language stats
    languages_stats = {}
    for lang in sorted(lang_counts.keys()):
        languages_stats[lang] = {
            "count": int(lang_counts[lang]),
            "duration_hours": float(lang_durations[lang] / 3600.0),
        }

    return {
        "summary": {
            "total_raw_entries": total_entries,
            "loaded_entries": len(valid_entries),
            "total_duration_hours": total_duration_hours,
            "avg_duration_sec": avg_duration,
            "dropped_bad_duration": dropped_bad_duration,
            "dropped_too_long": dropped_too_long,
            "dropped_empty_text": dropped_empty_text,
            "pct_filtered_out": float((total_entries - len(valid_entries)) / total_entries * 100.0)
            if total_entries > 0
            else 0.0,
        },
        "languages": languages_stats,
        "datasets": dataset_counts,
        "histograms": {
            "duration": {"counts": duration_counts.tolist(), "bins": duration_bins},
            "char_len": {"counts": char_counts.tolist(), "bins": char_bins},
            "token_len": {"counts": token_counts.tolist(), "bins": token_bins},
        },
        "top_20_shortest": top_20_shortest,
        "top_20_longest": top_20_longest,
    }


def main(cfg: DatasetStatsConfig) -> None:
    processor = spm.SentencePieceProcessor()
    processor.load(cfg.tokenizer_model)
    logger.info("Loaded tokenizer model from: {}", cfg.tokenizer_model)

    train_path = Path(cfg.train)
    val_path = Path(cfg.val)

    stats = {
        "metadata": {"tokenizer_model": cfg.tokenizer_model, "max_duration": cfg.max_duration},
        "train": compute_stats_for_manifest(train_path, processor, cfg.max_duration),
        "val": compute_stats_for_manifest(val_path, processor, cfg.max_duration),
    }

    output_path = Path(cfg.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    logger.info("Saved dataset stats cache to: {}", cfg.output)


if __name__ == "__main__":
    tyro.cli(main)

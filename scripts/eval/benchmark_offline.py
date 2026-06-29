from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

import pandas as pd
import tyro
from loguru import logger


@dataclass(frozen=True)
class BenchmarkConfig:
    """Configuration for offline benchmark aggregation."""

    eval_dir: Path = Path("results/eval")
    output_dir: Path = Path("results/benchmark")


class ModelFile(TypedDict):
    file: str
    name: str


def load_and_normalize_eval_csv(path: Path, model_name: str) -> pd.DataFrame:
    """Loads evaluation CSV, filters validation split, and normalizes column data."""
    if not path.exists():
        logger.warning(f"File not found: {path}")
        return pd.DataFrame()

    df = pd.read_csv(path)
    df = df[df["split"].isin(["val", "full"])].copy()
    if df.empty:
        logger.warning(f"No validation rows found in {path}")
        return pd.DataFrame()

    df["model_id"] = model_name
    lang_map = {"all": "Overall (EN+PL)", "en": "English (EN)", "pl": "Polish (PL)"}
    df["language_name"] = df["language"].map(lang_map).fillna(df["language"])

    decode_map = {
        "greedy": "Greedy CTC Decode",
        "beam_kenlm": "Beam Search + KenLM 5-gram",
        "attention_greedy": "Greedy Attention Decode",
    }
    df["decode_mode_name"] = df["decode_type"].map(decode_map).fillna(df["decode_type"])

    df["wer_pct"] = df["wer"] * 100.0
    df["cer_pct"] = df["cer"] * 100.0
    return df


def main(cfg: BenchmarkConfig) -> None:
    """Aggregates and deduplicates offline evaluation results."""
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    configs: list[ModelFile] = [
        {"file": "ctc_attn_4090_65m_v9_val.csv", "name": "FastConformer CTC+Attn v9"},
        {"file": "ctc_4090_65m_v9_val.csv", "name": "FastConformer CTC v9"},
        {"file": "ctc_4090_65m_v8_val.csv", "name": "FastConformer CTC v8"},
    ]

    all_dfs = [load_and_normalize_eval_csv(cfg.eval_dir / c["file"], c["name"]) for c in configs]
    all_dfs = [df for df in all_dfs if not df.empty]

    if not all_dfs:
        logger.error("No evaluation data could be loaded!")
        return

    summary_df = pd.concat(all_dfs, ignore_index=True)
    summary_cols = [
        "model_id",
        "language_name",
        "decode_mode_name",
        "num_samples",
        "wer_pct",
        "cer_pct",
        "alpha",
        "beta",
    ]
    summary_df = summary_df[summary_cols].copy()

    summary_df = summary_df.sort_values(by="wer_pct").drop_duplicates(  # type: ignore[arg-type]
        subset=["model_id", "language_name", "decode_mode_name"], keep="first"
    )

    summary_df = summary_df.sort_values(
        by=["model_id", "language_name", "decode_mode_name"]
    ).reset_index(drop=True)

    summary_df.to_csv(cfg.output_dir / "offline_summary.csv", index=False)

    records = summary_df.to_dict(orient="records")
    with open(cfg.output_dir / "offline_summary.json", "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    logger.info(f"Benchmark aggregation complete in: {cfg.output_dir}")


if __name__ == "__main__":
    main(tyro.cli(BenchmarkConfig))

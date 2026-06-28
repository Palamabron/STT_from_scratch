from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from loguru import logger


def load_and_normalize_eval_csv(path: Path, model_name: str) -> pd.DataFrame:
    """Load evaluation CSV, filter validation split, and normalize names."""
    if not path.exists():
        logger.warning(f"File not found: {path}")
        return pd.DataFrame()

    df = pd.read_csv(path)

    # Check if the split column contains 'val' or 'full' (which acts as val)
    df = df[df["split"].isin(["val", "full"])]
    if df.empty:
        logger.warning(f"No validation rows found in {path}")
        return pd.DataFrame()

    df = df.copy()
    df["split"] = "val"
    df["model_id"] = model_name

    # Normalize language labels for readability
    lang_map = {"all": "Overall (EN+PL)", "en": "English (EN)", "pl": "Polish (PL)"}
    df["language_name"] = df["language"].map(lang_map).fillna(df["language"])

    # Normalize decode type labels for readability
    decode_map = {
        "greedy": "Greedy CTC Decode",
        "beam_kenlm": "Beam Search + KenLM 5-gram",
        "attention_greedy": "Greedy Attention Decode",
    }
    df["decode_mode_name"] = df["decode_type"].map(decode_map).fillna(df["decode_type"])

    # Convert WER and CER to percentages
    df["wer_pct"] = df["wer"] * 100.0
    df["cer_pct"] = df["cer"] * 100.0

    return df


def main() -> None:
    logger.info("Starting unified offline benchmark aggregation...")

    eval_dir = Path("results/eval")
    output_dir = Path("results/benchmark")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Configuration for files to aggregate
    configs = [
        {"file": "ctc_attn_4090_65m_v9_val.csv", "name": "FastConformer CTC+Attn v9"},
        {"file": "ctc_4090_65m_v9_val.csv", "name": "FastConformer CTC v9"},
        {"file": "ctc_4090_65m_v8_val.csv", "name": "FastConformer CTC v8"},
    ]

    all_dfs = []
    for cfg in configs:
        csv_path = eval_dir / cfg["file"]
        df = load_and_normalize_eval_csv(csv_path, cfg["name"])
        if not df.empty:
            all_dfs.append(df)

    if not all_dfs:
        logger.error("No evaluation data could be loaded!")
        return

    merged_df = pd.concat(all_dfs, ignore_index=True)

    # Columns to keep in the clean summary
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
    summary_df = merged_df[summary_cols].copy()

    # Deduplicate: keep only the lowest WER row for each model, language, and decode mode
    summary_df = summary_df.sort_values(by="wer_pct")
    summary_df = summary_df.drop_duplicates(
        subset=["model_id", "language_name", "decode_mode_name"], keep="first"
    )

    # Sort results for presentation (Model, then Language, then Decode Mode)
    summary_df = summary_df.sort_values(
        by=["model_id", "language_name", "decode_mode_name"]
    ).reset_index(drop=True)

    # Save to CSV
    csv_output = output_dir / "offline_summary.csv"
    summary_df.to_csv(csv_output, index=False)
    logger.info(f"Saved deduplicated aggregated offline summary CSV to: {csv_output}")

    # Save to JSON for easier Gradio parsing
    json_output = output_dir / "offline_summary.json"
    records = summary_df.to_dict(orient="records")
    with open(json_output, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved deduplicated aggregated offline summary JSON to: {json_output}")


if __name__ == "__main__":
    main()

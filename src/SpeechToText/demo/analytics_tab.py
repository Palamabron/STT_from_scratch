from __future__ import annotations

import json
from importlib import import_module
from pathlib import Path
from typing import Any, cast

import gradio as gr
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

try:
    _plot_benchmark = import_module("scripts.eval.plot_benchmark")
except ImportError:
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[3] / "scripts"))
    _plot_benchmark = import_module("eval.plot_benchmark")

plot_language_asymmetry = _plot_benchmark.plot_language_asymmetry
plot_wer_comparison = _plot_benchmark.plot_wer_comparison
plot_wer_vs_cer = _plot_benchmark.plot_wer_vs_cer
set_academic_style = _plot_benchmark.set_academic_style


# Path definitions
STATS_JSON_PATH = Path("results/demo/dataset_stats.json")
BENCHMARK_JSON_PATH = Path("results/benchmark/offline_summary.json")
BENCHMARK_CSV_PATH = Path("results/benchmark/offline_summary.csv")


def load_stats() -> dict[str, Any] | None:
    if STATS_JSON_PATH.exists():
        with open(STATS_JSON_PATH, encoding="utf-8") as f:
            return cast(dict[str, Any], json.load(f))
    return None


def load_benchmark_data() -> list[dict[str, Any]] | None:
    if BENCHMARK_JSON_PATH.exists():
        with open(BENCHMARK_JSON_PATH, encoding="utf-8") as f:
            return cast(list[dict[str, Any]], json.load(f))
    return None


def load_benchmark_df() -> pd.DataFrame | None:
    if BENCHMARK_CSV_PATH.exists():
        return pd.read_csv(BENCHMARK_CSV_PATH)
    return None


# Matplotlib plot generation for Dataset Analytics
def plot_dataset_duration_hist(stats: dict, split: str = "train") -> plt.Figure:
    """Draw a professional matplotlib bar chart from the cached duration histogram stats."""
    set_academic_style()
    hist_data = stats[split]["histograms"]["duration"]
    counts = hist_data["counts"]
    bins = hist_data["bins"]

    fig, ax = plt.subplots(figsize=(6, 3.5))

    # Reconstruct the bins centers and widths for drawing
    bin_centers = [(bins[i] + bins[i + 1]) / 2 for i in range(len(counts))]
    bin_width = bins[1] - bins[0]

    ax.bar(
        bin_centers,
        counts,
        width=bin_width * 0.9,
        color="#34495e" if split == "train" else "#16a085",
        edgecolor="black",
        linewidth=0.5,
        alpha=0.85,
    )

    ax.set_title(f"Audio Duration Distribution ({split.upper()} set)", pad=10)
    ax.set_xlabel("Duration (seconds)", labelpad=5)
    ax.set_ylabel("Utterance Count", labelpad=5)
    sns.despine()
    fig.tight_layout()
    return fig


def plot_dataset_languages(stats: dict, split: str = "train") -> plt.Figure:
    """Draw a professional pie chart of the language hour distribution."""
    set_academic_style()
    lang_data = stats[split]["languages"]

    languages = list(lang_data.keys())
    hours = [lang_data[lang]["duration_hours"] for lang in languages]
    counts = [lang_data[lang]["count"] for lang in languages]

    fig, ax = plt.subplots(figsize=(5, 3.5))

    # Beautiful academic colors
    colors = ["#4a7c59", "#b23b3b", "#3d5a80", "#ee6c4d"]

    wedges, texts, autotexts = ax.pie(
        hours,
        labels=[f"{lang.upper()}\n({c:,} utts)" for lang, c in zip(languages, counts, strict=True)],
        autopct="%1.1f%%",
        startangle=90,
        colors=colors[: len(languages)],
        wedgeprops=dict(width=0.6, edgecolor="black", linewidth=0.5),  # donut chart
        pctdistance=0.75,
    )

    # Make percentages bold
    for autotext in autotexts:
        autotext.set_fontweight("bold")
        autotext.set_fontsize(10)

    ax.set_title(f"Language Distribution by Hours ({split.upper()} set)", pad=10)
    fig.tight_layout()
    return fig


def plot_dataset_token_len_hist(stats: dict, split: str = "train") -> plt.Figure:
    """Draw a professional matplotlib bar chart of the token length distribution."""
    set_academic_style()
    hist_data = stats[split]["histograms"]["token_len"]
    counts = hist_data["counts"]
    bins = hist_data["bins"]

    fig, ax = plt.subplots(figsize=(6, 3.5))

    bin_centers = [(bins[i] + bins[i + 1]) / 2 for i in range(len(counts))]
    bin_width = bins[1] - bins[0]

    ax.bar(
        bin_centers,
        counts,
        width=bin_width * 0.9,
        color="#2980b9" if split == "train" else "#8e44ad",
        edgecolor="black",
        linewidth=0.5,
        alpha=0.85,
    )

    ax.set_title(f"SentencePiece Token Count Distribution ({split.upper()} set)", pad=10)
    ax.set_xlabel("Token Length (units)", labelpad=5)
    ax.set_ylabel("Utterance Count", labelpad=5)
    sns.despine()
    fig.tight_layout()
    return fig


def get_dataset_summary_markdown(stats: dict, split: str = "train") -> str:
    summary = stats[split]["summary"]
    return f"""### Dataset Stats for **{split.upper()}** (Filtering threshold: `{stats["metadata"]["max_duration"]} s`):
- **Total raw entries:** {summary["total_raw_entries"]:,}
- **Loaded after filtering:** {summary["loaded_entries"]:,} (`{100 - summary["pct_filtered_out"]:.2f}%` of the split)
- **Total audio duration:** {summary["total_duration_hours"]:.2f} hours
- **Average utterance duration:** {summary["avg_duration_sec"]:.2f} seconds
- **Filtered (too long):** {summary["dropped_too_long"]:,} ({summary["pct_filtered_out"]:.2f}%)
- **Invalid / Empty transcripts:** {summary["dropped_bad_duration"] + summary["dropped_empty_text"]:,}
"""


# Main function to create Gradio tab layout
def create_analytics_tab() -> gr.Blocks:
    stats_cache = load_stats()
    benchmark_data = load_benchmark_data()
    benchmark_df = load_benchmark_df()

    with gr.Blocks() as tab:
        gr.Markdown(
            "## 📊 ASR Analytics & Offline Benchmark\n"
            "This tab presents advanced statistics of the training corpus (EN/PL) "
            "and evaluation results of Conformer-CTC and Hybrid-Attention models on the validation split."
        )

        with gr.Tab("Dataset Analytics"):
            if stats_cache is None:
                gr.Markdown(
                    "⚠️ Dataset statistics cache not found! Please run `scripts/generate_dataset_stats.py`."
                )
            else:
                with gr.Row():
                    with gr.Column(scale=1):
                        split_selector = gr.Radio(
                            choices=["train", "val"],
                            value="train",
                            label="Select Split",
                        )
                        summary_box = gr.Markdown(
                            value=get_dataset_summary_markdown(stats_cache, "train")
                        )

                    with gr.Column(scale=2):
                        with gr.Row():
                            pie_plot = gr.Plot(value=plot_dataset_languages(stats_cache, "train"))
                            duration_plot = gr.Plot(
                                value=plot_dataset_duration_hist(stats_cache, "train")
                            )
                        with gr.Row():
                            token_plot = gr.Plot(
                                value=plot_dataset_token_len_hist(stats_cache, "train")
                            )

                # Dynamic updates on Radio click
                def update_dataset_stats(
                    split: str,
                ) -> tuple[str, plt.Figure, plt.Figure, plt.Figure]:
                    return (
                        get_dataset_summary_markdown(stats_cache, split),
                        plot_dataset_languages(stats_cache, split),
                        plot_dataset_duration_hist(stats_cache, split),
                        plot_dataset_token_len_hist(stats_cache, split),
                    )

                split_selector.change(
                    update_dataset_stats,
                    inputs=[split_selector],
                    outputs=[summary_box, pie_plot, duration_plot, token_plot],
                )

        with gr.Tab("Offline Benchmark"):
            if benchmark_data is None or benchmark_df is None:
                gr.Markdown(
                    "⚠️ Evaluation results not found in `results/benchmark/`! Please run `scripts/benchmark_offline.py` first."
                )
            else:
                gr.Markdown("### 🏆 Model Performance Comparison (WER / CER %)")

                # Render results in a nice Gradio Dataframe
                gr.DataFrame(
                    value=benchmark_df,
                    interactive=False,
                    wrap=True,
                )

                gr.Markdown("### 📈 Publication-Ready Comparison Plots")
                with gr.Row():
                    gr.Plot(value=plot_wer_comparison(benchmark_data))
                    gr.Plot(value=plot_language_asymmetry(benchmark_data))
                with gr.Row():
                    gr.Plot(value=plot_wer_vs_cer(benchmark_data))

    return cast(gr.Blocks, tab)

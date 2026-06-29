from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def set_academic_style() -> None:
    """Set aesthetic, publication-ready style defaults."""
    sns.set_theme(style="whitegrid")
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "Liberation Serif"],
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9,
            "figure.titlesize": 13,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )


def get_dataframe(data: list[dict] | pd.DataFrame) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        return data
    return pd.DataFrame(data)


def plot_wer_comparison(data: list[dict] | pd.DataFrame) -> plt.Figure:
    """Grouped bar plot comparing WER across models and decode modes on the Overall split."""
    df = get_dataframe(data)

    
    df_overall = df[df["language_name"] == "Overall (EN+PL)"].copy()

    fig, ax = plt.subplots(figsize=(8, 5))

    sns.barplot(
        data=df_overall,
        x="model_id",
        y="wer_pct",
        hue="decode_mode_name",
        palette="Blues_d",
        ax=ax,
        edgecolor="black",
        linewidth=0.7,
    )

    
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(
                f"{height:.2f}%",
                (p.get_x() + p.get_width() / 2.0, height),
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
                xytext=(0, 3),
                textcoords="offset points",
            )

    ax.set_title("Word Error Rate (WER) Comparison on Overall Split (EN+PL)", pad=15)
    ax.set_xlabel("Acoustic Model / Pipeline Architecture", labelpad=10)
    ax.set_ylabel("WER (%)", labelpad=10)
    ax.set_ylim(0, max(df_overall["wer_pct"]) * 1.15)
    ax.legend(title="Decoding Algorithm", frameon=True, facecolor="white", edgecolor="0.8")

    sns.despine()
    fig.tight_layout()
    return fig


def plot_language_asymmetry(data: list[dict] | pd.DataFrame) -> plt.Figure:
    """Plot comparing Polish (PL) vs English (EN) showing language complexity/flexion impact."""
    df = get_dataframe(data)

    
    df_lang = df[
        (df["language_name"].isin(["English (EN)", "Polish (PL)"]))
        & (df["decode_mode_name"] == "Beam Search + KenLM 5-gram")
    ].copy()

    fig, ax = plt.subplots(figsize=(7, 5))

    
    colors = {"English (EN)": "

    sns.barplot(
        data=df_lang,
        x="model_id",
        y="wer_pct",
        hue="language_name",
        palette=colors,
        ax=ax,
        edgecolor="black",
        linewidth=0.7,
    )

    
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(
                f"{height:.2f}%",
                (p.get_x() + p.get_width() / 2.0, height),
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
                xytext=(0, 3),
                textcoords="offset points",
            )

    ax.set_title(
        "Language Complexity Impact: English (EN) vs Polish (PL)\n(Beam Search + KenLM 5-gram Decoding)",
        pad=15,
    )
    ax.set_xlabel("Acoustic Model / Pipeline Architecture", labelpad=10)
    ax.set_ylabel("WER (%)", labelpad=10)
    ax.set_ylim(0, max(df_lang["wer_pct"]) * 1.15)
    ax.legend(title="Target Language", frameon=True, facecolor="white", edgecolor="0.8")

    
    ax.text(
        0.5,
        -0.22,
        "Note: Polish exhibits significantly higher WER (24-27%) compared to English (13-15%)\ndue to rich morphology, grammatical flexion, and a larger vocabulary state space.",
        transform=ax.transAxes,
        ha="center",
        fontsize=9,
        style="italic",
        bbox=dict(facecolor="
    )

    sns.despine()
    fig.tight_layout()
    return fig


def plot_wer_vs_cer(data: list[dict] | pd.DataFrame) -> plt.Figure:
    """Plot demonstrating the correlation between WER and CER across all runs."""
    df = get_dataframe(data)

    fig, ax = plt.subplots(figsize=(7, 5))

    
    sns.scatterplot(
        data=df,
        x="cer_pct",
        y="wer_pct",
        hue="language_name",
        style="model_id",
        s=120,
        ax=ax,
        edgecolor="black",
        alpha=0.85,
    )

    
    if not df.empty:
        import numpy as np

        x = df["cer_pct"].values
        y = df["wer_pct"].values
        m, c = np.polyfit(x, y, 1)
        x_range = np.linspace(min(x) * 0.9, max(x) * 1.1, 100)
        ax.plot(
            x_range, m * x_range + c, color="gray", linestyle="--", alpha=0.5, label="Reference Fit"
        )

    ax.set_title("WER vs CER Correlation Across All Models & Splits", pad=15)
    ax.set_xlabel("Character Error Rate (CER %)", labelpad=10)
    ax.set_ylabel("Word Error Rate (WER %)", labelpad=10)
    ax.legend(frameon=True, facecolor="white", edgecolor="0.8")

    sns.despine()
    fig.tight_layout()
    return fig


def main() -> None:
    set_academic_style()

    summary_path = Path("results/benchmark/offline_summary.json")
    if not summary_path.exists():
        print(
            f"Error: Summary file {summary_path} does not exist. Run scripts/benchmark_offline.py first."
        )
        return

    with open(summary_path, encoding="utf-8") as f:
        data = json.load(f)

    
    fig_dir = Path("results/figures")
    fig_dir.mkdir(parents=True, exist_ok=True)

    
    fig1 = plot_wer_comparison(data)
    fig1.savefig(fig_dir / "wer_comparison.png", dpi=300)
    plt.close(fig1)

    
    fig2 = plot_language_asymmetry(data)
    fig2.savefig(fig_dir / "language_asymmetry.png", dpi=300)
    plt.close(fig2)

    
    fig3 = plot_wer_vs_cer(data)
    fig3.savefig(fig_dir / "wer_vs_cer.png", dpi=300)
    plt.close(fig3)

    print(f"Successfully generated and saved academic plots to: {fig_dir.absolute()}")


if __name__ == "__main__":
    main()

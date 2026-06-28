from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import torch

from SpeechToText.demo.analytics_tab import (
    plot_dataset_duration_hist,
    plot_dataset_languages,
    plot_dataset_token_len_hist,
)
from SpeechToText.demo.app import build_app

try:
    from scripts.eval.plot_benchmark import (
        plot_language_asymmetry,
        plot_wer_comparison,
        plot_wer_vs_cer,
    )
except ImportError:
    try:
        from scripts.eval.plot_benchmark import (
            plot_language_asymmetry,
            plot_wer_comparison,
            plot_wer_vs_cer,
        )
    except ImportError:
        import pathlib
        import sys

        sys.path.append(str(pathlib.Path(__file__).parent.parent.parent / "scripts/eval"))
        from plot_benchmark import plot_language_asymmetry, plot_wer_comparison, plot_wer_vs_cer


from SpeechToText.demo.transcribe_logic import normalize_stream_audio


def test_build_gradio_app() -> None:
    """Verify that the Gradio app blocks construct without exceptions."""
    app = build_app()
    assert app is not None


def test_benchmark_plotting_functions() -> None:
    """Verify that the core matplotlib plotting functions work on sample/mock data."""
    mock_data = [
        {
            "model_id": "FastConformer CTC v9",
            "language_name": "Overall (EN+PL)",
            "decode_mode_name": "Beam Search + KenLM 5-gram",
            "num_samples": 10545,
            "wer_pct": 19.89,
            "cer_pct": 8.77,
            "alpha": 0.5,
            "beta": 1.5,
        },
        {
            "model_id": "FastConformer CTC+Attn v9",
            "language_name": "Overall (EN+PL)",
            "decode_mode_name": "Beam Search + KenLM 5-gram",
            "num_samples": 10545,
            "wer_pct": 19.72,
            "cer_pct": 8.66,
            "alpha": 0.5,
            "beta": 1.5,
        },
        {
            "model_id": "FastConformer CTC+Attn v9",
            "language_name": "English (EN)",
            "decode_mode_name": "Beam Search + KenLM 5-gram",
            "num_samples": 4850,
            "wer_pct": 13.54,
            "cer_pct": 7.00,
            "alpha": 0.5,
            "beta": 1.5,
        },
        {
            "model_id": "FastConformer CTC+Attn v9",
            "language_name": "Polish (PL)",
            "decode_mode_name": "Beam Search + KenLM 5-gram",
            "num_samples": 5695,
            "wer_pct": 27.18,
            "cer_pct": 10.28,
            "alpha": 0.5,
            "beta": 1.5,
        },
    ]

    df = pd.DataFrame(mock_data)

    # Test plot_wer_comparison
    fig1 = plot_wer_comparison(df)
    assert isinstance(fig1, plt.Figure)
    plt.close(fig1)

    # Test plot_language_asymmetry
    fig2 = plot_language_asymmetry(df)
    assert isinstance(fig2, plt.Figure)
    plt.close(fig2)

    # Test plot_wer_vs_cer
    fig3 = plot_wer_vs_cer(df)
    assert isinstance(fig3, plt.Figure)
    plt.close(fig3)


def test_normalize_stream_audio_resamples_to_16khz() -> None:
    samples = torch.linspace(-1.0, 1.0, 4410)
    resampled = normalize_stream_audio(44_100, samples)
    assert resampled.dim() == 1
    assert resampled.numel() == 1600


def test_dataset_analytics_plotting_functions() -> None:
    """Verify dataset stats plots can render with mock dataset_stats."""
    mock_stats = {
        "metadata": {"max_duration": 16.0},
        "train": {
            "summary": {
                "total_raw_entries": 100,
                "loaded_entries": 90,
                "total_duration_hours": 1.5,
                "avg_duration_sec": 10.0,
                "dropped_bad_duration": 0,
                "dropped_too_long": 10,
                "dropped_empty_text": 0,
                "pct_filtered_out": 10.0,
            },
            "languages": {
                "en": {"count": 50, "duration_hours": 0.8},
                "pl": {"count": 40, "duration_hours": 0.7},
            },
            "histograms": {
                "duration": {"counts": [10, 20, 30], "bins": [0.0, 5.0, 10.0, 15.0]},
                "token_len": {"counts": [5, 15, 25], "bins": [0, 10, 20, 30]},
            },
        },
    }

    fig1 = plot_dataset_duration_hist(mock_stats, "train")
    assert isinstance(fig1, plt.Figure)
    plt.close(fig1)

    fig2 = plot_dataset_languages(mock_stats, "train")
    assert isinstance(fig2, plt.Figure)
    plt.close(fig2)

    fig3 = plot_dataset_token_len_hist(mock_stats, "train")
    assert isinstance(fig3, plt.Figure)
    plt.close(fig3)

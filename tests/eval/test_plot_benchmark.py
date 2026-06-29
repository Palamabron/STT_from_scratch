from __future__ import annotations

import pandas as pd
import pytest

from eval.plot_benchmark import _language_asymmetry_note, _ylim_max, plot_wer_comparison


def test_ylim_max_handles_empty_series() -> None:
    assert _ylim_max(pd.Series(dtype=float)) == 100.0


def test_language_asymmetry_note_uses_data_ranges() -> None:
    df = pd.DataFrame(
        {
            "language_name": ["English (EN)", "Polish (PL)"],
            "wer_pct": [13.5, 25.0],
        }
    )
    note = _language_asymmetry_note(df)
    assert "13.5-13.5%" in note
    assert "25.0-25.0%" in note


def test_plot_wer_comparison_requires_overall_rows() -> None:
    with pytest.raises(ValueError, match="Overall"):
        plot_wer_comparison(
            [
                {
                    "model_id": "m",
                    "language_name": "English (EN)",
                    "decode_mode_name": "Greedy CTC Decode",
                    "wer_pct": 10.0,
                }
            ]
        )

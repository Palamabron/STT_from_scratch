from __future__ import annotations

from pathlib import Path

import torch

from SpeechToText.augmentation import load_audio_bank


def test_load_audio_bank_from_pt(tmp_path: Path) -> None:
    clip = torch.randn(16_000)
    bank_path = tmp_path / "noise_bank.pt"
    torch.save((clip,), str(bank_path))

    loaded = load_audio_bank(str(bank_path), sample_rate=16_000, min_len_sec=0.5)
    assert loaded is not None
    assert len(loaded) == 1
    assert loaded[0].shape == clip.shape


def test_load_audio_bank_empty_pt_returns_none(tmp_path: Path) -> None:
    bank_path = tmp_path / "empty_bank.pt"
    torch.save(tuple(), str(bank_path))

    assert load_audio_bank(str(bank_path), sample_rate=16_000) is None

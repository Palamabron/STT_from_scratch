from __future__ import annotations

from pathlib import Path

import pytest
from sentencepiece import SentencePieceProcessor

from SpeechToText.models.tdt.config import TrainConfig
from SpeechToText.models.tdt.lit import LitFastConformerTDT

REPO_ROOT = Path(__file__).resolve().parents[1]
SPM_PATH = REPO_ROOT / "models/spm_unigram_4k_trainval.model"


def _load_sp() -> SentencePieceProcessor:
    if not SPM_PATH.is_file():
        pytest.skip(f"SentencePiece model not found: {SPM_PATH}")
    sp = SentencePieceProcessor()
    sp.load(str(SPM_PATH))
    return sp


def test_tdt_duration_head_follows_use_tdt_config() -> None:
    sp = _load_sp()
    vocab_size = int(sp.get_piece_size()) + 1

    rnnt_lit = LitFastConformerTDT(TrainConfig(use_tdt=False), sp=sp, vocab_size=vocab_size)
    assert rnnt_lit.net.joint.duration_out is None

    tdt_lit = LitFastConformerTDT(TrainConfig(use_tdt=True), sp=sp, vocab_size=vocab_size)
    assert tdt_lit.net.joint.duration_out is not None

from __future__ import annotations

from pathlib import Path

import pytest
from sentencepiece import SentencePieceProcessor

from SpeechToText.models.ctc_attention.config import TrainConfig
from SpeechToText.models.ctc_attention.lit import LitFastConformerCTCAttention

REPO_ROOT = Path(__file__).resolve().parents[1]
SPM_PATH = REPO_ROOT / "models/spm_unigram_2k_trainval.model"


def _require_spm() -> SentencePieceProcessor:
    if not SPM_PATH.is_file():
        pytest.skip(f"SentencePiece model not found: {SPM_PATH}")
    sp = SentencePieceProcessor()
    sp.load(str(SPM_PATH))
    return sp


def _tiny_lit(**overrides: object) -> LitFastConformerCTCAttention:
    config = TrainConfig()
    config.model.encoder.d_model = 64
    config.model.encoder.n_layers = 1
    config.model.encoder.n_heads = 2
    config.model.aux_layer = 0
    for key, value in overrides.items():
        setattr(config, key, value)
    return LitFastConformerCTCAttention(config, sp=_require_spm())


def test_decoder_warmup_stage_freezes_encoder_and_ctc() -> None:
    lit = _tiny_lit(decoder_warmup_epochs=5)
    stage = lit._training_stage_for_epoch(2)
    assert stage.name == "decoder_warmup"
    assert stage.effective_ctc_weight == 0.0
    assert stage.effective_aux_ctc_weight == 0.0
    assert stage.include_attn is True
    assert stage.freeze_encoder is True
    assert stage.freeze_ctc_heads is True
    assert stage.freeze_decoder is False


def test_ctc_calibration_stage_trains_ctc_only() -> None:
    lit = _tiny_lit(ctc_calibration_epochs=5, aux_ctc_weight=0.3)
    stage = lit._training_stage_for_epoch(1)
    assert stage.name == "ctc_calibration"
    assert stage.effective_ctc_weight == 1.0
    assert stage.include_attn is False
    assert stage.freeze_encoder is True
    assert stage.freeze_ctc_heads is False
    assert stage.freeze_decoder is True


def test_joint_stage_uses_config_weights() -> None:
    lit = _tiny_lit(ctc_weight=0.3, aux_ctc_weight=0.2)
    stage = lit._training_stage_for_epoch(10)
    assert stage.name == "joint"
    assert stage.effective_ctc_weight == pytest.approx(0.3)
    assert stage.effective_aux_ctc_weight == pytest.approx(0.2)
    assert stage.include_attn is True


def test_apply_training_stage_sets_requires_grad() -> None:
    lit = _tiny_lit(decoder_warmup_epochs=5)
    stage = lit._training_stage_for_epoch(0)
    lit._apply_training_stage(stage)
    assert any(p.requires_grad for p in lit.net.decoder.parameters())
    assert not any(p.requires_grad for p in lit.net.encoder.parameters())
    assert not any(p.requires_grad for p in lit.net.ctc_proj.parameters())


def test_ctc_calibration_stage_disables_attention_in_loss() -> None:
    lit = _tiny_lit(ctc_calibration_epochs=5, ctc_weight=0.3, aux_ctc_weight=0.2)
    stage = lit._training_stage_for_epoch(1)
    assert stage.include_attn is False
    assert stage.effective_ctc_weight == 1.0
    assert stage.effective_aux_ctc_weight == pytest.approx(0.2)

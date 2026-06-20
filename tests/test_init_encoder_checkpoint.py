from __future__ import annotations

import sys
from pathlib import Path

import lightning.pytorch as pl
import pytest
import torch
from sentencepiece import SentencePieceProcessor

from SpeechToText.models.common.checkpoint_io import load_lightning_checkpoint
from SpeechToText.models.common.inference import load_lit_module
from SpeechToText.models.ctc.config import TrainConfig as CtcTrainConfig
from SpeechToText.models.ctc.lit import LitFastConformerCTC

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
SPM_PATH = REPO_ROOT / "models/spm_unigram_4k_trainval.model"


def _require_spm() -> SentencePieceProcessor:
    if not SPM_PATH.is_file():
        pytest.skip(f"SentencePiece model not found: {SPM_PATH}")
    sp = SentencePieceProcessor()
    sp.load(str(SPM_PATH))
    return sp


def _write_minimal_ctc_checkpoint(path: Path, sp: SentencePieceProcessor) -> None:
    config = CtcTrainConfig()
    config.model.encoder.d_model = 64
    config.model.encoder.n_layers = 1
    config.model.encoder.n_heads = 2

    lit = LitFastConformerCTC(config, sp=sp)
    torch.save(
        {
            "epoch": 1,
            "global_step": 10,
            "state_dict": lit.state_dict(),
            "hyper_parameters": {"config": config},
            "pytorch-lightning-version": pl.__version__,
        },
        path,
    )


def test_init_encoder_checkpoint_from_synthetic_ctc(tmp_path: Path) -> None:
    sp = _require_spm()
    source_ckpt = tmp_path / "source_ctc.ckpt"
    _write_minimal_ctc_checkpoint(source_ckpt, sp)

    sys.path.insert(0, str(SCRIPTS_DIR))
    from init_encoder_from_checkpoint import InitEncoderConfig, main

    output = tmp_path / "init_rnnt.ckpt"
    main(
        InitEncoderConfig(
            source_checkpoint=str(source_ckpt),
            tokenizer_model=str(SPM_PATH),
            target="rnnt",
            output=str(output),
        )
    )

    ckpt = load_lightning_checkpoint(output)
    assert "state_dict" in ckpt
    assert "hyper_parameters" in ckpt
    assert ckpt["hyper_parameters"]["config"] is not None

    module, model_type = load_lit_module(str(output), sp=sp, model_type="tdt")
    assert model_type == "tdt"
    assert any(key.startswith("net.encoder.") for key in module.state_dict())

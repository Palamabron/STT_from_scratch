from __future__ import annotations

import sys
from pathlib import Path

import pytest
from sentencepiece import SentencePieceProcessor

from SpeechToText.models.common.checkpoint_io import load_lightning_checkpoint
from SpeechToText.models.common.inference import load_lit_module

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"


def test_init_encoder_checkpoint_is_lightning_compatible(tmp_path: Path) -> None:
    spm_path = REPO_ROOT / "models/spm_unigram_4k_trainval.model"
    ctc_ckpt = REPO_ROOT / "checkpoints/integration/ctc/last.ckpt"
    if not spm_path.is_file() or not ctc_ckpt.is_file():
        pytest.skip("integration artifacts not present")

    sys.path.insert(0, str(SCRIPTS_DIR))
    from init_encoder_from_checkpoint import InitEncoderConfig, main

    output = tmp_path / "init_rnnt.ckpt"
    main(
        InitEncoderConfig(
            source_checkpoint=str(ctc_ckpt),
            tokenizer_model=str(spm_path),
            target="rnnt",
            output=str(output),
        )
    )

    ckpt = load_lightning_checkpoint(output)
    assert "state_dict" in ckpt
    assert "hyper_parameters" in ckpt

    sp = SentencePieceProcessor()
    sp.load(str(spm_path))
    module, model_type = load_lit_module(str(output), sp=sp, model_type="tdt")
    assert model_type == "tdt"
    assert any(key.startswith("net.encoder.") for key in module.state_dict())

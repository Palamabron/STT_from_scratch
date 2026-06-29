from __future__ import annotations

import pytest

from SpeechToText.models.common.inference import detect_model_type


def test_detect_model_type_shared_checkpoint_keys() -> None:
    state = {
        "net.ctc_proj.weight": object(),
        "net.tdt_joint.enc_proj.weight": object(),
        "net.tdt_decoder.embed.weight": object(),
        "net.decoder.layers.0.self_attn.in_proj_weight": object(),
    }
    ckpt = {"state_dict": state}

    import SpeechToText.models.common.inference as inference

    original = inference.load_lightning_checkpoint
    inference.load_lightning_checkpoint = lambda _path: ckpt
    try:
        assert detect_model_type("dummy.ckpt") == "shared"
    finally:
        inference.load_lightning_checkpoint = original


def test_detect_model_type_standalone_tdt() -> None:
    state = {
        "net.joint.enc_proj.weight": object(),
        "net.decoder.embed.weight": object(),
    }
    ckpt = {"state_dict": state}

    import SpeechToText.models.common.inference as inference

    original = inference.load_lightning_checkpoint
    inference.load_lightning_checkpoint = lambda _path: ckpt
    try:
        assert detect_model_type("dummy.ckpt") == "tdt"
    finally:
        inference.load_lightning_checkpoint = original


def test_detect_model_type_unknown_raises() -> None:
    import SpeechToText.models.common.inference as inference

    original = inference.load_lightning_checkpoint
    inference.load_lightning_checkpoint = lambda _path: {"state_dict": {"foo": 1}}
    try:
        with pytest.raises(ValueError, match="Could not detect model type"):
            detect_model_type("dummy.ckpt")
    finally:
        inference.load_lightning_checkpoint = original

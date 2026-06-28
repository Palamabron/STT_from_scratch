from __future__ import annotations

import torch

from SpeechToText.models.conformer import FastConformerEncoderConfig
from SpeechToText.models.shared.config import SharedASRConfig
from SpeechToText.models.shared.model import SharedFastConformerASR


def test_shared_asr_model_instantiation() -> None:
    # 1. Prepare config with all active heads
    cfg = SharedASRConfig(
        active_heads=["ctc", "attn", "tdt"],
        aux_layer=1,
        encoder=FastConformerEncoderConfig(
            d_model=64, n_layers=2, n_heads=2, conv_kernel=5, subsampling_factor=8
        ),
    )

    vocab_size = 200
    model = SharedFastConformerASR(
        cfg, vocab_size=vocab_size, blank_id=0, pad_id=4, bos_id=2, eos_id=3
    )

    # Verify module instantiation
    assert model.encoder is not None
    assert model.ctc_proj is not None
    assert model.decoder is not None
    assert model.tdt_decoder is not None
    assert model.tdt_joint is not None


def test_shared_asr_forward_shapes() -> None:
    cfg = SharedASRConfig(
        active_heads=["ctc", "attn", "tdt"],
        aux_layer=1,
        encoder=FastConformerEncoderConfig(
            d_model=64, n_layers=2, n_heads=2, conv_kernel=5, subsampling_factor=8
        ),
    )

    vocab_size = 200
    model = SharedFastConformerASR(
        cfg, vocab_size=vocab_size, blank_id=0, pad_id=4, bos_id=2, eos_id=3
    )

    # Prepare dummy log-mel input: batch_size=1, time_steps=80, channels=80
    feats = torch.randn(1, 80, 80)
    feat_lengths = torch.tensor([80], dtype=torch.long)

    # 1. Encode
    enc, out_lengths, layer_outs = model.encode(feats, feat_lengths)

    # Check subsampling factor 8 output length: (80 / 8) -> around 10 frames
    assert enc.dim() == 3
    assert enc.size(0) == 1
    assert enc.size(2) == 64
    assert out_lengths.item() <= 10

    # 2. CTC Forward
    ctc_probs, aux_probs = model.forward_ctc(enc, out_lengths, layer_outs)
    assert ctc_probs.shape == (1, enc.size(1), vocab_size)

    # 3. Attention Forward
    # Dummy targets: batch_size=1, label_len=5
    targets = torch.randint(1, vocab_size, (1, 5))
    target_lengths = torch.tensor([5], dtype=torch.long)

    attn_probs = model.forward_attention(enc, out_lengths, targets, target_lengths)
    assert attn_probs.shape == (1, 5, vocab_size)

    # 4. TDT Forward
    targets_concat = torch.randint(1, vocab_size, (5,))
    token_logits, duration_logits = model.forward_tdt(
        enc, out_lengths, targets_concat, target_lengths
    )

    # Joint logits shape: [batch, time, labels, vocab]
    assert token_logits.size(0) == 1
    assert token_logits.size(1) == enc.size(1)
    assert token_logits.size(2) == 6  # target_len + 1
    assert token_logits.size(3) == vocab_size

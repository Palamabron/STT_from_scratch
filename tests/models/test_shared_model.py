from __future__ import annotations

import torch
import torch.nn.functional as F

from SpeechToText.models.conformer import FastConformerEncoderConfig
from SpeechToText.models.shared.config import SharedASRConfig
from SpeechToText.models.shared.model import SharedFastConformerASR


def test_shared_asr_model_instantiation() -> None:
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

    feats = torch.randn(1, 80, 80)
    feat_lengths = torch.tensor([80], dtype=torch.long)

    enc, out_lengths, layer_outs = model.encode(feats, feat_lengths)

    assert enc.dim() == 3
    assert enc.size(0) == 1
    assert enc.size(2) == 64
    assert out_lengths.item() <= 10

    ctc_log_probs, aux_log_probs = model.forward_ctc(enc, out_lengths, layer_outs)
    assert ctc_log_probs.shape == (1, enc.size(1), vocab_size)
    assert torch.allclose(ctc_log_probs.exp().sum(dim=-1), torch.ones_like(out_lengths, dtype=torch.float32), atol=1e-4)
    if aux_log_probs.numel() > 0:
        assert aux_log_probs.shape[1:] == ctc_log_probs.shape

    targets = torch.randint(1, vocab_size, (1, 5))
    target_lengths = torch.tensor([5], dtype=torch.long)
    dec_in, _ = model.build_decoder_sequences(targets, target_lengths)

    attn_log_probs = model.forward_attn(enc, out_lengths, dec_in)
    assert attn_log_probs.shape == (1, 6, vocab_size)
    assert torch.allclose(
        attn_log_probs.exp().sum(dim=-1),
        torch.ones((1, 6), dtype=torch.float32),
        atol=1e-4,
    )

    token_logits, _, _ = model.forward_tdt(enc, out_lengths, targets, target_lengths)

    assert token_logits.size(0) == 1
    assert token_logits.size(1) == enc.size(1)
    assert token_logits.size(2) == 6
    assert token_logits.size(3) == vocab_size


def test_forward_ctc_uses_configured_aux_layer() -> None:
    cfg = SharedASRConfig(
        active_heads=["ctc"],
        aux_layer=1,
        encoder=FastConformerEncoderConfig(
            d_model=64, n_layers=2, n_heads=2, conv_kernel=5, subsampling_factor=8
        ),
    )
    model = SharedFastConformerASR(
        cfg, vocab_size=32, blank_id=0, pad_id=4, bos_id=2, eos_id=3
    )

    enc = torch.randn(1, 4, 64)
    layer_outs = [torch.zeros(1, 4, 64), torch.ones(1, 4, 64)]
    _, aux_log_probs = model.forward_ctc(enc, torch.tensor([4]), layer_outs)

    expected = F.log_softmax(model.aux_projs[0](layer_outs[1]), dim=-1)
    assert torch.allclose(aux_log_probs[0], expected)


def test_shared_asr_forward_entrypoint() -> None:
    cfg = SharedASRConfig(
        active_heads=["ctc", "attn"],
        aux_layer=1,
        encoder=FastConformerEncoderConfig(
            d_model=64, n_layers=2, n_heads=2, conv_kernel=5, subsampling_factor=8
        ),
    )
    model = SharedFastConformerASR(
        cfg, vocab_size=64, blank_id=0, pad_id=4, bos_id=2, eos_id=3
    )

    feats = torch.randn(1, 80, 80)
    feat_lengths = torch.tensor([80], dtype=torch.long)
    targets = torch.randint(1, 64, (1, 5))
    target_lengths = torch.tensor([5], dtype=torch.long)

    output = model(feats, feat_lengths, targets=targets, target_lengths=target_lengths)
    assert output.ctc_log_probs is not None
    assert output.dec_log_probs is not None
    assert output.out_lengths.item() <= 10


def test_shared_asr_tdt_head_enables_duration_logits() -> None:
    cfg = SharedASRConfig(
        active_heads=["tdt"],
        use_tdt=True,
        encoder=FastConformerEncoderConfig(
            d_model=64, n_layers=2, n_heads=2, conv_kernel=5, subsampling_factor=8
        ),
    )
    model = SharedFastConformerASR(
        cfg, vocab_size=64, blank_id=0, pad_id=4, bos_id=2, eos_id=3
    )

    feats = torch.randn(1, 80, 80)
    feat_lengths = torch.tensor([80], dtype=torch.long)
    targets = torch.randint(1, 64, (1, 5))
    target_lengths = torch.tensor([5], dtype=torch.long)

    output = model(feats, feat_lengths, targets=targets, target_lengths=target_lengths)
    assert output.duration_logits is not None

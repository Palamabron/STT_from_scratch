from __future__ import annotations

import torch

from SpeechToText.models.common.rnnt import rnnt_loss_mean
from SpeechToText.models.tdt.loss import _sigma_penalty, compute_tdt_losses


def _make_toy_logits(
    *,
    batch_size: int = 1,
    time_steps: int = 4,
    label_steps: int = 3,
    vocab_size: int = 4,
    blank_id: int = 0,
    device: torch.device | None = None,
) -> torch.Tensor:
    device = device or torch.device("cpu")
    logits = torch.full(
        (batch_size, time_steps, label_steps, vocab_size),
        -4.0,
        device=device,
    )
    logits[..., blank_id] = 2.0
    return logits


def test_rnnt_only_matches_torchaudio_wrapper() -> None:
    logits = _make_toy_logits()
    out_lengths = torch.tensor([4], dtype=torch.long)
    targets = torch.tensor([[1, 2]], dtype=torch.long)
    target_lengths = torch.tensor([2], dtype=torch.long)

    direct = rnnt_loss_mean(
        logits=logits,
        out_lengths=out_lengths,
        targets_1d_or_2d=targets,
        target_lengths=target_lengths,
        blank_id=0,
        clamp=-1.0,
        fused_log_softmax=True,
    )
    wrapped = compute_tdt_losses(
        token_logits=logits,
        duration_logits=None,
        out_lengths=out_lengths,
        targets_padded=targets,
        target_lengths=target_lengths,
        blank_id=0,
        label_smoothing=0.0,
        use_tdt=False,
    )
    assert torch.isfinite(direct)
    assert torch.allclose(direct, wrapped.rnnt, atol=1e-5)
    assert torch.allclose(wrapped.total, wrapped.rnnt, atol=1e-5)


def test_tdt_path_adds_duration_and_sigma_terms() -> None:
    token_logits = _make_toy_logits()
    duration_logits = torch.zeros_like(token_logits[..., :5])
    out_lengths = torch.tensor([4], dtype=torch.long)
    targets = torch.tensor([[1, 2]], dtype=torch.long)
    target_lengths = torch.tensor([2], dtype=torch.long)

    losses = compute_tdt_losses(
        token_logits=token_logits,
        duration_logits=duration_logits,
        out_lengths=out_lengths,
        targets_padded=targets,
        target_lengths=target_lengths,
        blank_id=0,
        label_smoothing=0.0,
        use_tdt=True,
        tdt_sigma=0.05,
        tdt_omega=0.1,
    )
    assert torch.isfinite(losses.total)
    assert losses.total.item() >= 0.0
    assert not torch.allclose(losses.total, losses.rnnt)


def test_sigma_penalty_increases_when_mass_below_one() -> None:
    low_mass_logits = torch.full((1, 1, 1, 4), -1.609)
    high_mass_logits = torch.full((1, 1, 1, 4), -1.386)
    out_lengths = torch.tensor([1], dtype=torch.long)
    target_lengths = torch.tensor([0], dtype=torch.long)
    low_mass = _sigma_penalty(low_mass_logits, out_lengths, target_lengths)
    high_mass = _sigma_penalty(high_mass_logits, out_lengths, target_lengths)
    assert low_mass.item() < high_mass.item()

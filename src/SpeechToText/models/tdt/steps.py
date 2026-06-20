from __future__ import annotations

import torch
import torch.nn.functional as F

from SpeechToText.models.common.rnnt import rnnt_loss_mean

from .typing import TDTLosses


def _masked_uniform_kl(
    log_probs: torch.Tensor,
    out_lengths: torch.Tensor,
    target_lengths: torch.Tensor,
    blank_id: int,
    exclude_blank: bool = True,
) -> torch.Tensor:
    """Compute masked KL divergence to a uniform distribution.

    Args:
        log_probs: Joint log-probabilities with shape ``[batch, time, label, vocab]``.
        out_lengths: Valid encoder steps per utterance with shape ``[batch]``.
        target_lengths: Target lengths excluding the start symbol with shape ``[batch]``.
        blank_id: Index of the blank symbol.
        exclude_blank: Whether to exclude the blank symbol from the uniform target.

    Returns:
        Scalar KL divergence averaged over valid positions.
    """
    batch_size, time_steps, label_steps, vocab_size = log_probs.shape

    time_ids = torch.arange(time_steps, device=log_probs.device).view(1, time_steps, 1, 1)
    label_ids = torch.arange(label_steps, device=log_probs.device).view(1, 1, label_steps, 1)

    valid_time = time_ids < out_lengths.view(batch_size, 1, 1, 1)
    valid_label = label_ids < (target_lengths.view(batch_size, 1, 1, 1) + 1)
    valid = (valid_time & valid_label).squeeze(-1)

    log_probs_valid = log_probs[valid]
    if log_probs_valid.numel() == 0:
        return torch.zeros((), device=log_probs.device)

    if exclude_blank and 0 <= blank_id < vocab_size and vocab_size > 1:
        mask = torch.ones(vocab_size, device=log_probs.device, dtype=torch.bool)
        mask[blank_id] = False
        log_probs_valid = log_probs_valid[:, mask]
        effective_vocab = log_probs_valid.size(-1)
        if effective_vocab <= 0:
            return torch.zeros((), device=log_probs.device)
        uniform = torch.full_like(log_probs_valid, 1.0 / float(effective_vocab))
    else:
        uniform = torch.full_like(log_probs_valid, 1.0 / float(vocab_size))

    return F.kl_div(log_probs_valid, uniform, reduction="batchmean", log_target=False)


def compute_tdt_losses(
    *,
    logits: torch.Tensor,
    out_lengths: torch.Tensor,
    targets_padded: torch.Tensor,
    target_lengths: torch.Tensor,
    blank_id: int,
    label_smoothing: float,
    rnnt_clamp: float = 1.0,
    fused_log_softmax: bool = True,
) -> TDTLosses:
    """Compute RNN-T and label-smoothing losses for TDT training.

    Args:
        logits: Joint logits with shape ``[batch, time, label, vocab]``.
        out_lengths: Valid encoder steps per utterance with shape ``[batch]``.
        targets_padded: Padded targets without the start symbol.
        target_lengths: Target lengths with shape ``[batch]``.
        blank_id: Index of the blank symbol.
        label_smoothing: Weight applied to the label-smoothing term.
        rnnt_clamp: Clamp value passed to ``rnnt_loss``.
        fused_log_softmax: Whether to use fused log-softmax in ``rnnt_loss``.

    Returns:
        Combined total, RNN-T, and label-smoothing loss tensors.
    """
    out_lengths_i = out_lengths.to(dtype=torch.long)
    target_lengths_i = target_lengths.to(dtype=torch.long)
    targets_i = targets_padded.to(dtype=torch.long)

    rnnt = rnnt_loss_mean(
        logits=logits,
        out_lengths=out_lengths_i,
        targets_1d_or_2d=targets_i,
        target_lengths=target_lengths_i,
        blank_id=int(blank_id),
        clamp=float(rnnt_clamp),
        fused_log_softmax=bool(fused_log_softmax),
    )

    if label_smoothing <= 0.0:
        return TDTLosses(total=rnnt, rnnt=rnnt, lsm=torch.zeros((), device=rnnt.device))

    log_probs = F.log_softmax(logits.float(), dim=-1)
    lsm = _masked_uniform_kl(
        log_probs=log_probs,
        out_lengths=out_lengths_i,
        target_lengths=target_lengths_i,
        blank_id=int(blank_id),
        exclude_blank=True,
    )

    total = (1.0 - float(label_smoothing)) * rnnt + float(label_smoothing) * lsm
    return TDTLosses(total=total, rnnt=rnnt, lsm=lsm)

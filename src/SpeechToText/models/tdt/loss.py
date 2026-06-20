from __future__ import annotations

import torch
import torch.nn.functional as F

from SpeechToText.models.common.rnnt import rnnt_loss_mean

from .typing import TDTLosses


def _sigma_penalty(
    log_probs: torch.Tensor, out_lengths: torch.Tensor, target_lengths: torch.Tensor
) -> torch.Tensor:
    """Logit under-normalization penalty over valid joint positions."""
    batch_size, time_steps, label_steps, _ = log_probs.shape
    time_ids = torch.arange(time_steps, device=log_probs.device).view(1, time_steps, 1)
    label_ids = torch.arange(label_steps, device=log_probs.device).view(1, 1, label_steps)
    valid_time = time_ids < out_lengths.view(batch_size, 1, 1)
    valid_label = label_ids < (target_lengths.view(batch_size, 1, 1) + 1)
    valid = valid_time & valid_label
    if not bool(valid.any()):
        return torch.zeros((), device=log_probs.device)
    probs = log_probs.exp()
    mass = probs.sum(dim=-1)
    return torch.log(mass[valid] + 1e-8).mean()


def _duration_supervision_loss(
    duration_logits: torch.Tensor,
    token_logits: torch.Tensor,
    out_lengths: torch.Tensor,
    target_lengths: torch.Tensor,
    blank_id: int,
) -> torch.Tensor:
    """Weak duration targets: 0 on blank argmax, 1 on token argmax."""
    batch_size, time_steps, label_steps, _ = duration_logits.shape
    time_ids = torch.arange(time_steps, device=duration_logits.device).view(1, time_steps, 1)
    label_ids = torch.arange(label_steps, device=duration_logits.device).view(1, 1, label_steps)
    valid_time = time_ids < out_lengths.view(batch_size, 1, 1)
    valid_label = label_ids < (target_lengths.view(batch_size, 1, 1) + 1)
    valid = (valid_time & valid_label).reshape(-1)
    if not bool(valid.any()):
        return torch.zeros((), device=duration_logits.device)

    token_preds = torch.argmax(token_logits, dim=-1)
    duration_targets = (token_preds != blank_id).long().clamp(max=duration_logits.size(-1) - 1)
    flat_logits = duration_logits.reshape(-1, duration_logits.size(-1))
    flat_targets = duration_targets.reshape(-1)
    flat_valid = valid.reshape(-1)
    return F.cross_entropy(flat_logits[flat_valid], flat_targets[flat_valid])


def compute_tdt_losses(
    *,
    token_logits: torch.Tensor,
    duration_logits: torch.Tensor | None,
    out_lengths: torch.Tensor,
    targets_padded: torch.Tensor,
    target_lengths: torch.Tensor,
    blank_id: int,
    label_smoothing: float,
    rnnt_clamp: float = -1.0,
    fused_log_softmax: bool = True,
    use_tdt: bool = False,
    tdt_sigma: float = 0.05,
    tdt_omega: float = 0.1,
) -> TDTLosses:
    """Compute RNN-T or TDT (token + duration) training losses."""
    out_lengths_i = out_lengths.to(dtype=torch.long)
    target_lengths_i = target_lengths.to(dtype=torch.long)
    targets_i = targets_padded.to(dtype=torch.long)

    rnnt = rnnt_loss_mean(
        logits=token_logits,
        out_lengths=out_lengths_i,
        targets_1d_or_2d=targets_i,
        target_lengths=target_lengths_i,
        blank_id=int(blank_id),
        clamp=float(rnnt_clamp),
        fused_log_softmax=bool(fused_log_softmax),
    )

    log_probs = F.log_softmax(token_logits.float(), dim=-1)
    lsm = torch.zeros((), device=rnnt.device)
    if label_smoothing > 0.0:
        from SpeechToText.models.tdt.steps import _masked_uniform_kl

        lsm = _masked_uniform_kl(
            log_probs=log_probs,
            out_lengths=out_lengths_i,
            target_lengths=target_lengths_i,
            blank_id=int(blank_id),
            exclude_blank=True,
        )

    if not use_tdt or duration_logits is None:
        total = (1.0 - float(label_smoothing)) * rnnt + float(label_smoothing) * lsm
        return TDTLosses(total=total, rnnt=rnnt, lsm=lsm, tdt=rnnt)

    duration_loss = _duration_supervision_loss(
        duration_logits,
        token_logits,
        out_lengths_i,
        target_lengths_i,
        blank_id=int(blank_id),
    )
    sigma = float(tdt_sigma) * _sigma_penalty(log_probs, out_lengths_i, target_lengths_i)
    omega = float(tdt_omega)
    tdt_component = duration_loss + sigma
    blended = omega * rnnt + (1.0 - omega) * tdt_component
    total = (1.0 - float(label_smoothing)) * blended + float(label_smoothing) * lsm
    return TDTLosses(total=total, rnnt=rnnt, lsm=lsm, tdt=blended)

from __future__ import annotations

import torch
import torch.nn.functional as F

from SpeechToText.models.common.rnnt import rnnt_loss_mean

from .typing import TDTLosses


def _sigma_penalty(
    token_logits: torch.Tensor,
    out_lengths: torch.Tensor,
    target_lengths: torch.Tensor,
    *,
    chunk_t: int = 16,
    chunk_u: int = 32,
) -> torch.Tensor:
    """Logit under-normalization penalty over valid joint positions."""
    values: list[torch.Tensor] = []
    batch_size = int(token_logits.size(0))
    for b in range(batch_size):
        t_max = min(int(out_lengths[b].item()), int(token_logits.size(1)))
        u_max = min(int(target_lengths[b].item()) + 1, int(token_logits.size(2)))
        if t_max <= 0 or u_max <= 0:
            continue
        log_mass_rows: list[torch.Tensor] = []
        for t0 in range(0, t_max, chunk_t):
            t1 = min(t0 + chunk_t, t_max)
            row_chunks: list[torch.Tensor] = []
            for u0 in range(0, u_max, chunk_u):
                u1 = min(u0 + chunk_u, u_max)
                row_chunks.append(token_logits[b, t0:t1, u0:u1].logsumexp(dim=-1))
            log_mass_rows.append(torch.cat(row_chunks, dim=-1))
        values.append(torch.cat(log_mass_rows, dim=0).mean())
    if not values:
        return torch.zeros((), device=token_logits.device)
    return torch.stack(values).mean()


def _duration_supervision_loss(
    duration_logits: torch.Tensor,
    token_logits: torch.Tensor,
    out_lengths: torch.Tensor,
    target_lengths: torch.Tensor,
    blank_id: int,
) -> torch.Tensor:
    """Weak duration targets: 0 on blank argmax, 1 on token argmax."""
    batch_size = int(duration_logits.size(0))
    losses: list[torch.Tensor] = []
    for b in range(batch_size):
        t_max = min(int(out_lengths[b].item()), int(duration_logits.size(1)))
        u_max = min(int(target_lengths[b].item()) + 1, int(duration_logits.size(2)))
        if t_max <= 0 or u_max <= 0:
            continue
        token_preds = token_logits[b, :t_max, :u_max].argmax(dim=-1)
        duration_targets = (token_preds != blank_id).long().clamp(max=duration_logits.size(-1) - 1)
        flat_logits = duration_logits[b, :t_max, :u_max].reshape(-1, duration_logits.size(-1))
        flat_targets = duration_targets.reshape(-1)
        losses.append(F.cross_entropy(flat_logits, flat_targets))
    if not losses:
        return torch.zeros((), device=duration_logits.device)
    return torch.stack(losses).mean()


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
    out_lengths_i = out_lengths.to(dtype=torch.long).clamp(min=1, max=int(token_logits.size(1)))
    target_lengths_i = target_lengths.to(dtype=torch.long)
    max_time = int(out_lengths_i.max().item()) if out_lengths_i.numel() else int(token_logits.size(1))
    max_target = int(target_lengths_i.max().item()) if target_lengths_i.numel() else 0
    label_steps = max_target + 1

    token_logits = token_logits[:, :max_time, :label_steps, :].contiguous()
    if duration_logits is not None:
        duration_logits = duration_logits[:, :max_time, :label_steps, :].contiguous()
    targets_i = targets_padded[:, :max_target].to(dtype=torch.long)

    rnnt = rnnt_loss_mean(
        logits=token_logits,
        out_lengths=out_lengths_i,
        targets_1d_or_2d=targets_i,
        target_lengths=target_lengths_i,
        blank_id=int(blank_id),
        clamp=float(rnnt_clamp),
        fused_log_softmax=bool(fused_log_softmax),
    )

    lsm = torch.zeros((), device=rnnt.device)
    if label_smoothing > 0.0:
        from SpeechToText.models.tdt.steps import _masked_uniform_kl

        lsm = _masked_uniform_kl(
            token_logits=token_logits,
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
    sigma = (
        float(tdt_sigma) * _sigma_penalty(token_logits, out_lengths_i, target_lengths_i)
        if tdt_sigma > 0.0
        else torch.zeros((), device=rnnt.device)
    )
    omega = float(tdt_omega)
    tdt_component = duration_loss + sigma
    blended = omega * rnnt + (1.0 - omega) * tdt_component
    total = (1.0 - float(label_smoothing)) * blended + float(label_smoothing) * lsm
    return TDTLosses(total=total, rnnt=rnnt, lsm=lsm, tdt=blended)

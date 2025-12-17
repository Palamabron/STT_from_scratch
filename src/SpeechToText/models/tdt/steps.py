from __future__ import annotations

import torch
import torch.nn.functional as F

from SpeechToText.models.common.rnnt import rnnt_loss_mean

from .typing import TDTLosses


def _masked_uniform_kl(
    log_probs: torch.Tensor,  # [B,T,U,V]
    out_lengths: torch.Tensor,  # [B]
    target_lengths: torch.Tensor,  # [B] (targets length, NOT incl. start)
    blank_id: int,
    exclude_blank: bool = True,
) -> torch.Tensor:
    b, t, u, v = log_probs.shape

    time_ids = torch.arange(t, device=log_probs.device).view(1, t, 1, 1)
    u_ids = torch.arange(u, device=log_probs.device).view(1, 1, u, 1)

    valid_t = time_ids < out_lengths.view(b, 1, 1, 1)
    valid_u = u_ids < (target_lengths.view(b, 1, 1, 1) + 1)  # include start symbol position
    valid = (valid_t & valid_u).squeeze(-1)  # [B,T,U]

    lp = log_probs[valid]  # [N,V]
    if lp.numel() == 0:
        return torch.zeros((), device=log_probs.device)

    if exclude_blank and 0 <= blank_id < v and v > 1:
        mask = torch.ones(v, device=log_probs.device, dtype=torch.bool)
        mask[blank_id] = False
        lp = lp[:, mask]
        v_eff = lp.size(-1)
        if v_eff <= 0:
            return torch.zeros((), device=log_probs.device)
        uniform = torch.full_like(lp, 1.0 / float(v_eff))
    else:
        uniform = torch.full_like(lp, 1.0 / float(v))

    return F.kl_div(lp, uniform, reduction="batchmean", log_target=False)


def compute_tdt_losses(
    *,
    logits: torch.Tensor,  # [B,T,U,V]
    out_lengths: torch.Tensor,  # [B]
    targets_padded: torch.Tensor,  # [B,U_max] (targets WITHOUT start symbol), padded
    target_lengths: torch.Tensor,  # [B]
    blank_id: int,
    label_smoothing: float,
    rnnt_clamp: float = 1.0,
    fused_log_softmax: bool = True,
) -> TDTLosses:
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

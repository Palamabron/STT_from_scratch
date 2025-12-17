from __future__ import annotations

from typing import cast

import torch
from torchaudio.functional import rnnt_loss


def targets_1d_to_padded_2d(
    targets_1d: torch.Tensor,
    target_lengths: torch.Tensor,
    pad_value: int,
) -> torch.Tensor:
    device = targets_1d.device
    b = int(target_lengths.shape[0])
    max_u = int(target_lengths.max().item()) if b > 0 else 0

    out = torch.full((b, max_u), pad_value, dtype=torch.long, device=device)
    off = 0
    for i in range(b):
        u = int(target_lengths[i].item())
        if u > 0:
            out[i, :u] = targets_1d[off : off + u]
            off += u
    return out


def greedy_rnnt_path_decode_one(
    log_probs: torch.Tensor,
    out_length: int,
    max_symbols_per_t: int,
    blank_id: int,
) -> list[int]:
    t = 0
    u = 0
    emitted: list[int] = []
    max_u = int(log_probs.size(2))

    while t < out_length and u < max_u:
        n_emit = 0
        while n_emit < max_symbols_per_t and u < max_u:
            p = log_probs[0, t, u]
            k = int(torch.argmax(p).item())
            if k == blank_id:
                break
            emitted.append(k)
            u += 1
            n_emit += 1
        t += 1

    return emitted


def rnnt_loss_mean(
    *,
    logits: torch.Tensor,
    out_lengths: torch.Tensor,
    targets_1d_or_2d: torch.Tensor,
    target_lengths: torch.Tensor,
    blank_id: int,
    clamp: float,
    fused_log_softmax: bool,
) -> torch.Tensor:
    targets_2d = (
        targets_1d_to_padded_2d(targets_1d_or_2d, target_lengths, pad_value=blank_id)
        if targets_1d_or_2d.dim() == 1
        else targets_1d_or_2d
    )

    loss_any = rnnt_loss(
        logits=logits,
        targets=targets_2d,
        logit_lengths=out_lengths,
        target_lengths=target_lengths,
        blank=int(blank_id),
        clamp=float(clamp),
        reduction="mean",
        fused_log_softmax=bool(fused_log_softmax),
    )
    return cast(torch.Tensor, loss_any)

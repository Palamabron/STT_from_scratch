from __future__ import annotations

import torch
import torch.nn.functional as F

from SpeechToText.models.tdt.loss import compute_tdt_losses

__all__ = ["compute_tdt_losses", "_masked_uniform_kl"]


def _masked_uniform_kl(
    log_probs: torch.Tensor,
    out_lengths: torch.Tensor,
    target_lengths: torch.Tensor,
    blank_id: int,
    exclude_blank: bool = True,
) -> torch.Tensor:
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

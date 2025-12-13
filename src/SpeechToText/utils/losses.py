from __future__ import annotations

import torch
import torch.nn.functional as F


def ctc_loss_with_label_smoothing(
    log_probs_t: torch.Tensor,
    targets: torch.Tensor,
    input_lengths: torch.Tensor,
    target_lengths: torch.Tensor,
    blank_id: int,
    lsm_weight: float,
    autocast_device_type: str,
    exclude_blank_from_ls: bool = True,
) -> torch.Tensor:
    log_probs_t = log_probs_t.float()
    input_lengths = input_lengths.to(dtype=torch.long)
    target_lengths = target_lengths.to(dtype=torch.long)

    with torch.autocast(device_type=autocast_device_type, enabled=False):
        ctc = F.ctc_loss(
            log_probs_t,
            targets,
            input_lengths,
            target_lengths,
            reduction="mean",
            blank=blank_id,
            zero_infinity=True,
        )

        if lsm_weight <= 0.0:
            return ctc

        T, B, V = log_probs_t.shape
        if V <= 1:
            return ctc

        time_ids = torch.arange(T, device=log_probs_t.device).unsqueeze(1)
        valid = time_ids < input_lengths.unsqueeze(0)

        log_probs_valid = log_probs_t[valid]
        if log_probs_valid.numel() == 0:
            return ctc

        if exclude_blank_from_ls and 0 <= blank_id < V and V > 1:
            vocab_mask = torch.ones(V, device=log_probs_t.device, dtype=torch.bool)
            vocab_mask[blank_id] = False
            log_probs_valid = log_probs_valid[:, vocab_mask]
            V_eff = log_probs_valid.size(-1)
            if V_eff <= 0:
                return ctc
            uniform = torch.full_like(log_probs_valid, 1.0 / float(V_eff))
        else:
            uniform = torch.full_like(log_probs_valid, 1.0 / float(V))

        ls = F.kl_div(log_probs_valid, uniform, reduction="batchmean", log_target=False)

        return (1.0 - lsm_weight) * ctc + lsm_weight * ls

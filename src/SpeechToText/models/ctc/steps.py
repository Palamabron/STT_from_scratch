from __future__ import annotations

from dataclasses import dataclass

import torch

from SpeechToText.models.common import ctc_loss_with_label_smoothing


@dataclass(frozen=True)
class CTCLosses:
    total: torch.Tensor
    main: torch.Tensor
    aux: torch.Tensor


def compute_ctc_losses(
    *,
    log_probs: torch.Tensor,  # [B,T,V]
    out_lengths: torch.Tensor,  # [B]
    aux_log_probs: torch.Tensor,  # [Naux,B,T,V] or empty
    targets: torch.Tensor,  # concat 1D
    target_lengths: torch.Tensor,  # [B]
    blank_id: int,
    lsm_weight: float,
    aux_weight: float,
) -> CTCLosses:
    lp_t = log_probs.transpose(0, 1)
    autocast_device_type = "cuda" if lp_t.is_cuda else "cpu"

    main = ctc_loss_with_label_smoothing(
        log_probs_t=lp_t,
        targets=targets,
        input_lengths=out_lengths,
        target_lengths=target_lengths,
        blank_id=blank_id,
        lsm_weight=lsm_weight,
        autocast_device_type=autocast_device_type,
        exclude_blank_from_ls=True,
    )

    if aux_weight > 0.0 and aux_log_probs.numel() > 0:
        aux_losses: list[torch.Tensor] = []
        for i in range(aux_log_probs.size(0)):
            aux_losses.append(
                ctc_loss_with_label_smoothing(
                    log_probs_t=aux_log_probs[i].transpose(0, 1),
                    targets=targets,
                    input_lengths=out_lengths,
                    target_lengths=target_lengths,
                    blank_id=blank_id,
                    lsm_weight=lsm_weight,
                    autocast_device_type=autocast_device_type,
                    exclude_blank_from_ls=True,
                )
            )
        aux = torch.stack(aux_losses).mean()
    else:
        aux = torch.tensor(0.0, device=main.device)

    total = main + float(aux_weight) * aux
    return CTCLosses(total=total, main=main, aux=aux)

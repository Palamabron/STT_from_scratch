from __future__ import annotations

from dataclasses import dataclass

import torch

from SpeechToText.models.common import ctc_loss_with_label_smoothing


@dataclass(frozen=True)
class CTCLosses:
    """CTC loss breakdown for the main and auxiliary heads."""

    total: torch.Tensor
    main: torch.Tensor
    aux: torch.Tensor


def compute_ctc_losses(
    *,
    log_probs: torch.Tensor,
    out_lengths: torch.Tensor,
    aux_log_probs: torch.Tensor,
    targets: torch.Tensor,
    target_lengths: torch.Tensor,
    blank_id: int,
    lsm_weight: float,
    aux_weight: float,
) -> CTCLosses:
    """Compute main and auxiliary CTC losses with optional label smoothing.

    Args:
        log_probs: Log-probabilities with shape ``[batch, time, vocab]``.
        out_lengths: Valid encoder steps per utterance with shape ``[batch]``.
        aux_log_probs: Auxiliary head log-probabilities or an empty tensor.
        targets: Concatenated target token ids.
        target_lengths: Target lengths with shape ``[batch]``.
        blank_id: Index of the CTC blank symbol.
        lsm_weight: Label-smoothing weight for the main head.
        aux_weight: Weight applied to the auxiliary loss term.

    Returns:
        Combined total, main, and auxiliary loss tensors.
    """
    log_probs_time_major = log_probs.transpose(0, 1)
    autocast_device_type = "cuda" if log_probs_time_major.is_cuda else "cpu"

    max_time = log_probs_time_major.size(0)
    out_lengths = out_lengths.clamp(max=max_time)

    main = ctc_loss_with_label_smoothing(
        log_probs_t=log_probs_time_major,
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
        for head_index in range(aux_log_probs.size(0)):
            aux_losses.append(
                ctc_loss_with_label_smoothing(
                    log_probs_t=aux_log_probs[head_index].transpose(0, 1),
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

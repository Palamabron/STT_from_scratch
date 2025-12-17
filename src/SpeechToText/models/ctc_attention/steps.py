from __future__ import annotations

from dataclasses import dataclass

import torch

from SpeechToText.models.common import ctc_loss_with_label_smoothing


@dataclass(frozen=True)
class CTCAttnLosses:
    total: torch.Tensor
    ctc_main: torch.Tensor
    ctc_aux: torch.Tensor
    attn: torch.Tensor


def compute_ctc_attn_losses(
    *,
    ctc_log_probs: torch.Tensor,  # [B,T,V_ctc]
    out_lengths: torch.Tensor,  # [B]
    aux_log_probs: torch.Tensor,  # [Naux,B,T,V_ctc] or empty
    targets: torch.Tensor,  # concat 1D
    target_lengths: torch.Tensor,  # [B]
    dec_log_probs: torch.Tensor,  # [B,U,V_dec]
    dec_out: torch.Tensor,  # [B,U]
    blank_id: int,
    ctc_label_smoothing: float,
    aux_ctc_weight: float,
    ctc_weight: float,
    autocast_device_type: str,
    attn_loss_fn: torch.nn.Module,  # NLLLoss
) -> CTCAttnLosses:
    ctc_lp_t = ctc_log_probs.transpose(0, 1)

    main_ctc = ctc_loss_with_label_smoothing(
        log_probs_t=ctc_lp_t,
        targets=targets,
        input_lengths=out_lengths,
        target_lengths=target_lengths,
        blank_id=blank_id,
        lsm_weight=ctc_label_smoothing,
        autocast_device_type=autocast_device_type,
        exclude_blank_from_ls=True,
    )

    if aux_ctc_weight > 0.0 and aux_log_probs.numel() > 0:
        aux_losses: list[torch.Tensor] = []
        for i in range(aux_log_probs.size(0)):
            aux_losses.append(
                ctc_loss_with_label_smoothing(
                    log_probs_t=aux_log_probs[i].transpose(0, 1),
                    targets=targets,
                    input_lengths=out_lengths,
                    target_lengths=target_lengths,
                    blank_id=blank_id,
                    lsm_weight=ctc_label_smoothing,
                    autocast_device_type=autocast_device_type,
                    exclude_blank_from_ls=True,
                )
            )
        aux_ctc = torch.stack(aux_losses).mean()
    else:
        aux_ctc = torch.tensor(0.0, device=main_ctc.device)

    b, u, v = dec_log_probs.shape
    attn = attn_loss_fn(dec_log_probs.reshape(b * u, v), dec_out.reshape(b * u))

    lam = float(ctc_weight)
    total = lam * main_ctc + (1.0 - lam) * attn + float(aux_ctc_weight) * aux_ctc
    return CTCAttnLosses(total=total, ctc_main=main_ctc, ctc_aux=aux_ctc, attn=attn)

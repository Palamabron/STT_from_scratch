from __future__ import annotations

from dataclasses import dataclass

import torch

from SpeechToText.models.common import ctc_loss_with_label_smoothing


@dataclass(frozen=True)
class CTCAttnLosses:
    """Loss breakdown for the CTC + attention hybrid model."""

    total: torch.Tensor
    ctc_main: torch.Tensor
    ctc_aux: torch.Tensor
    attn: torch.Tensor


def compute_ctc_attn_losses(
    *,
    ctc_log_probs: torch.Tensor,
    out_lengths: torch.Tensor,
    aux_log_probs: torch.Tensor,
    targets: torch.Tensor,
    target_lengths: torch.Tensor,
    dec_log_probs: torch.Tensor | None,
    dec_out: torch.Tensor | None,
    blank_id: int,
    ctc_label_smoothing: float,
    aux_ctc_weight: float,
    ctc_weight: float,
    autocast_device_type: str,
    attn_loss_fn: torch.nn.Module,
    include_attn: bool = True,
) -> CTCAttnLosses:
    """Compute CTC, auxiliary CTC, and attention decoder losses.

    Args:
        ctc_log_probs: Main CTC log-probabilities with shape ``[batch, time, vocab]``.
        out_lengths: Valid encoder steps per utterance with shape ``[batch]``.
        aux_log_probs: Auxiliary CTC log-probabilities or an empty tensor.
        targets: Concatenated target token ids.
        target_lengths: Target lengths with shape ``[batch]``.
        dec_log_probs: Decoder log-probabilities with shape ``[batch, label, vocab]``.
        dec_out: Decoder target ids with shape ``[batch, label]``.
        blank_id: Index of the CTC blank symbol.
        ctc_label_smoothing: Label-smoothing weight for CTC heads.
        aux_ctc_weight: Weight applied to auxiliary CTC losses.
        ctc_weight: Interpolation weight between CTC and attention losses.
        autocast_device_type: Device type string used for mixed precision.
        attn_loss_fn: Negative log-likelihood loss for the decoder head.

    Returns:
        Combined total, CTC, auxiliary CTC, and attention loss tensors.
    """
    ctc_log_probs_time_major = ctc_log_probs.transpose(0, 1)

    max_time = ctc_log_probs_time_major.size(0)
    out_lengths = out_lengths.clamp(max=max_time)

    main_ctc = ctc_loss_with_label_smoothing(
        log_probs_t=ctc_log_probs_time_major,
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
        for head_index in range(aux_log_probs.size(0)):
            aux_losses.append(
                ctc_loss_with_label_smoothing(
                    log_probs_t=aux_log_probs[head_index].transpose(0, 1),
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

    if include_attn:
        assert dec_log_probs is not None and dec_out is not None
        batch_size, label_steps, vocab_size = dec_log_probs.shape
        attn = attn_loss_fn(
            dec_log_probs.reshape(batch_size * label_steps, vocab_size),
            dec_out.reshape(batch_size * label_steps),
        )
    else:
        attn = torch.tensor(0.0, device=main_ctc.device)

    ctc_mix = float(ctc_weight)
    attn_mix = (1.0 - ctc_mix) if include_attn else 0.0
    total = ctc_mix * main_ctc + attn_mix * attn + float(aux_ctc_weight) * aux_ctc
    return CTCAttnLosses(total=total, ctc_main=main_ctc, ctc_aux=aux_ctc, attn=attn)

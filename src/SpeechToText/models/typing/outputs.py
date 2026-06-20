from __future__ import annotations

from typing import NamedTuple

import torch


class CTCOutput(NamedTuple):
    """Outputs from the CTC acoustic model."""

    log_probs: torch.Tensor
    out_lengths: torch.Tensor
    aux_log_probs: torch.Tensor


class CTCAttnOutput(NamedTuple):
    """Outputs from the CTC + attention hybrid model."""

    ctc_log_probs: torch.Tensor
    out_lengths: torch.Tensor
    aux_log_probs: torch.Tensor
    dec_log_probs: torch.Tensor | None


class TDTOutput(NamedTuple):
    """Outputs from the RNN-T / TDT transducer model."""

    log_probs: torch.Tensor
    out_lengths: torch.Tensor
    target_lengths: torch.Tensor
    token_logits: torch.Tensor | None = None
    duration_logits: torch.Tensor | None = None
    duration_log_probs: torch.Tensor | None = None

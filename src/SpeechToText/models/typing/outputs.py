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
    """Outputs from the token-and-duration transducer model."""

    log_probs: torch.Tensor
    out_lengths: torch.Tensor
    target_lengths: torch.Tensor

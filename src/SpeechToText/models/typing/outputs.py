from __future__ import annotations

from typing import NamedTuple

import torch


class CTCOutput(NamedTuple):
    log_probs: torch.Tensor  # [B, T, V]
    out_lengths: torch.Tensor  # [B]
    aux_log_probs: torch.Tensor  # [Naux, B, T, V] or empty


class CTCAttnOutput(NamedTuple):
    ctc_log_probs: torch.Tensor  # [B, T, V_ctc]
    out_lengths: torch.Tensor  # [B]
    aux_log_probs: torch.Tensor  # [Naux, B, T, V_ctc] or empty
    dec_log_probs: torch.Tensor | None  # [B, U, V_dec]


class TDTOutput(NamedTuple):
    log_probs: torch.Tensor  # [B, T, U, V]
    out_lengths: torch.Tensor  # [B]
    target_lengths: torch.Tensor  # [B]

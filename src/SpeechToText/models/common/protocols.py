from __future__ import annotations

from typing import Any, Protocol

import torch
import torch.nn as nn


class EncoderProtocol(Protocol):
    def __call__(
        self, feats: torch.Tensor, feat_lengths: torch.Tensor, return_layer_outputs: bool = ...
    ) -> tuple[torch.Tensor, torch.Tensor, Any]: ...


class ASRModelProtocol(Protocol):
    encoder: nn.Module
    ctc_proj: nn.Module | None

    def encode(
        self, feats: torch.Tensor, feat_lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, Any]: ...

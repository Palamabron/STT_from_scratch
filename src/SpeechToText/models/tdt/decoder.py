from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch
import torch.nn as nn


@dataclass
class TDTDecoderConfig:
    """Configuration for the RNN-T label predictor."""

    vocab_size: int = 1025
    d_model: int = 256
    num_layers: int = 1
    dropout: float = 0.1


class TDTDecoder(nn.Module):
    """Embedding plus LSTM predictor for token-and-duration transducer training."""

    def __init__(self, cfg: TDTDecoderConfig) -> None:
        super().__init__()
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.lstm = nn.LSTM(
            input_size=cfg.d_model,
            hidden_size=cfg.d_model,
            num_layers=cfg.num_layers,
            batch_first=True,
            dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Return predictor states with shape ``[batch, time, d_model]``."""
        embedded = self.embed(tokens)
        states, _ = self.lstm(embedded)
        return cast(torch.Tensor, states)

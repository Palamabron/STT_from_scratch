from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch
import torch.nn as nn


@dataclass
class JointNetConfig:
    """Configuration for the RNN-T joint network."""

    vocab_size: int = 1025
    enc_d: int = 256
    pred_d: int = 256
    joint_d: int = 512


class JointNet(nn.Module):
    """Combine encoder and predictor states into vocabulary logits."""

    def __init__(self, cfg: JointNetConfig) -> None:
        super().__init__()
        self.enc_proj = nn.Linear(cfg.enc_d, cfg.joint_d)
        self.pred_proj = nn.Linear(cfg.pred_d, cfg.joint_d)
        self.out = nn.Linear(cfg.joint_d, cfg.vocab_size)

    def forward(self, enc: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
        """Return joint logits with shape ``[batch, time, label, vocab]``."""
        enc_h = self.enc_proj(enc).unsqueeze(2)
        pred_h = self.pred_proj(pred).unsqueeze(1)
        return cast(torch.Tensor, self.out(torch.tanh(enc_h + pred_h)))

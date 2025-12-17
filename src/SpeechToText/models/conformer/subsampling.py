from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class ConvSubsamplingConfig:
    in_feats: int = 80
    d_model: int = 256
    dropout: float = 0.1


class ConvSubsampling4(nn.Module):
    """
    4x time reduction: two Conv2d layers with stride 2 on time axis.
    Input:  [B, T, F]
    Output: [B, T', d_model], lengths updated
    """

    def __init__(self, cfg: ConvSubsamplingConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.conv = nn.Sequential(
            nn.Conv2d(1, cfg.d_model, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(cfg.d_model, cfg.d_model, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.out = nn.Sequential(
            nn.Linear(cfg.d_model * ((cfg.in_feats + 3) // 4), cfg.d_model),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b, t, f = x.shape
        x = x.unsqueeze(1)  # [B, 1, T, F]
        x = self.conv(x)  # [B, C, T', F']
        b, c, t2, f2 = x.shape
        x = x.transpose(1, 2).contiguous().view(b, t2, c * f2)
        x = self.out(x)

        lengths = torch.div(lengths + 1, 2, rounding_mode="floor")
        lengths = torch.div(lengths + 1, 2, rounding_mode="floor")
        return x, lengths

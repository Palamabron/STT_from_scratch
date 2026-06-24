from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch
import torch.nn as nn


def subsample_lengths(lengths: int | torch.Tensor, factor: int) -> int | torch.Tensor:
    """Apply Conformer conv subsampling to sequence lengths."""
    if factor not in (2, 4, 8):
        raise ValueError(f"Unsupported subsampling_factor: {factor}")
    if factor == 2:
        return ((lengths - 1) // 2) + 1
    if factor == 4:
        return lengths >> 2
    return lengths >> 3


@dataclass
class ConvSubsamplingConfig:
    """Configuration for convolutional subsampling front-ends."""

    in_feats: int = 80
    d_model: int = 256
    dropout: float = 0.1


class ConvSubsampling2(nn.Module):
    """2x time reduction via two convolution layers (stride 1 then stride 2)."""

    def __init__(self, cfg: ConvSubsamplingConfig) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, cfg.d_model, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(cfg.d_model, cfg.d_model, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        freq_out = (cfg.in_feats - 1) // 2 + 1
        self.out = nn.Linear(cfg.d_model * freq_out, cfg.d_model)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert ``[B, T, F]`` features to ``[B, T', d_model]``."""
        x = x.unsqueeze(1).transpose(2, 3)
        x = self.conv(x)
        batch, channels, freq, time = x.size()
        x = x.transpose(2, 3).contiguous().view(batch, time, channels * freq)
        x = self.out(x)
        out_lengths = cast(torch.Tensor, subsample_lengths(lengths, 2))
        return x, out_lengths


class ConvSubsampling4(nn.Module):
    """4x time reduction via two stride-2 convolution layers."""

    def __init__(self, cfg: ConvSubsamplingConfig) -> None:
        super().__init__()
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
        """Convert ``[B, T, F]`` features to ``[B, T', d_model]``."""
        x = x.unsqueeze(1)
        x = self.conv(x)
        batch, channels, time, freq = x.shape
        x = x.transpose(1, 2).contiguous().view(batch, time, channels * freq)
        x = self.out(x)
        return x, cast(torch.Tensor, subsample_lengths(lengths, 4))


class ConvSubsampling8(nn.Module):
    """8x time reduction via three stride-2 convolution layers."""

    def __init__(self, cfg: ConvSubsamplingConfig) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, cfg.d_model, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(cfg.d_model, cfg.d_model, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(cfg.d_model, cfg.d_model, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        freq_out = (cfg.in_feats + 7) // 8
        self.out = nn.Sequential(
            nn.Linear(cfg.d_model * freq_out, cfg.d_model),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert ``[B, T, F]`` features to ``[B, T', d_model]``."""
        x = x.unsqueeze(1)
        x = self.conv(x)
        batch, channels, time, freq = x.shape
        x = x.transpose(1, 2).contiguous().view(batch, time, channels * freq)
        x = self.out(x)
        return x, cast(torch.Tensor, subsample_lengths(lengths, 8))

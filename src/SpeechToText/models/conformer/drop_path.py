from __future__ import annotations

import torch
import torch.nn as nn


class DropPath(nn.Module):
    """Stochastic depth for residual branches."""

    def __init__(self, drop_prob: float) -> None:
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (not self.training) or self.drop_prob <= 0.0:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.size(0),) + (1,) * (x.dim() - 1)
        mask = torch.bernoulli(torch.full(shape, keep, device=x.device, dtype=x.dtype))
        return x * mask / keep

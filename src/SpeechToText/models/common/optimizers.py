from __future__ import annotations

import math
from collections.abc import Callable

import torch
from lightning.pytorch.utilities.types import OptimizerLRSchedulerConfig


def build_noam_lambda_lr(
    *,
    warmup_steps: int = 4000,
    d_model: int = 512,
) -> Callable[[int], float]:
    """Build the Noam learning-rate multiplier used with AdamW."""
    if warmup_steps <= 0:
        raise ValueError("warmup_steps must be > 0")

    warmup = float(warmup_steps)

    def lr_lambda(step_idx: int) -> float:
        step = float(step_idx + 1)
        return min(step / warmup, math.sqrt(warmup / step))

    return lr_lambda


def configure_adamw_noam(
    module: torch.nn.Module,
    *,
    lr: float,
    betas: tuple[float, float],
    epsilon: float = 1e-8,
    weight_decay: float,
    warmup_steps: int,
    d_model: int,
) -> OptimizerLRSchedulerConfig:
    """Configure AdamW with a Noam schedule for a Lightning module."""
    optimizer = torch.optim.AdamW(
        module.parameters(),
        lr=float(lr),
        betas=betas,
        eps=float(epsilon),
        weight_decay=float(weight_decay),
    )

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=build_noam_lambda_lr(warmup_steps=warmup_steps, d_model=d_model),
    )

    return {
        "optimizer": optimizer,
        "lr_scheduler": {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1,
        },
    }

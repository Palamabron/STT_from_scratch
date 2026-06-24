from __future__ import annotations

import math
from collections.abc import Callable

import torch
from lightning.pytorch.utilities.types import OptimizerLRSchedulerConfig


def build_noam_lambda_lr(
    *,
    warmup_steps: int = 4000,
) -> Callable[[int], float]:
    """Build the Noam learning-rate multiplier used with AdamW.

    Peak LR is controlled by the AdamW ``lr`` argument; recipes tune that value
    per encoder width rather than applying an extra ``d_model`` scale here.
    """
    if warmup_steps <= 0:
        raise ValueError("warmup_steps must be > 0")

    warmup = float(warmup_steps)

    def lr_lambda(step_idx: int) -> float:
        step = float(step_idx + 1)
        return min(step / warmup, math.sqrt(warmup / step))

    return lr_lambda


def build_cosine_warmup_lambda_lr(
    *,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.0,
) -> Callable[[int], float]:
    """Build linear warmup followed by cosine decay to ``min_lr_ratio``."""
    if warmup_steps <= 0:
        raise ValueError("warmup_steps must be > 0")
    if total_steps <= warmup_steps:
        raise ValueError("total_steps must be > warmup_steps")
    if not 0.0 <= min_lr_ratio <= 1.0:
        raise ValueError("min_lr_ratio must be in [0, 1]")

    warmup = float(warmup_steps)
    decay_steps = float(total_steps - warmup_steps)

    def lr_lambda(step_idx: int) -> float:
        step = float(step_idx + 1)
        if step <= warmup:
            return step / warmup
        progress = min(1.0, (step - warmup) / decay_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return lr_lambda


def configure_adamw_cosine_warmup(
    module: torch.nn.Module,
    *,
    lr: float,
    betas: tuple[float, float],
    epsilon: float = 1e-8,
    weight_decay: float,
    warmup_steps: int,
    total_steps: int,
    eta_min: float = 0.0,
) -> OptimizerLRSchedulerConfig:
    """Configure AdamW with linear warmup and cosine annealing for a Lightning module."""
    base_lr = float(lr)
    min_lr_ratio = min(1.0, max(0.0, float(eta_min) / base_lr)) if base_lr > 0 else 0.0

    optimizer = torch.optim.AdamW(
        module.parameters(),
        lr=base_lr,
        betas=betas,
        eps=float(epsilon),
        weight_decay=float(weight_decay),
    )

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=build_cosine_warmup_lambda_lr(
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            min_lr_ratio=min_lr_ratio,
        ),
    )

    return {
        "optimizer": optimizer,
        "lr_scheduler": {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1,
        },
    }


def configure_adamw_noam(
    module: torch.nn.Module,
    *,
    lr: float,
    betas: tuple[float, float],
    epsilon: float = 1e-8,
    weight_decay: float,
    warmup_steps: int,
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
        lr_lambda=build_noam_lambda_lr(warmup_steps=warmup_steps),
    )

    return {
        "optimizer": optimizer,
        "lr_scheduler": {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1,
        },
    }

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
    if warmup_steps <= 0:
        raise ValueError("warmup_steps must be > 0")
    if d_model <= 0:
        raise ValueError("d_model must be > 0")

    dm = float(d_model)
    wu = float(warmup_steps)

    inv_sqrt_dm = 1.0 / math.sqrt(dm)
    wu_15 = wu * math.sqrt(wu)  # wu^(1.5)

    def lr_lambda(step_idx: int) -> float:
        step = float(step_idx + 1)
        inv_sqrt_step = 1.0 / math.sqrt(step)
        warmup_term = step / wu_15
        return float(inv_sqrt_dm * min(inv_sqrt_step, warmup_term))

    return lr_lambda


def configure_adamw_noam(
    module: torch.nn.Module,
    *,
    learning_rate: float,
    betas: tuple[float, float],
    epsilon: float,
    weight_decay: float,
    warmup_steps: int,
    d_model: int,
) -> OptimizerLRSchedulerConfig:
    opt = torch.optim.AdamW(
        module.parameters(),
        lr=float(learning_rate),
        betas=betas,
        eps=float(epsilon),
        weight_decay=float(weight_decay),
    )

    sched = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lr_lambda=build_noam_lambda_lr(warmup_steps=warmup_steps, d_model=d_model),
    )

    return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "step"}}

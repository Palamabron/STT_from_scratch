"""Select AdamW + LR scheduler based on optimizer config."""

from __future__ import annotations

from typing import Any

import torch.nn as nn
from lightning.pytorch.utilities.types import OptimizerLRSchedulerConfig

from SpeechToText.models.common.optimizers import (
    configure_adamw_cosine_warmup,
    configure_adamw_noam,
)


def configure_adamw_scheduler(
    module: nn.Module,
    *,
    optimizer_cfg: Any,
    total_steps: int,
) -> OptimizerLRSchedulerConfig:
    """Configure AdamW with Noam or cosine-warmup schedule from ``optimizer_cfg``."""
    warmup_steps = max(1, int(total_steps * optimizer_cfg.warmup_ratio))
    scheduler = getattr(optimizer_cfg, "scheduler", "noam")

    common = dict(
        lr=optimizer_cfg.lr,
        betas=optimizer_cfg.betas,
        epsilon=1e-8,
        weight_decay=getattr(optimizer_cfg, "weight_decay", 0.01),
        warmup_steps=warmup_steps,
    )

    if scheduler == "cosine":
        return configure_adamw_cosine_warmup(
            module,
            total_steps=max(warmup_steps + 1, total_steps),
            eta_min=float(getattr(optimizer_cfg, "cosine_eta_min", 0.0)),
            **common,
        )

    return configure_adamw_noam(module, **common)

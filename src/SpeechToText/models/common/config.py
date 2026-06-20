from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from ...augmentation import AudioAugmentConfig, SpecAugmentConfig
from ...dataset import DataConfig

PrecisionType = Literal["64", "32", "16", "64-true", "32-true", "16-true", "16-mixed", "bf16-mixed"]


@dataclass(slots=True)
class BaseOptimizerConfig:
    lr: float = 2e-3
    betas: tuple[float, float] = (0.9, 0.98)
    weight_decay: float = 1e-2
    warmup_ratio: float = 0.1


@dataclass(slots=True)
class BaseTrainConfig:
    data: DataConfig = field(default_factory=DataConfig)
    spec_augment: SpecAugmentConfig = field(default_factory=SpecAugmentConfig)
    audio_augment: AudioAugmentConfig = field(default_factory=AudioAugmentConfig)
    spec_augment_start_epoch: int = 0
    audio_augment_start_epoch: int = 3
    seed: int = 42
    ckpt_path: str | None = None
    max_epochs: int = 50
    accumulate_grad_batches: int = 1
    gradient_clip_val: float = 5.0
    val_check_interval: float = 1.0
    log_every_n_steps: int = 10
    precision: PrecisionType = "32-true"
    wandb_project: str = "multilingual_asr"
    wandb_run_name: str | None = None

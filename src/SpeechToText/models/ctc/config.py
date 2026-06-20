from __future__ import annotations

import os
from dataclasses import dataclass, field

from SpeechToText.augmentation import DEFAULT_NOISE_BANK_PATH, DEFAULT_RIR_BANK_PATH
from SpeechToText.models.common.config import BaseOptimizerConfig, BaseTrainConfig, PrecisionType
from SpeechToText.models.ctc.model import FastConformerCTCConfig


@dataclass(slots=True)
class TrainConfig(BaseTrainConfig):
    checkpoint_dir: str = "./checkpoints/ctc"
    musan_path: str | None = DEFAULT_NOISE_BANK_PATH
    rirs_path: str | None = DEFAULT_RIR_BANK_PATH
    model: FastConformerCTCConfig = field(default_factory=FastConformerCTCConfig)
    optimizer: BaseOptimizerConfig = field(default_factory=BaseOptimizerConfig)
    max_epochs: int = 50
    precision: PrecisionType = "bf16-mixed"
    log_every_n_steps: int = 10
    val_check_interval: float = 1.0
    accumulate_grad_batches: int = 1
    gradient_clip_val: float = 1.0
    ctc_label_smoothing: float = 0.1
    aux_ctc_weight: float = 0.3
    wandb_project: str = os.getenv("WANDB_PROJECT", "multilingual_asr_ctc")
    wandb_run_name: str | None = None

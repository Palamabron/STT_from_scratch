from __future__ import annotations

import os
from dataclasses import dataclass, field

from SpeechToText.augmentation import DEFAULT_NOISE_BANK_PATH, DEFAULT_RIR_BANK_PATH
from SpeechToText.models.common.config import BaseOptimizerConfig, BaseTrainConfig
from SpeechToText.models.tdt.model import FastConformerTDTConfig


@dataclass
class TrainConfig(BaseTrainConfig):
    checkpoint_dir: str = "./checkpoints/rnnt"
    musan_path: str | None = DEFAULT_NOISE_BANK_PATH
    rirs_path: str | None = DEFAULT_RIR_BANK_PATH
    model: FastConformerTDTConfig = field(default_factory=FastConformerTDTConfig)
    optimizer: BaseOptimizerConfig = field(default_factory=BaseOptimizerConfig)
    blank_id: int = 0
    rnnt_clamp: float = -1.0
    fused_log_softmax: bool = True
    compute_eval_loss: bool = False
    val_max_symbols_per_t: int = 10
    joint_fused_batch_size: int | None = 4
    label_smoothing: float = 0.1
    use_tdt: bool = False
    tdt_sigma: float = 0.05
    tdt_omega: float = 0.1
    early_stopping_patience: int | None = None
    max_epochs: int = 50
    wandb_project: str = os.getenv("WANDB_PROJECT", "multilingual_asr_rnnt")

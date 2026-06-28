from __future__ import annotations

import os
from dataclasses import dataclass, field

from SpeechToText.augmentation import DEFAULT_NOISE_BANK_PATH, DEFAULT_RIR_BANK_PATH
from SpeechToText.models.common.config import BaseOptimizerConfig, BaseTrainConfig, PrecisionType
from SpeechToText.models.ctc_attention.model import FastConformerCTCAttentionConfig


@dataclass
class TrainConfig(BaseTrainConfig):
    checkpoint_dir: str = "./checkpoints/ctc_attention"
    musan_path: str | None = DEFAULT_NOISE_BANK_PATH
    rirs_path: str | None = DEFAULT_RIR_BANK_PATH
    model: FastConformerCTCAttentionConfig = field(default_factory=FastConformerCTCAttentionConfig)
    optimizer: BaseOptimizerConfig = field(default_factory=BaseOptimizerConfig)
    ctc_label_smoothing: float = 0.1
    aux_ctc_weight: float = 0.3
    ctc_weight: float = 0.3
    freeze_encoder_epochs: int = 0
    decoder_warmup_epochs: int = 0
    ctc_calibration_epochs: int = 0
    freeze_decoder_after_epoch: int | None = None
    val_decode_mode: str = "ctc_greedy"
    wandb_project: str = os.getenv("WANDB_PROJECT", "multilingual_asr_ctc_attn")
    precision: PrecisionType = "bf16-mixed"

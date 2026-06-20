from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field

import lightning.pytorch as pl
import tyro

from SpeechToText.augmentation import DEFAULT_NOISE_BANK_PATH, DEFAULT_RIR_BANK_PATH
from SpeechToText.dataset import create_dataloaders
from SpeechToText.models.common.config import BaseOptimizerConfig, BaseTrainConfig, PrecisionType
from SpeechToText.models.common.train_factory import (
    apply_ctc_augment_banks,
    build_training_logger,
    configure_matmul_precision,
    run_training,
    wire_data_filter_from_model,
)
from SpeechToText.models.ctc_attention.lit import LitFastConformerCTCAttention
from SpeechToText.models.ctc_attention.model import FastConformerCTCAttentionConfig

warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")


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

    wandb_project: str = os.getenv("WANDB_PROJECT", "multilingual_asr_ctc_attn")
    precision: PrecisionType = "bf16-mixed"


def main(config: TrainConfig) -> None:
    pl.seed_everything(config.seed, workers=True)
    configure_matmul_precision()
    wire_data_filter_from_model(config)

    train_loader, val_loader, sp = create_dataloaders(
        data_config=config.data,
        audio_config=config.audio_augment,
        audio_augment_start_epoch=config.audio_augment_start_epoch,
    )

    noise_bank, rir_bank = apply_ctc_augment_banks(config)
    model = LitFastConformerCTCAttention(
        config=config,
        sp=sp,
        rir_bank=rir_bank,
        noise_bank=noise_bank,
    )

    run_training(
        config=config,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        logger=build_training_logger(config),
        checkpoint_dir=config.checkpoint_dir,
    )


if __name__ == "__main__":
    main(tyro.cli(TrainConfig))

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, cast

import lightning.pytorch as pl
import tyro
from dotenv import load_dotenv
from lightning.pytorch.loggers import WandbLogger

from SpeechToText.dataset import create_dataloaders
from SpeechToText.models.common.config import BaseOptimizerConfig, BaseTrainConfig
from SpeechToText.models.common.train_factory import (
    configure_matmul_precision,
    run_training,
    wire_data_filter_from_model,
)
from SpeechToText.models.tdt.lit import LitFastConformerTDT
from SpeechToText.models.tdt.model import FastConformerTDTConfig

load_dotenv()


@dataclass
class TrainConfig(BaseTrainConfig):
    checkpoint_dir: str = "./checkpoints/tdt"

    model: FastConformerTDTConfig = field(default_factory=FastConformerTDTConfig)
    optimizer: BaseOptimizerConfig = field(default_factory=BaseOptimizerConfig)

    blank_id: int = 0
    rnnt_clamp: float = 1.0
    fused_log_softmax: bool = False
    val_max_symbols_per_t: int = 4
    label_smoothing: float = 0.1

    wandb_project: str = os.getenv("WANDB_PROJECT", "multilingual_asr_tdt")


def main(config: TrainConfig) -> None:
    pl.seed_everything(config.seed, workers=True)
    configure_matmul_precision()
    wire_data_filter_from_model(config)

    train_loader, val_loader, sp = create_dataloaders(
        data_config=config.data,
        audio_config=config.audio_augment,
        audio_augment_start_epoch=config.audio_augment_start_epoch,
    )

    vocab_size = int(sp.get_piece_size()) + 1
    model_config = cast(Any, config.model)
    model_config.blank_id = int(config.blank_id)
    model_config.decoder.vocab_size = vocab_size
    model_config.joint.vocab_size = vocab_size
    model_config.decoder.d_model = int(model_config.encoder.d_model)
    model_config.joint.enc_d = int(model_config.encoder.d_model)
    model_config.joint.pred_d = int(model_config.encoder.d_model)

    lit = LitFastConformerTDT(config, sp=sp, vocab_size=vocab_size)

    wandb_logger = WandbLogger(
        project=config.wandb_project, name=config.wandb_run_name, log_model=False
    )

    run_training(
        config=config,
        model=lit,
        train_loader=train_loader,
        val_loader=val_loader,
        logger=wandb_logger,
        checkpoint_dir=config.checkpoint_dir,
    )


if __name__ == "__main__":
    main(tyro.cli(TrainConfig))

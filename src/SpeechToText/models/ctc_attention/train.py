from __future__ import annotations

import os
from dataclasses import dataclass, field

import lightning.pytorch as pl
import torch
import tyro
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from SpeechToText.dataset import create_dataloaders
from SpeechToText.models.common.config import BaseOptimizerConfig, BaseTrainConfig
from SpeechToText.models.ctc_attention.lit import LitFastConformerCTCAttention
from SpeechToText.models.ctc_attention.model import FastConformerCTCAttentionConfig


@dataclass
class TrainConfig(BaseTrainConfig):
    checkpoint_dir: str = "./checkpoints/ctc_attention"

    model: FastConformerCTCAttentionConfig = field(default_factory=FastConformerCTCAttentionConfig)
    optimizer: BaseOptimizerConfig = field(default_factory=BaseOptimizerConfig)

    ctc_label_smoothing: float = 0.1
    aux_ctc_weight: float = 0.3
    ctc_weight: float = 0.3

    wandb_project: str = os.getenv("WANDB_PROJECT", "multilingual_asr_ctc_attn")


def main(config: TrainConfig) -> None:
    pl.seed_everything(config.seed, workers=True)

    train_loader, val_loader, sp = create_dataloaders(
        data_config=config.data,
        spec_config=config.spec_augment,
        audio_config=config.audio_augment,
        augment_start_epoch=config.augment_start_epoch,
    )

    sp_vocab_size = int(sp.get_piece_size())
    ctc_vocab_size = sp_vocab_size + 1
    blank_id = 0

    model = LitFastConformerCTCAttention(
        config,
        ctc_vocab_size=ctc_vocab_size,
        sp_vocab_size=sp_vocab_size,
        sp=sp,
        blank_id=blank_id,
    )

    wandb_logger = WandbLogger(
        project=config.wandb_project,
        name=config.wandb_run_name,
        log_model=False,
    )

    ckpt = ModelCheckpoint(
        dirpath=config.checkpoint_dir,
        monitor="val/wer/overall",
        mode="min",
        save_top_k=3,
        save_last=True,
        every_n_epochs=1,
        filename="{epoch:03d}-{val_wer_overall:.2f}",
    )

    trainer = pl.Trainer(
        max_epochs=config.max_epochs,
        logger=wandb_logger,
        callbacks=[ckpt, LearningRateMonitor(logging_interval="step")],
        accelerator="gpu" if torch.cuda.is_available() else "auto",
        devices=1,
        precision=config.precision,
        log_every_n_steps=config.log_every_n_steps,
        val_check_interval=config.val_check_interval,
        gradient_clip_val=config.gradient_clip_val,
        gradient_clip_algorithm="norm",
        accumulate_grad_batches=config.accumulate_grad_batches,
        num_sanity_val_steps=0,
        benchmark=True,
    )

    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=config.ckpt_path,
    )


if __name__ == "__main__":
    main(tyro.cli(TrainConfig))

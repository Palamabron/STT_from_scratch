from __future__ import annotations

import os
from dataclasses import dataclass, field

import lightning.pytorch as pl
import torch
import tyro
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader

from SpeechToText.augmentation import AudioAugmentConfig, SpecAugmentConfig
from SpeechToText.dataset import DataConfig, create_dataloaders
from SpeechToText.models.common import greedy_ctc_decode
from SpeechToText.models.common.config import BaseOptimizerConfig, BaseTrainConfig, PrecisionType
from SpeechToText.models.ctc.lit import LitFastConformerCTC
from SpeechToText.models.ctc.model import FastConformerCTCConfig

__all__ = [
    "AudioAugmentConfig",
    "DataConfig",
    "LitFastConformerCTC",
    "SpecAugmentConfig",
    "TrainConfig",
    "greedy_decoder",
]


class DatasetEpochSync(pl.Callback):
    def __init__(self, train_loader: DataLoader) -> None:
        super().__init__()
        self._ds = train_loader.dataset

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if hasattr(self._ds, "set_current_epoch"):
            self._ds.set_current_epoch(int(trainer.current_epoch))

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if hasattr(self._ds, "set_current_epoch"):
            self._ds.set_current_epoch(int(trainer.current_epoch))


@dataclass
class TrainConfig(BaseTrainConfig):
    checkpoint_dir: str = "./checkpoints/ctc"
    data: DataConfig
    model: FastConformerCTCConfig = field(default_factory=FastConformerCTCConfig)
    optimizer: BaseOptimizerConfig = field(default_factory=BaseOptimizerConfig)

    max_epochs: int = 50
    precision: PrecisionType = "32-true"
    log_every_n_steps: int = 10
    val_check_interval: float = 1.0
    accumulate_grad_batches: int = 1
    gradient_clip_val: float = 5.0

    ctc_label_smoothing: float = 0.1
    aux_ctc_weight: float = 0.3

    wandb_project: str = os.getenv("WANDB_PROJECT", "multilingual_asr_ctc")
    wandb_run_name: str | None = None


def greedy_decoder(
    log_probs: torch.Tensor,
    out_lengths: torch.Tensor,
    blank_id: int,
) -> list[list[int]]:
    return greedy_ctc_decode(log_probs, out_lengths, blank_id=blank_id)


def main(config: TrainConfig) -> None:
    pl.seed_everything(config.seed, workers=True)

    train_loader, val_loader, sp = create_dataloaders(
        data_config=config.data,
        spec_config=config.spec_augment,
        audio_config=config.audio_augment,
        augment_start_epoch=config.augment_start_epoch,
    )

    sp_vocab_size = int(sp.get_piece_size())
    vocab_size = sp_vocab_size + 1
    blank_id = 0

    model = LitFastConformerCTC(config, vocab_size=vocab_size, sp=sp, blank_id=blank_id)

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

    epoch_sync = DatasetEpochSync(train_loader)

    trainer = pl.Trainer(
        max_epochs=config.max_epochs,
        logger=wandb_logger,
        callbacks=[ckpt, epoch_sync, LearningRateMonitor(logging_interval="step")],
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

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)


if __name__ == "__main__":
    main(tyro.cli(TrainConfig))

from __future__ import annotations

from typing import Any, Protocol

import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import Callback, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import Logger
from torch.utils.data import DataLoader

from SpeechToText.augmentation import load_audio_bank
from SpeechToText.models.common.callbacks import DatasetEpochSync


class _HasEncoderConfig(Protocol):
    encoder: Any


class TrainRunConfig(Protocol):
    model: Any
    checkpoint_dir: str
    data: Any
    audio_augment: Any
    ckpt_path: str | None
    max_epochs: int
    precision: Any
    log_every_n_steps: int
    val_check_interval: float
    gradient_clip_val: float
    accumulate_grad_batches: int


MATMUL_PRECISION = "high"


def configure_matmul_precision() -> None:
    torch.set_float32_matmul_precision(MATMUL_PRECISION)
    if torch.cuda.is_available():
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)


def wire_data_filter_from_model(config: TrainRunConfig) -> None:
    config.data.filter.subsampling_factor = int(config.model.encoder.subsampling_factor)


def apply_ctc_augment_banks(config: TrainRunConfig) -> tuple[tuple[torch.Tensor, ...] | None, ...]:
    """Load optional MUSAN/RIR banks and tune augmentation probabilities for CTC training."""
    sample_rate = config.data.features.sample_rate
    noise_bank = load_audio_bank(getattr(config, "musan_path", None), sample_rate=sample_rate)
    rir_bank = load_audio_bank(
        getattr(config, "rirs_path", None),
        sample_rate=sample_rate,
        normalize_rirs=True,
        max_rir_len_sec=0.5,
    )

    config.audio_augment.bg_noise_prob = 0.4 if noise_bank else 0.0
    config.audio_augment.rir_prob = 0.3 if rir_bank else 0.0
    return noise_bank, rir_bank


def build_checkpoint_callback(
    checkpoint_dir: str,
    *,
    monitor: str = "val/wer/overall",
) -> ModelCheckpoint:
    return ModelCheckpoint(
        dirpath=checkpoint_dir,
        monitor=monitor,
        mode="min",
        save_top_k=3,
        save_last=True,
        every_n_epochs=1,
        filename="{epoch:03d}-{val_wer_overall:.2f}",
    )


def build_trainer(
    config: TrainRunConfig,
    *,
    train_loader: DataLoader,
    logger: Logger,
    extra_callbacks: list[Callback] | None = None,
    monitor: str = "val/wer/overall",
    checkpoint_dir: str | None = None,
) -> pl.Trainer:
    checkpoint_cb = build_checkpoint_callback(
        str(checkpoint_dir or config.checkpoint_dir),
        monitor=monitor,
    )
    epoch_sync = DatasetEpochSync(train_loader)
    callbacks: list[Callback] = [
        checkpoint_cb,
        epoch_sync,
        LearningRateMonitor(logging_interval="step"),
    ]
    if extra_callbacks:
        callbacks.extend(extra_callbacks)

    return pl.Trainer(
        max_epochs=config.max_epochs,
        logger=logger,
        callbacks=callbacks,
        accelerator="gpu" if torch.cuda.is_available() else "auto",
        devices=1,
        precision=config.precision,
        log_every_n_steps=config.log_every_n_steps,
        val_check_interval=config.val_check_interval,
        gradient_clip_val=config.gradient_clip_val,
        gradient_clip_algorithm="norm",
        accumulate_grad_batches=config.accumulate_grad_batches,
        num_sanity_val_steps=2,
        benchmark=True,
    )


def run_training(
    *,
    config: TrainRunConfig,
    model: pl.LightningModule,
    train_loader: DataLoader,
    val_loader: DataLoader,
    logger: Logger,
    extra_callbacks: list[Callback] | None = None,
    monitor: str = "val/wer/overall",
    checkpoint_dir: str | None = None,
) -> None:
    trainer = build_trainer(
        config,
        train_loader=train_loader,
        logger=logger,
        extra_callbacks=extra_callbacks,
        monitor=monitor,
        checkpoint_dir=checkpoint_dir,
    )
    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=config.ckpt_path,
    )

from __future__ import annotations

import os

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import warnings
from dataclasses import dataclass, field

import tyro
from lightning.pytorch.loggers import WandbLogger

from SpeechToText.augmentation import DEFAULT_NOISE_BANK_PATH, DEFAULT_RIR_BANK_PATH
from SpeechToText.dataset import create_dataloaders
from SpeechToText.models.common.config import BaseOptimizerConfig, BaseTrainConfig, PrecisionType
from SpeechToText.models.common.train_factory import (
    apply_ctc_augment_banks,
    configure_matmul_precision,
    run_training,
    wire_data_filter_from_model,
)
from SpeechToText.models.ctc.lit import LitFastConformerCTC
from SpeechToText.models.ctc.model import FastConformerCTCConfig

warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")


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


def main(config: TrainConfig) -> None:
    """Train a Fast-Conformer CTC model from CLI configuration."""
    import lightning.pytorch as pl

    pl.seed_everything(config.seed, workers=True)
    configure_matmul_precision()

    noise_bank, rir_bank = apply_ctc_augment_banks(config)
    wire_data_filter_from_model(config)

    train_loader, val_loader, sp = create_dataloaders(
        data_config=config.data,
        audio_config=config.audio_augment,
        audio_augment_start_epoch=config.audio_augment_start_epoch,
    )

    if int(sp.get_piece_size()) <= 0:
        raise ValueError("Tokenizer vocab size must be positive")

    model = LitFastConformerCTC(
        config=config,
        sp=sp,
        rir_bank=rir_bank,
        noise_bank=noise_bank,
    )

    wandb_logger = WandbLogger(
        project=config.wandb_project,
        name=config.wandb_run_name,
    )

    run_training(
        config=config,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        logger=wandb_logger,
        checkpoint_dir=config.checkpoint_dir,
    )


if __name__ == "__main__":
    main(tyro.cli(TrainConfig))

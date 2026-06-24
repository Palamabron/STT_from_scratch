from __future__ import annotations

import warnings

import lightning.pytorch as pl
import tyro

from SpeechToText.dataset import create_dataloaders
from SpeechToText.models.common.train_factory import (
    apply_ctc_augment_banks,
    build_training_logger,
    configure_matmul_precision,
    run_training,
    wire_data_filter_from_model,
)
from SpeechToText.models.ctc_attention.config import TrainConfig
from SpeechToText.models.ctc_attention.lit import LitFastConformerCTCAttention

warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")


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

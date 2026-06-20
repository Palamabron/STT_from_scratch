from __future__ import annotations

import os
import warnings

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import tyro
from dotenv import load_dotenv

from SpeechToText.dataset import create_dataloaders
from SpeechToText.models.common.train_factory import (
    apply_ctc_augment_banks,
    build_training_logger,
    configure_matmul_precision,
    run_training,
    wire_data_filter_from_model,
)
from SpeechToText.models.ctc.config import TrainConfig
from SpeechToText.models.ctc.lit import LitFastConformerCTC

load_dotenv()

warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")


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

    wandb_logger = build_training_logger(config)

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

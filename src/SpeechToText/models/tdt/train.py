from __future__ import annotations

from typing import Any, cast

import lightning.pytorch as pl
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
from SpeechToText.models.tdt.config import TrainConfig
from SpeechToText.models.tdt.lit import LitFastConformerTDT

load_dotenv()


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
    model_config.joint.use_tdt = bool(config.use_tdt)
    model_config.decoder.d_model = int(model_config.encoder.d_model)
    model_config.joint.enc_d = int(model_config.encoder.d_model)
    model_config.joint.pred_d = int(model_config.encoder.d_model)
    model_config.joint_fused_batch_size = config.joint_fused_batch_size

    noise_bank, rir_bank = apply_ctc_augment_banks(config)
    lit = LitFastConformerTDT(
        config,
        sp=sp,
        vocab_size=vocab_size,
        rir_bank=rir_bank,
        noise_bank=noise_bank,
    )

    run_training(
        config=config,
        model=lit,
        train_loader=train_loader,
        val_loader=val_loader,
        logger=build_training_logger(config),
        checkpoint_dir=config.checkpoint_dir,
    )


if __name__ == "__main__":
    main(tyro.cli(TrainConfig))

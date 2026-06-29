from __future__ import annotations

from typing import Any, cast

import lightning.pytorch as pl
import torch
from sentencepiece import SentencePieceProcessor

from SpeechToText.augmentation import GPUAudioAugmentation, SpecAugment
from SpeechToText.features import WaveformFeaturizer
from SpeechToText.models.shared.config import SharedASRConfig
from SpeechToText.models.shared.model import SharedFastConformerASR


class LitSharedFastConformerASR(pl.LightningModule):
    """Lightning wrapper for shared multi-head ASR (inference and future training)."""

    def __init__(
        self,
        config: Any,
        sp: SentencePieceProcessor,
        rir_bank: tuple[torch.Tensor, ...] | None = None,
        noise_bank: tuple[torch.Tensor, ...] | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.sp = sp

        sp_vocab = int(sp.get_piece_size())
        self.vocab_size = sp_vocab + 1
        self.blank_id = 0
        self.pad_id = sp.pad_id() + 1
        self.bos_id = sp.bos_id() + 1
        self.eos_id = sp.eos_id() + 1

        shared_cfg = cast(SharedASRConfig, config.model)

        spec_augment = SpecAugment(
            config.spec_augment, augment_start_epoch=config.spec_augment_start_epoch
        )
        gpu_augment = GPUAudioAugmentation(
            config.audio_augment,
            rir_bank,
            noise_bank,
            augment_start_epoch=config.audio_augment_start_epoch,
        )
        self.featurizer = WaveformFeaturizer(
            config.data.features,
            spec_augment=spec_augment,
            gpu_augment=gpu_augment,
        )

        self.net = SharedFastConformerASR(
            shared_cfg,
            vocab_size=self.vocab_size,
            blank_id=self.blank_id,
            pad_id=self.pad_id,
            bos_id=self.bos_id,
            eos_id=self.eos_id,
        )
        self.save_hyperparameters(ignore=["sp", "rir_bank", "noise_bank"])

    def on_fit_start(self) -> None:
        self.featurizer = self.featurizer.to(self.device)

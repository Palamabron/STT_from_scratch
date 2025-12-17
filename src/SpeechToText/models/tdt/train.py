from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, cast

import lightning.pytorch as pl
import torch
import tyro
from dotenv import load_dotenv
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from loguru import logger
from sentencepiece import SentencePieceProcessor

from SpeechToText.models.common.config import BaseOptimizerConfig, BaseTrainConfig
from SpeechToText.models.common.ctc_decode import ctc_ids_to_texts_spm, greedy_ctc_decode
from SpeechToText.models.common.metrics_wer_cer import wer_cer_by_lang
from SpeechToText.models.tdt.model import FastConformerTDT, FastConformerTDTConfig
from SpeechToText.models.tdt.steps import compute_tdt_losses
from SpeechToText.typing import TrainBatch, ValBatch

from ...dataset import create_dataloaders

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


class LitFastConformerTDT(pl.LightningModule):
    def __init__(self, config: TrainConfig, sp: SentencePieceProcessor) -> None:
        super().__init__()
        self.config = config
        self.sp = sp

        model_config: Any = config.model
        self.blank_id = int(model_config.blank_id)

        self.pad_id = int(model_config.decoder.vocab_size - 1)
        self.model = FastConformerTDT(model_config)

        self.example_buffer: dict[str, list[tuple[str, str]]] = {"en": [], "pl": []}
        self.examples_per_lang: int = 2

        self.save_hyperparameters(ignore=["sp"])

    def training_step(self, batch: TrainBatch, batch_idx: int) -> torch.Tensor:
        feats = batch["features"]
        feat_lengths = batch["feature_lengths"]
        targets = batch["targets"]
        target_lengths = batch["target_lengths"]

        out = self.model(
            feats=feats,
            feat_lengths=feat_lengths,
            targets_concat=targets,
            target_lengths=target_lengths,
        )

        targets_padded = self.model.pad_targets_from_concat(
            targets_concat=targets,
            target_lengths=target_lengths,
            pad_id=self.pad_id,
        )

        losses = compute_tdt_losses(
            logits=out.logits,
            out_lengths=out.out_lengths,
            targets_padded=targets_padded,
            target_lengths=target_lengths,
            blank_id=self.blank_id,
            label_smoothing=self.config.label_smoothing,
            rnnt_clamp=self.config.rnnt_clamp,
            fused_log_softmax=self.config.fused_log_softmax,
        )

        loss_total = cast(torch.Tensor, losses.total)

        self.log_dict(
            {
                "train/loss": loss_total,
                "train/loss_rnnt": losses.rnnt,
                "train/loss_lsm": losses.lsm,
            },
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            batch_size=feats.size(0),
        )

        if not torch.isfinite(loss_total):
            raise RuntimeError("NaN/Inf in loss.")
        return loss_total

    def validation_step(self, batch: ValBatch, batch_idx: int) -> dict[str, float]:
        """
        For now: quick greedy decode via "CTC-like" argmax over (U=0) slice to get rough WER/CER.
        Proper transducer beam search can be added later.
        """
        feats = batch["features"]
        feat_lengths = batch["feature_lengths"]
        targets = batch["targets"]
        target_lengths = batch["target_lengths"]
        texts = batch["text"]
        langs = batch.get("language")

        out = self.model(
            feats=feats,
            feat_lengths=feat_lengths,
            targets_concat=targets,
            target_lengths=target_lengths,
        )

        # log_probs: [B,T,U,V] -> [B,T,V]
        lp = out.log_probs[:, :, 0, :]
        decoded = greedy_ctc_decode(lp, out.out_lengths, blank_id=self.blank_id)
        pred_texts = ctc_ids_to_texts_spm(self.sp, decoded)

        metrics = wer_cer_by_lang(texts, pred_texts, langs)
        bs = len(texts)

        self.log_dict(
            {
                "val/wer/overall": metrics["wer/overall"],
                "val/cer/overall": metrics["cer/overall"],
                **(
                    {"val/wer/en": metrics["wer/en"], "val/cer/en": metrics["cer/en"]}
                    if "wer/en" in metrics
                    else {}
                ),
                **(
                    {"val/wer/pl": metrics["wer/pl"], "val/cer/pl": metrics["cer/pl"]}
                    if "wer/pl" in metrics
                    else {}
                ),
            },
            prog_bar=True,
            on_epoch=True,
            batch_size=bs,
        )

        for lang, ref, hyp in zip((langs or ["unknown"] * bs), texts, pred_texts, strict=True):
            if (
                lang in self.example_buffer
                and len(self.example_buffer[lang]) < self.examples_per_lang
            ):
                self.example_buffer[lang].append((ref, hyp))

        return {"wer": metrics["wer/overall"], "cer": metrics["cer/overall"]}

    def on_validation_epoch_end(self) -> None:
        for lang in ["en", "pl"]:
            ex = self.example_buffer.get(lang, [])
            if not ex:
                continue
            logger.info(f"[VAL][{lang}] --- examples epoch {self.current_epoch} ---")
            for ref, hyp in ex:
                logger.info(f"[VAL][{lang}] REF: {ref}")
                logger.info(f"[VAL][{lang}] HYP: {hyp}")
        self.example_buffer = {"en": [], "pl": []}


def main(config: TrainConfig) -> None:
    pl.seed_everything(config.seed, workers=True)

    train_loader, val_loader, sp = create_dataloaders(
        config.data,
        config.spec_augment,
        config.audio_augment,
        augment_start_epoch=config.augment_start_epoch,
    )

    sp_vocab = int(sp.get_piece_size())
    vocab_size = sp_vocab + 1

    model_config: Any = config.model
    model_config.decoder.vocab_size = vocab_size
    model_config.joint.vocab_size = vocab_size
    model_config.blank_id = 0

    lit = LitFastConformerTDT(config, sp=sp)

    wandb_logger = WandbLogger(
        project=config.wandb_project, name=config.wandb_run_name, log_model=False
    )
    checkpoint_cb = ModelCheckpoint(
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
        callbacks=[checkpoint_cb, LearningRateMonitor(logging_interval="step")],
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
        lit, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=config.ckpt_path
    )


if __name__ == "__main__":
    config = tyro.cli(TrainConfig)
    main(config)

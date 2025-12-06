from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import lightning.pytorch as pl
import torch
import torch.nn as nn
import tyro
from dotenv import load_dotenv
from jiwer import cer as jiwer_cer
from jiwer import wer as jiwer_wer
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from loguru import logger

from .augmentation import AudioAugmentConfig, SpecAugmentConfig
from .dataset import DataConfig, create_dataloaders
from .model import FastConformerCTC, FastConformerCTCConfig

torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True
load_dotenv()


@dataclass
class OptimizerConfig:
    learning_rate: float = 5e-4
    betas: tuple[float, float] = (0.9, 0.98)
    epsilon: float = 1e-9
    weight_decay: float = 1e-3


@dataclass
class TrainConfig:
    data: DataConfig
    model: FastConformerCTCConfig = field(default_factory=FastConformerCTCConfig)
    optim: OptimizerConfig = field(default_factory=OptimizerConfig)
    checkpoint_dir: str = "./checkpoints"
    max_epochs: int = 50
    accumulate_grad_batches: int = 1
    gradient_clip_val: float = 5.0
    val_check_interval: float = 1.0
    log_every_n_steps: int = 10
    precision: str = "32-true"
    spec_augment: SpecAugmentConfig = field(default_factory=SpecAugmentConfig)
    audio_augment: AudioAugmentConfig = field(default_factory=AudioAugmentConfig)
    augment_start_epoch: int = 3
    wandb_project: str = os.getenv("WANDB_PROJECT", "multilingual_asr")
    wandb_run_name: str | None = None


def greedy_decoder(
    log_probs: torch.Tensor,
    out_lengths: torch.Tensor,
    blank_id: int,
) -> list[list[int]]:
    """
    log_probs: (B, T, V)
    out_lengths: (B,)
    """
    preds = torch.argmax(log_probs, dim=-1).cpu()  # (B, T)
    out_lengths = out_lengths.cpu()

    decoded: list[list[int]] = []
    for seq, L in zip(preds, out_lengths, strict=True):
        T = int(L.item())
        prev = -1
        tokens: list[int] = []

        for p in seq[:T]:
            p_int = int(p.item())
            if p_int != prev and p_int != blank_id:
                tokens.append(p_int)
            prev = p_int

        decoded.append(tokens)

    return decoded


class LitFastConformerCTC(pl.LightningModule):
    def __init__(self, cfg: TrainConfig, vocab_size: int, sp, blank_id: int = 0) -> None:
        super().__init__()
        self.cfg = cfg
        self.sp = sp
        self.blank_id = blank_id
        self.augment_start_epoch = cfg.augment_start_epoch

        self.save_hyperparameters(ignore=["sp"])

        self.model = FastConformerCTC(
            cfg.model,
            vocab_size=vocab_size,
            blank_id=blank_id,
        )
        self.ctc_loss = nn.CTCLoss(
            blank=blank_id,
            zero_infinity=True,
            reduction="mean",
        )

    def forward(self, feats: torch.Tensor, feat_lengths: torch.Tensor):
        return self.model(feats, feat_lengths)

    def configure_optimizers(self):
        opt_cfg = self.cfg.optim
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=opt_cfg.learning_rate,
            betas=opt_cfg.betas,
            eps=opt_cfg.epsilon,
            weight_decay=opt_cfg.weight_decay,
        )
        # Na start bez schedulerów – mniej ruchomych części.
        return optimizer

    def on_train_epoch_start(self) -> None:
        epoch = int(self.current_epoch)
        if self.trainer is None:
            return

        train_dl = self.trainer.train_dataloader
        ds = getattr(train_dl, "dataset", None)
        if ds is not None and hasattr(ds, "set_current_epoch"):
            ds.set_current_epoch(epoch)

    def training_step(self, batch: dict[str, Any], batch_idx: int):
        feats = batch["features"]  # (B, T, F)
        feat_lengths = batch["feature_lengths"]
        targets = batch["targets"]  # (sum_L,)
        target_lengths = batch["target_lengths"]

        log_probs, out_lengths = self(feats, feat_lengths)  # (B, T', V)
        log_probs_t = log_probs.transpose(0, 1)  # (T', B, V)

        if (out_lengths < target_lengths).any():
            diff = (out_lengths - target_lengths).min()
            raise RuntimeError(
                f"CTC input length smaller than target length. Min(input_len - target_len) = {diff}"
            )

        loss = self.ctc_loss(log_probs_t, targets, out_lengths, target_lengths)

        self.log(
            "train_loss",
            loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            sync_dist=False,
            batch_size=feats.size(0),
        )
        return loss

    def validation_step(self, batch: dict[str, Any], batch_idx: int):
        feats = batch["features"]
        feat_lengths = batch["feature_lengths"]
        targets = batch["targets"]
        target_lengths = batch["target_lengths"]
        texts = batch["text"]
        langs = batch.get("language", ["unknown"] * len(texts))

        log_probs, out_lengths = self(feats, feat_lengths)
        log_probs_t = log_probs.transpose(0, 1)

        if (out_lengths < target_lengths).any():
            diff = (out_lengths - target_lengths).min()
            raise RuntimeError(
                "[VAL] CTC input length smaller than target length. "
                f"Min(input_len - target_len) = {diff}"
            )

        loss = self.ctc_loss(log_probs_t, targets, out_lengths, target_lengths)

        decoded_ids = greedy_decoder(
            log_probs,
            out_lengths,
            blank_id=self.blank_id,
        )

        pred_texts: list[str] = []
        for seq in decoded_ids:
            sp_ids = [i - 1 for i in seq if i > 0]
            if len(sp_ids) == 0:
                pred_texts.append("")
            else:
                pred_texts.append(self.sp.decode_ids(sp_ids))

        batch_wer = jiwer_wer(texts, pred_texts)
        batch_cer = jiwer_cer(texts, pred_texts)
        bs = len(texts)

        self.log(
            "val_loss",
            loss,
            prog_bar=True,
            on_epoch=True,
            sync_dist=False,
            batch_size=bs,
        )
        self.log(
            "val_wer",
            batch_wer,
            prog_bar=True,
            on_epoch=True,
            sync_dist=False,
            batch_size=bs,
        )
        self.log(
            "val_cer",
            batch_cer,
            prog_bar=False,
            on_epoch=True,
            sync_dist=False,
            batch_size=bs,
        )

        # EN / PL
        en_refs, en_hyps = [], []
        pl_refs, pl_hyps = [], []

        for lang, ref, hyp in zip(langs, texts, pred_texts, strict=True):
            if lang == "en":
                en_refs.append(ref)
                en_hyps.append(hyp)
            elif lang == "pl":
                pl_refs.append(ref)
                pl_hyps.append(hyp)

        if en_refs:
            en_wer = jiwer_wer(en_refs, en_hyps)
            en_cer = jiwer_cer(en_refs, en_hyps)
            self.log(
                "val_wer_en",
                en_wer,
                prog_bar=False,
                on_epoch=True,
                batch_size=len(en_refs),
            )
            self.log(
                "val_cer_en",
                en_cer,
                prog_bar=False,
                on_epoch=True,
                batch_size=len(en_refs),
            )

        if pl_refs:
            pl_wer = jiwer_wer(pl_refs, pl_hyps)
            pl_cer = jiwer_cer(pl_refs, pl_hyps)
            self.log(
                "val_wer_pl",
                pl_wer,
                prog_bar=False,
                on_epoch=True,
                batch_size=len(pl_refs),
            )
            self.log(
                "val_cer_pl",
                pl_cer,
                prog_bar=False,
                on_epoch=True,
                batch_size=len(pl_refs),
            )

        if batch_idx == 0:
            self._log_val_examples(langs, texts, pred_texts)

        return {"wer": batch_wer, "cer": batch_cer}

    def _log_val_examples(
        self,
        langs: list[str],
        refs: list[str],
        hyps: list[str],
    ) -> None:
        to_log: list[tuple[str, str, str]] = []
        en_needed = 2
        pl_needed = 2

        for lang, ref, hyp in zip(langs, refs, hyps, strict=True):
            if lang == "en" and en_needed > 0:
                to_log.append((lang, ref, hyp))
                en_needed -= 1
            elif lang == "pl" and pl_needed > 0:
                to_log.append((lang, ref, hyp))
                pl_needed -= 1

            if en_needed == 0 and pl_needed == 0:
                break

        for lang, ref, hyp in to_log:
            logger.info(f"[VAL][{lang}] REF: {ref}")
            logger.info(f"[VAL][{lang}] HYP: {hyp}")


def main(cfg: TrainConfig):
    pl.seed_everything(42, workers=True)

    train_loader, val_loader, sp = create_dataloaders(
        cfg.data,
        cfg.spec_augment,
        cfg.audio_augment,
        augment_start_epoch=cfg.augment_start_epoch,
    )

    sp_vocab_size = sp.GetPieceSize()
    vocab_size = sp_vocab_size + 1
    blank_id = 0

    model = LitFastConformerCTC(
        cfg,
        vocab_size=vocab_size,
        sp=sp,
        blank_id=blank_id,
    )

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Number of trainable parameters: {num_params}")
    logger.info(
        "Model d_model=%s, n_layers=%s, num_heads=%s",
        cfg.model.d_model,
        cfg.model.n_layers,
        cfg.model.num_heads,
    )
    logger.info(f"Vocab size: {vocab_size}")
    logger.info(f"Precision: {cfg.precision}")

    wandb_logger = WandbLogger(
        project=cfg.wandb_project,
        name=cfg.wandb_run_name,
        log_model=False,
    )

    checkpoint_cb = ModelCheckpoint(
        dirpath=cfg.checkpoint_dir,
        monitor="val_wer",
        mode="min",
        save_top_k=3,
        verbose=True,
        filename="{epoch:03d}-{val_wer:.2f}",
        save_last=True,
        every_n_epochs=1,
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")

    trainer = pl.Trainer(
        max_epochs=cfg.max_epochs,
        logger=wandb_logger,
        callbacks=[checkpoint_cb, lr_monitor],
        accelerator="gpu" if torch.cuda.is_available() else "auto",
        devices=1,
        precision=cfg.precision,
        log_every_n_steps=cfg.log_every_n_steps,
        val_check_interval=cfg.val_check_interval,
        gradient_clip_val=cfg.gradient_clip_val,
        gradient_clip_algorithm="norm",
        accumulate_grad_batches=cfg.accumulate_grad_batches,
        num_sanity_val_steps=0,
        benchmark=True,
        enable_progress_bar=True,
        enable_model_summary=True,
    )

    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )


if __name__ == "__main__":
    cfg = tyro.cli(TrainConfig)
    main(cfg)

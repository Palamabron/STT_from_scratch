from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Any, Literal, cast

import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import tyro
from dotenv import load_dotenv
from jiwer import cer as jiwer_cer
from jiwer import wer as jiwer_wer
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.utilities.types import OptimizerLRSchedulerConfig
from loguru import logger
from sentencepiece import SentencePieceProcessor

from .augmentation import AudioAugmentConfig, SpecAugmentConfig
from .dataset import DataConfig, create_dataloaders
from .model import FastConformerCTC, FastConformerCTCConfig

torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True
load_dotenv()


PrecisionType = Literal[
    "64",
    "32",
    "16",
    "64-true",
    "32-true",
    "16-true",
    "16-mixed",
    "bf16",
    "bf16-mixed",
    "bf16-true",
    "transformer-engine",
    "transformer-engine-float16",
]


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
    precision: PrecisionType = "32-true"

    ctc_label_smoothing: float = 0.1
    aux_ctc_weight: float = 0.3
    spec_augment: SpecAugmentConfig = field(default_factory=SpecAugmentConfig)
    audio_augment: AudioAugmentConfig = field(default_factory=AudioAugmentConfig)
    augment_start_epoch: int = 3

    wandb_project: str = os.getenv("WANDB_PROJECT", "multilingual_asr")
    wandb_run_name: str | None = None

    optim_warmup_steps: int = 5000
    optim_lr_peak: float = 0.002
    scheduler_restart_interval: int = 50


def greedy_decoder(
    log_probs: torch.Tensor,
    out_lengths: torch.Tensor,
    blank_id: int,
) -> list[list[int]]:
    preds = torch.argmax(log_probs, dim=-1).cpu()
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
    def __init__(
        self, cfg: TrainConfig, vocab_size: int, sp: SentencePieceProcessor, blank_id: int = 0
    ) -> None:
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

        self.example_buffer: dict[str, list[tuple[str, str]]] = {
            "en": [],
            "pl": [],
        }
        self.examples_per_lang: int = 2

    def ctc_loss_with_label_smoothing(
        self,
        log_probs_t: torch.Tensor,
        targets: torch.Tensor,
        input_lengths: torch.Tensor,
        target_lengths: torch.Tensor,
        epsilon: float,
    ) -> torch.Tensor:
        log_probs_t = log_probs_t.float()
        input_lengths = input_lengths.to(dtype=torch.long)
        target_lengths = target_lengths.to(dtype=torch.long)

        with torch.autocast(device_type="cuda", enabled=False):
            base_loss = F.ctc_loss(
                log_probs_t,
                targets,
                input_lengths,
                target_lengths,
                reduction="mean",
                blank=self.blank_id,
                zero_infinity=True,
            )

            if epsilon <= 0.0:
                return base_loss

            T, B, V = log_probs_t.shape
            non_blank = V - 1
            if non_blank <= 0:
                return base_loss

            uniform_neglog = torch.log(
                torch.tensor(
                    float(non_blank),
                    device=log_probs_t.device,
                    dtype=log_probs_t.dtype,
                )
            )

            smoothing_loss = uniform_neglog
            return (1.0 - epsilon) * base_loss + epsilon * smoothing_loss

    def forward(
        self,
        feats: torch.Tensor,
        feat_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        log_probs, out_lengths, aux_log_probs = cast(
            tuple[torch.Tensor, torch.Tensor, torch.Tensor],
            self.model(feats, feat_lengths),
        )
        return log_probs, out_lengths, aux_log_probs

    def configure_optimizers(self) -> OptimizerLRSchedulerConfig:
        opt_cfg = self.cfg.optim
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=opt_cfg.learning_rate,
            betas=opt_cfg.betas,
            eps=opt_cfg.epsilon,
            weight_decay=opt_cfg.weight_decay,
        )

        max_epochs = self.cfg.max_epochs
        warmup_epochs = 5
        eta_min_factor = 0.1

        def lr_lambda(epoch: int) -> float:
            if epoch < warmup_epochs:
                return float(epoch + 1) / float(warmup_epochs)

            progress = (epoch - warmup_epochs) / max(1.0, float(max_epochs - warmup_epochs))
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return eta_min_factor + (1.0 - eta_min_factor) * cosine

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def on_train_epoch_start(self) -> None:
        epoch = int(self.current_epoch)
        if self.trainer is None:
            return

        train_dl = self.trainer.train_dataloader
        ds = getattr(train_dl, "dataset", None)
        if ds is not None and hasattr(ds, "set_current_epoch"):
            ds.set_current_epoch(epoch)

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        feats = batch["features"]
        feat_lengths = batch["feature_lengths"]
        targets = batch["targets"]
        target_lengths = batch["target_lengths"]

        log_probs, out_lengths, aux_log_probs = self(feats, feat_lengths)
        log_probs_t = log_probs.transpose(0, 1)
        aux_log_probs_t = aux_log_probs.transpose(0, 1)

        if (out_lengths < target_lengths).any():
            diff = (out_lengths - target_lengths).min()
            raise RuntimeError(
                f"CTC input length smaller than target length. Min(input_len - target_len) = {diff}"
            )

        main_loss = self.ctc_loss_with_label_smoothing(
            log_probs_t,
            targets,
            out_lengths,
            target_lengths,
            epsilon=self.cfg.ctc_label_smoothing,
        )

        if self.cfg.aux_ctc_weight > 0.0:
            aux_loss = self.ctc_loss_with_label_smoothing(
                aux_log_probs_t,
                targets,
                out_lengths,
                target_lengths,
                epsilon=self.cfg.ctc_label_smoothing,
            )
            loss = main_loss + self.cfg.aux_ctc_weight * aux_loss
        else:
            aux_loss = torch.tensor(0.0, device=main_loss.device)
            loss = main_loss

        self.log(
            "train/loss",
            loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            sync_dist=False,
            batch_size=feats.size(0),
        )
        self.log(
            "train/loss_main",
            main_loss,
            prog_bar=False,
            on_step=True,
            on_epoch=True,
            sync_dist=False,
            batch_size=feats.size(0),
        )
        self.log(
            "train/loss_aux",
            aux_loss,
            prog_bar=False,
            on_step=True,
            on_epoch=True,
            sync_dist=False,
            batch_size=feats.size(0),
        )

        if not torch.isfinite(loss):
            logger.error(
                f"[NaN] loss={loss}, main={main_loss}, aux={aux_loss}, "
                f"step={self.global_step}, epoch={self.current_epoch}"
            )
            raise RuntimeError("NaN in loss – aborting training.")

        return loss

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> dict[str, float]:
        feats = batch["features"]
        feat_lengths = batch["feature_lengths"]
        targets = batch["targets"]
        target_lengths = batch["target_lengths"]
        texts = batch["text"]
        langs = batch.get("language", ["unknown"] * len(texts))

        log_probs, out_lengths, _ = self(feats, feat_lengths)
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
            "val/loss",
            loss,
            prog_bar=True,
            on_epoch=True,
            sync_dist=False,
            batch_size=bs,
        )
        self.log(
            "val/wer/overall",
            batch_wer,
            prog_bar=True,
            on_epoch=True,
            sync_dist=False,
            batch_size=bs,
        )
        self.log(
            "val/cer/overall",
            batch_cer,
            prog_bar=False,
            on_epoch=True,
            sync_dist=False,
            batch_size=bs,
        )

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
                "val/wer/en",
                en_wer,
                prog_bar=False,
                on_epoch=True,
                batch_size=len(en_refs),
            )
            self.log(
                "val/cer/en",
                en_cer,
                prog_bar=False,
                on_epoch=True,
                batch_size=len(en_refs),
            )

        if pl_refs:
            pl_wer = jiwer_wer(pl_refs, pl_hyps)
            pl_cer = jiwer_cer(pl_refs, pl_hyps)
            self.log(
                "val/wer/pl",
                pl_wer,
                prog_bar=False,
                on_epoch=True,
                batch_size=len(pl_refs),
            )
            self.log(
                "val/cer/pl",
                pl_cer,
                prog_bar=False,
                on_epoch=True,
                batch_size=len(pl_refs),
            )

        for lang, ref, hyp in zip(langs, texts, pred_texts, strict=True):
            if lang in self.example_buffer:
                if len(self.example_buffer[lang]) < self.examples_per_lang:
                    self.example_buffer[lang].append((ref, hyp))

        return {"wer": batch_wer, "cer": batch_cer}

    def on_validation_epoch_end(self) -> None:
        if self.trainer is not None:
            metrics = self.trainer.callback_metrics

            def _get(name: str) -> float | None:
                if name in metrics:
                    return float(metrics[name])
                return None

            wer = _get("val/wer/overall")
            cer = _get("val/cer/overall")
            wer_en = _get("val/wer/en")
            cer_en = _get("val/cer/en")
            wer_pl = _get("val/wer/pl")
            cer_pl = _get("val/cer/pl")

            epoch = int(self.current_epoch)
            logger.info(
                f"[VAL][epoch={epoch}] WER={wer:.4f} CER={cer:.4f}"
            ) if wer is not None and cer is not None else None
            if wer_en is not None and cer_en is not None:
                logger.info(f"[VAL][epoch={epoch}][en] WER={wer_en:.4f} CER={cer_en:.4f}")
            if wer_pl is not None and cer_pl is not None:
                logger.info(f"[VAL][epoch={epoch}][pl] WER={wer_pl:.4f} CER={cer_pl:.4f}")

        for lang in ["en", "pl"]:
            examples = self.example_buffer.get(lang, [])
            if not examples:
                continue
            logger.info(f"[VAL][{lang}] --- examples for epoch {self.current_epoch} ---")
            for ref, hyp in examples:
                logger.info(f"[VAL][{lang}] REF: {ref}")
                logger.info(f"[VAL][{lang}] HYP: {hyp}")

        self.example_buffer = {"en": [], "pl": []}


def main(cfg: TrainConfig) -> None:
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
        f"Model d_model={cfg.model.d_model}, "
        f"n_layers={cfg.model.n_layers}, "
        f"num_heads={cfg.model.num_heads}"
    )
    logger.info(f"Vocab size: {vocab_size}")
    logger.info(f"Precision: {cfg.precision}")

    wandb_logger = WandbLogger(
        project=cfg.wandb_project,
        name=cfg.wandb_run_name,
        log_model=False,
    )

    wandb_logger.experiment.config.update(
        {
            "data": vars(cfg.data),
            "model": vars(cfg.model),
            "optim": vars(cfg.optim),
            "train": {
                "max_epochs": cfg.max_epochs,
                "precision": cfg.precision,
                "ctc_label_smoothing": cfg.ctc_label_smoothing,
                "aux_ctc_weight": cfg.aux_ctc_weight,
                "optim_lr_peak": cfg.optim_lr_peak,
                "scheduler_restart_interval": cfg.scheduler_restart_interval,
            },
        },
        allow_val_change=True,
    )

    checkpoint_cb = ModelCheckpoint(
        dirpath=cfg.checkpoint_dir,
        monitor="val/wer/overall",
        mode="min",
        save_top_k=3,
        verbose=True,
        filename="{epoch:03d}-{val_wer_overall:.2f}",
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

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Literal, NotRequired, TypedDict, cast

import lightning.pytorch as pl
import torch
import torch.nn as nn
import tyro
from dotenv import load_dotenv
from jiwer import cer as jiwer_cer
from jiwer import wer as jiwer_wer
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.utilities.types import OptimizerLRSchedulerConfig
from loguru import logger
from sentencepiece import SentencePieceProcessor

from SpeechToText.utils.losses import ctc_loss_with_label_smoothing

from ...augmentation import AudioAugmentConfig, SpecAugmentConfig
from ...dataset import DataConfig, create_dataloaders
from ..ctc.model import FastConformerCTCConfig
from .model import FastConformerCTCAttention

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


class TrainBatch(TypedDict):
    features: torch.Tensor
    feature_lengths: torch.Tensor
    targets: torch.Tensor
    target_lengths: torch.Tensor


class ValBatch(TypedDict):
    features: torch.Tensor
    feature_lengths: torch.Tensor
    targets: torch.Tensor
    target_lengths: torch.Tensor
    text: list[str]
    language: NotRequired[list[str]]


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

    checkpoint_dir: str = "./checkpoints/ctc_attn"
    max_epochs: int = 50
    accumulate_grad_batches: int = 1
    gradient_clip_val: float = 5.0
    val_check_interval: float = 1.0
    log_every_n_steps: int = 10
    precision: PrecisionType = "32-true"

    ctc_label_smoothing: float = 0.1
    aux_ctc_weight: float = 0.3
    ctc_weight: float = 0.3

    spec_augment: SpecAugmentConfig = field(default_factory=SpecAugmentConfig)
    audio_augment: AudioAugmentConfig = field(default_factory=AudioAugmentConfig)
    augment_start_epoch: int = 3

    wandb_project: str = os.getenv("WANDB_PROJECT", "multilingual_asr_ctc_attn")
    wandb_run_name: str | None = None

    optim_warmup_steps: int = 5000
    optim_lr_peak: float = 0.002
    scheduler_restart_interval: int = 50

    ckpt_path: str | None = None


def greedy_decoder(
    log_probs: torch.Tensor,
    out_lengths: torch.Tensor,
    blank_id: int,
) -> list[list[int]]:
    preds = torch.argmax(log_probs, dim=-1).cpu()
    out_lengths_cpu = out_lengths.cpu()

    decoded: list[list[int]] = []
    for seq, L in zip(preds, out_lengths_cpu, strict=True):
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


class LitFastConformerCTCAttention(pl.LightningModule):
    def __init__(
        self,
        cfg: TrainConfig,
        vocab_size: int,
        sp_vocab_size: int,
        sp: SentencePieceProcessor,
        blank_id: int = 0,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.sp = sp
        self.blank_id = blank_id
        self.sp_vocab_size = sp_vocab_size

        self.pad_id = sp_vocab_size
        self.bos_id = sp_vocab_size + 1
        self.eos_id = sp_vocab_size + 2

        self.augment_start_epoch = cfg.augment_start_epoch

        self.save_hyperparameters(ignore=["sp"])

        self.model = FastConformerCTCAttention(
            enc_cfg=cfg.model,
            ctc_vocab_size=vocab_size,
            sp_vocab_size=sp_vocab_size,
            blank_id=blank_id,
        )

        self.ctc_loss = nn.CTCLoss(
            blank=blank_id,
            zero_infinity=True,
            reduction="mean",
        )
        self.attn_loss = nn.NLLLoss(ignore_index=self.pad_id, reduction="mean")

        self.example_buffer: dict[str, list[tuple[str, str]]] = {"en": [], "pl": []}
        self.examples_per_lang: int = 2

    def build_decoder_sequences(
        self,
        targets: torch.Tensor,
        target_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        device = targets.device
        B = int(target_lengths.shape[0])
        max_len = int(target_lengths.max().item())

        dec_in = torch.full(
            (B, max_len + 1),
            self.pad_id,
            dtype=torch.long,
            device=device,
        )
        dec_out = torch.full(
            (B, max_len + 1),
            self.pad_id,
            dtype=torch.long,
            device=device,
        )

        offset = 0
        for b in range(B):
            L = int(target_lengths[b].item())
            if L == 0:
                dec_in[b, 0] = self.bos_id
                dec_out[b, 0] = self.eos_id
                continue

            seq = targets[offset : offset + L]
            offset += L

            piece_seq = seq - 1

            dec_in[b, 0] = self.bos_id
            dec_in[b, 1 : L + 1] = piece_seq

            dec_out[b, 0:L] = piece_seq
            dec_out[b, L] = self.eos_id

        return dec_in, dec_out

    def forward(
        self,
        feats: torch.Tensor,
        feat_lengths: torch.Tensor,
        decoder_input: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        ctc_log_probs, out_lengths, aux_log_probs, dec_log_probs = cast(
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None],
            self.model(feats, feat_lengths, decoder_input=decoder_input),
        )
        return ctc_log_probs, out_lengths, aux_log_probs, dec_log_probs

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
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch", "frequency": 1},
        }

    def on_train_epoch_start(self) -> None:
        epoch = int(self.current_epoch)
        if self.trainer is None:
            return

        train_dl = self.trainer.train_dataloader
        ds = getattr(train_dl, "dataset", None)
        if ds is not None and hasattr(ds, "set_current_epoch"):
            ds.set_current_epoch(epoch)

    def training_step(self, batch: TrainBatch, batch_idx: int) -> torch.Tensor:
        feats = batch["features"]
        feat_lengths = batch["feature_lengths"]
        targets = batch["targets"]
        target_lengths = batch["target_lengths"]

        dec_in, dec_out = self.build_decoder_sequences(targets, target_lengths)

        ctc_log_probs, out_lengths, aux_log_probs, dec_log_probs = self.forward(
            feats,
            feat_lengths,
            decoder_input=dec_in,
        )
        ctc_log_probs_t = ctc_log_probs.transpose(0, 1)

        if (out_lengths < target_lengths).any():
            diff = (out_lengths - target_lengths).min()
            raise RuntimeError(
                f"CTC input length smaller than target length. Min(input_len - target_len) = {diff}"
            )

        autocast_device_type = "cuda" if ctc_log_probs_t.is_cuda else "cpu"

        main_ctc = ctc_loss_with_label_smoothing(
            log_probs_t=ctc_log_probs_t,
            targets=targets,
            input_lengths=out_lengths,
            target_lengths=target_lengths,
            blank_id=self.blank_id,
            lsm_weight=self.cfg.ctc_label_smoothing,
            autocast_device_type=autocast_device_type,
            exclude_blank_from_ls=True,
        )

        if self.cfg.aux_ctc_weight > 0.0 and aux_log_probs.numel() > 0:
            aux_losses: list[torch.Tensor] = []
            for i in range(aux_log_probs.size(0)):
                aux_i = aux_log_probs[i].transpose(0, 1)
                aux_losses.append(
                    ctc_loss_with_label_smoothing(
                        log_probs_t=aux_i,
                        targets=targets,
                        input_lengths=out_lengths,
                        target_lengths=target_lengths,
                        blank_id=self.blank_id,
                        lsm_weight=self.cfg.ctc_label_smoothing,
                        autocast_device_type=autocast_device_type,
                        exclude_blank_from_ls=True,
                    )
                )
            aux_ctc: torch.Tensor = torch.stack(aux_losses).mean()
        else:
            aux_ctc = torch.tensor(0.0, device=main_ctc.device)

        if dec_log_probs is None:
            raise RuntimeError("Expected decoder outputs during training, got None.")

        B, L, V_dec = dec_log_probs.shape
        attn_loss: torch.Tensor = self.attn_loss(
            dec_log_probs.reshape(B * L, V_dec),
            dec_out.reshape(B * L),
        )

        lambda_ctc = float(self.cfg.ctc_weight)
        total_loss: torch.Tensor = (
            lambda_ctc * main_ctc
            + (1.0 - lambda_ctc) * attn_loss
            + float(self.cfg.aux_ctc_weight) * aux_ctc
        )

        self.log(
            "train/loss",
            total_loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            sync_dist=False,
            batch_size=feats.size(0),
        )
        self.log(
            "train/loss_ctc_main",
            main_ctc,
            prog_bar=False,
            on_step=True,
            on_epoch=True,
            sync_dist=False,
            batch_size=feats.size(0),
        )
        self.log(
            "train/loss_ctc_aux",
            aux_ctc,
            prog_bar=False,
            on_step=True,
            on_epoch=True,
            sync_dist=False,
            batch_size=feats.size(0),
        )
        self.log(
            "train/loss_attn",
            attn_loss,
            prog_bar=False,
            on_step=True,
            on_epoch=True,
            sync_dist=False,
            batch_size=feats.size(0),
        )

        if not torch.isfinite(total_loss):
            logger.error(
                f"[NaN] loss={total_loss}, main_ctc={main_ctc}, aux_ctc={aux_ctc}, "
                f"attn={attn_loss}, step={self.global_step}, epoch={self.current_epoch}"
            )
            raise RuntimeError("NaN in loss.")

        return total_loss

    def validation_step(self, batch: ValBatch, batch_idx: int) -> dict[str, float]:
        feats = batch["features"]
        feat_lengths = batch["feature_lengths"]
        targets = batch["targets"]
        target_lengths = batch["target_lengths"]
        texts = batch["text"]
        langs = batch.get("language") or ["unknown"] * len(texts)

        ctc_log_probs, out_lengths, _, _ = self.forward(feats, feat_lengths, decoder_input=None)
        ctc_log_probs_t = ctc_log_probs.transpose(0, 1)

        if (out_lengths < target_lengths).any():
            diff = (out_lengths - target_lengths).min()
            raise RuntimeError(
                "[VAL] CTC input length smaller than target length. "
                f"Min(input_len - target_len) = {diff}"
            )

        loss: torch.Tensor = self.ctc_loss(ctc_log_probs_t, targets, out_lengths, target_lengths)

        decoded_ids = greedy_decoder(ctc_log_probs, out_lengths, blank_id=self.blank_id)

        pred_texts: list[str] = []
        for seq in decoded_ids:
            sp_ids = [i - 1 for i in seq if i > 0]
            pred_texts.append("" if not sp_ids else self.sp.decode_ids(sp_ids))

        batch_wer: float = float(jiwer_wer(texts, pred_texts))
        batch_cer: float = float(jiwer_cer(texts, pred_texts))
        bs = len(texts)

        self.log("val/loss", loss, prog_bar=True, on_epoch=True, sync_dist=False, batch_size=bs)
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

        en_refs: list[str] = []
        en_hyps: list[str] = []
        pl_refs: list[str] = []
        pl_hyps: list[str] = []

        for lang, ref, hyp in zip(langs, texts, pred_texts, strict=True):
            if lang == "en":
                en_refs.append(ref)
                en_hyps.append(hyp)
            elif lang == "pl":
                pl_refs.append(ref)
                pl_hyps.append(hyp)

        if en_refs:
            en_wer: float = float(jiwer_wer(en_refs, en_hyps))
            en_cer: float = float(jiwer_cer(en_refs, en_hyps))
            self.log("val/wer/en", en_wer, prog_bar=False, on_epoch=True, batch_size=len(en_refs))
            self.log("val/cer/en", en_cer, prog_bar=False, on_epoch=True, batch_size=len(en_refs))

        if pl_refs:
            pl_wer: float = float(jiwer_wer(pl_refs, pl_hyps))
            pl_cer: float = float(jiwer_cer(pl_refs, pl_hyps))
            self.log("val/wer/pl", pl_wer, prog_bar=False, on_epoch=True, batch_size=len(pl_refs))
            self.log("val/cer/pl", pl_cer, prog_bar=False, on_epoch=True, batch_size=len(pl_refs))

        for lang, ref, hyp in zip(langs, texts, pred_texts, strict=True):
            if (
                lang in self.example_buffer
                and len(self.example_buffer[lang]) < self.examples_per_lang
            ):
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
            if wer is not None and cer is not None:
                logger.info(f"[VAL][epoch={epoch}] WER={wer:.4f} CER={cer:.4f}")
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

    model = LitFastConformerCTCAttention(
        cfg,
        vocab_size=vocab_size,
        sp_vocab_size=sp_vocab_size,
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
    logger.info(f"Vocab size (CTC): {vocab_size}, SP vocab (decoder): {sp_vocab_size}")
    logger.info(f"Precision: {cfg.precision}")
    logger.info(f"CTC weight lambda={cfg.ctc_weight}")

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
                "ctc_weight": cfg.ctc_weight,
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
        ckpt_path=cfg.ckpt_path,
    )


if __name__ == "__main__":
    cfg = tyro.cli(TrainConfig)
    main(cfg)

from __future__ import annotations

from typing import cast

import lightning.pytorch as pl
import torch
import torch.nn as nn
from lightning.pytorch.utilities.types import OptimizerLRSchedulerConfig
from loguru import logger
from sentencepiece import SentencePieceProcessor

from SpeechToText.models.common import (
    ExamplesBuffer,
    ctc_ids_to_texts_spm,
    greedy_ctc_decode,
    wer_cer_by_lang,
)
from SpeechToText.models.common.optimizers import configure_adamw_noam
from SpeechToText.models.ctc.model import FastConformerCTC
from SpeechToText.models.ctc.steps import compute_ctc_losses
from SpeechToText.models.ctc.train import TrainConfig
from SpeechToText.models.typing import TrainBatch, ValBatch


class LitFastConformerCTC(pl.LightningModule):
    def __init__(
        self,
        config: TrainConfig,
        vocab_size: int,
        sp: SentencePieceProcessor,
        blank_id: int = 0,
    ) -> None:
        super().__init__()
        self.config = config
        self.sp = sp
        self.blank_id = int(blank_id)

        self.net = FastConformerCTC(config.model, vocab_size=vocab_size, blank_id=self.blank_id)
        self.ctc_loss = nn.CTCLoss(blank=self.blank_id, zero_infinity=True, reduction="mean")

        self.examples = ExamplesBuffer(per_lang=2)
        self.save_hyperparameters(ignore=["sp"])

    def forward(self, feats: torch.Tensor, feat_lengths: torch.Tensor) -> torch.Tensor:
        out = self.net(feats, feat_lengths)
        return cast(torch.Tensor, out.log_probs)

    def configure_optimizers(self) -> OptimizerLRSchedulerConfig:
        optimizer_config = self.config.optimizer
        d_model = self.config.model.encoder.d_model
        warmup_steps = optimizer_config.warmup_steps

        return configure_adamw_noam(
            self,
            learning_rate=optimizer_config.learning_rate,
            betas=optimizer_config.betas,
            epsilon=optimizer_config.epsilon,
            weight_decay=optimizer_config.weight_decay,
            warmup_steps=warmup_steps,
            d_model=int(d_model),
        )

    def training_step(self, batch: TrainBatch, batch_idx: int) -> torch.Tensor:
        feats = batch["features"]
        feat_lens = batch["feature_lengths"]
        targets = batch["targets"]
        target_lens = batch["target_lengths"]

        out = self.net(feats, feat_lens)
        if (out.out_lengths < target_lens).any():
            raise RuntimeError("CTC input length < target length")

        losses = compute_ctc_losses(
            log_probs=out.log_probs,
            out_lengths=out.out_lengths,
            aux_log_probs=out.aux_log_probs,
            targets=targets,
            target_lengths=target_lens,
            blank_id=self.blank_id,
            lsm_weight=self.config.ctc_label_smoothing,
            aux_weight=self.config.aux_ctc_weight,
        )

        self.log_dict(
            {
                "train/loss": losses.total,
                "train/loss_main": losses.main,
                "train/loss_aux": losses.aux,
            },
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            batch_size=feats.size(0),
        )
        if not torch.isfinite(losses.total):
            logger.error(f"NaN loss at step={self.global_step}")
            raise RuntimeError("NaN/Inf in loss")
        return losses.total

    def validation_step(self, batch: ValBatch, batch_idx: int) -> dict[str, float]:
        feats = batch["features"]
        feat_lens = batch["feature_lengths"]
        targets = batch["targets"]
        target_lens = batch["target_lengths"]
        texts = batch["text"]
        langs = batch.get("language")

        out = self.net(feats, feat_lens)
        lp_t = out.log_probs.transpose(0, 1)
        loss = self.ctc_loss(lp_t, targets, out.out_lengths, target_lens)

        decoded = greedy_ctc_decode(out.log_probs, out.out_lengths, blank_id=self.blank_id)
        preds = ctc_ids_to_texts_spm(self.sp, decoded)
        m = wer_cer_by_lang(texts, preds, langs)

        self.log("val/loss", loss, prog_bar=True, on_epoch=True, batch_size=len(texts))
        self.log_dict(
            {f"val/{k}": v for k, v in m.items()},
            prog_bar=True,
            on_epoch=True,
            batch_size=len(texts),
        )
        for lang, ref, hyp in zip((langs or ["unknown"] * len(texts)), texts, preds, strict=True):
            self.examples.add(lang, ref, hyp)

        return {"wer": m["wer/overall"], "cer": m["cer/overall"]}

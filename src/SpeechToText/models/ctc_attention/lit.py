from __future__ import annotations

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
from SpeechToText.models.ctc_attention.model import FastConformerCTCAttention
from SpeechToText.models.ctc_attention.steps import compute_ctc_attn_losses
from SpeechToText.models.ctc_attention.train import TrainConfig
from SpeechToText.models.typing import CTCAttnOutput, TrainBatch, ValBatch


class LitFastConformerCTCAttention(pl.LightningModule):
    def __init__(
        self,
        config: TrainConfig,
        *,
        ctc_vocab_size: int,
        sp_vocab_size: int,
        sp: SentencePieceProcessor,
        blank_id: int = 0,
    ) -> None:
        super().__init__()
        self.config = config
        self.sp = sp

        self.blank_id = int(blank_id)
        self.sp_vocab_size = int(sp_vocab_size)

        self.pad_id = self.sp_vocab_size
        self.bos_id = self.sp_vocab_size + 1
        self.eos_id = self.sp_vocab_size + 2

        self.net: FastConformerCTCAttention = FastConformerCTCAttention(
            config.model,
            ctc_vocab_size=int(ctc_vocab_size),
            sp_vocab_size=self.sp_vocab_size,
            blank_id=self.blank_id,
        )

        self.ctc_loss = nn.CTCLoss(blank=self.blank_id, zero_infinity=True, reduction="mean")
        self.attn_loss = nn.NLLLoss(ignore_index=self.pad_id, reduction="mean")

        self.examples = ExamplesBuffer(per_lang=2)
        self.save_hyperparameters(ignore=["sp"])

    def build_decoder_sequences(
        self,
        targets_concat: torch.Tensor,
        target_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        targets_concat: concat 1D of CTC targets (blank=0, sp_id+1)
        decoder vocab: sp_id in [0..sp_vocab-1] plus pad/bos/eos
        """
        device = targets_concat.device
        b = int(target_lengths.shape[0])
        max_len = int(target_lengths.max().item()) if b > 0 else 0

        dec_in = torch.full((b, max_len + 1), self.pad_id, dtype=torch.long, device=device)
        dec_out = torch.full((b, max_len + 1), self.pad_id, dtype=torch.long, device=device)

        off = 0
        for i in range(b):
            u = int(target_lengths[i].item())
            if u == 0:
                dec_in[i, 0] = self.bos_id
                dec_out[i, 0] = self.eos_id
                continue

            seq = targets_concat[off : off + u]
            off += u

            piece_seq = seq - 1  # CTC target (sp+1) -> sp_id

            dec_in[i, 0] = self.bos_id
            dec_in[i, 1 : u + 1] = piece_seq

            dec_out[i, 0:u] = piece_seq
            dec_out[i, u] = self.eos_id

        return dec_in, dec_out

    def forward(
        self,
        feats: torch.Tensor,
        feat_lengths: torch.Tensor,
        decoder_input: torch.Tensor | None = None,
    ) -> CTCAttnOutput:
        output: CTCAttnOutput = self.net(feats, feat_lengths, decoder_input=decoder_input)
        return output

    def configure_optimizers(self) -> OptimizerLRSchedulerConfig:
        opt_cfg = self.config.optimizer
        d_model = int(self.config.model.encoder.d_model)

        return configure_adamw_noam(
            self,
            learning_rate=opt_cfg.learning_rate,
            betas=opt_cfg.betas,
            epsilon=opt_cfg.epsilon,
            weight_decay=opt_cfg.weight_decay,
            warmup_steps=int(opt_cfg.warmup_steps),
            d_model=d_model,
        )

    def training_step(self, batch: TrainBatch, batch_idx: int) -> torch.Tensor:
        feats = batch["features"]
        feat_lengths = batch["feature_lengths"]
        targets = batch["targets"]
        target_lengths = batch["target_lengths"]

        dec_in, dec_out = self.build_decoder_sequences(targets, target_lengths)
        out = self.net(feats, feat_lengths, decoder_input=dec_in)

        if (out.out_lengths < target_lengths).any():
            diff = (out.out_lengths - target_lengths).min().item()
            raise RuntimeError(
                f"CTC input length smaller than target length. Min(input_len-target_len)={diff}"
            )

        if out.dec_log_probs is None:
            raise RuntimeError("Expected decoder outputs during training, got None.")

        autocast_device_type = "cuda" if out.ctc_log_probs.is_cuda else "cpu"

        losses = compute_ctc_attn_losses(
            ctc_log_probs=out.ctc_log_probs,
            out_lengths=out.out_lengths,
            aux_log_probs=out.aux_log_probs,
            targets=targets,
            target_lengths=target_lengths,
            dec_log_probs=out.dec_log_probs,
            dec_out=dec_out,
            blank_id=self.blank_id,
            ctc_label_smoothing=self.config.ctc_label_smoothing,
            aux_ctc_weight=self.config.aux_ctc_weight,
            ctc_weight=self.config.ctc_weight,
            autocast_device_type=autocast_device_type,
            attn_loss_fn=self.attn_loss,
        )

        self.log_dict(
            {
                "train/loss": losses.total,
                "train/loss_ctc_main": losses.ctc_main,
                "train/loss_ctc_aux": losses.ctc_aux,
                "train/loss_attn": losses.attn,
            },
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            sync_dist=False,
            batch_size=feats.size(0),
        )

        if not torch.isfinite(losses.total):
            logger.error(
                f"[NaN] loss={losses.total} step={self.global_step} epoch={self.current_epoch}"
            )
            raise RuntimeError("NaN/Inf in loss.")
        return losses.total

    def validation_step(self, batch: ValBatch, batch_idx: int) -> dict[str, float]:
        feats = batch["features"]
        feat_lengths = batch["feature_lengths"]
        targets = batch["targets"]
        target_lengths = batch["target_lengths"]
        texts = batch["text"]
        langs = batch.get("language")

        out = self.net(feats, feat_lengths, decoder_input=None)

        if (out.out_lengths < target_lengths).any():
            diff = (out.out_lengths - target_lengths).min().item()
            raise RuntimeError(
                f"[VAL] CTC input length smaller than target length. Min(input_len-target_len)={diff}"
            )

        # nn.CTCLoss wants (T, N, C) so we transpose(0, 1)
        loss = self.ctc_loss(
            out.ctc_log_probs.transpose(0, 1),
            targets,
            out.out_lengths,
            target_lengths,
        )

        decoded = greedy_ctc_decode(out.ctc_log_probs, out.out_lengths, blank_id=self.blank_id)
        pred_texts = ctc_ids_to_texts_spm(self.sp, decoded)
        m = wer_cer_by_lang(texts, pred_texts, langs)

        bs = len(texts)
        self.log("val/loss", loss, prog_bar=True, on_epoch=True, sync_dist=False, batch_size=bs)
        self.log_dict(
            {f"val/{k}": v for k, v in m.items()},
            prog_bar=True,
            on_epoch=True,
            sync_dist=False,
            batch_size=bs,
        )

        for lang, ref, hyp in zip((langs or ["unknown"] * bs), texts, pred_texts, strict=True):
            self.examples.add(lang, ref, hyp)

        return {"wer": m["wer/overall"], "cer": m["cer/overall"]}

    def on_validation_epoch_end(self) -> None:
        for lang, pairs in self.examples.pop_all().items():
            if not pairs:
                continue
            logger.info(f"[VAL][{lang}] --- examples epoch {self.current_epoch} ---")
            for ref, hyp in pairs:
                logger.info(f"[VAL][{lang}] REF: {ref}")
                logger.info(f"[VAL][{lang}] HYP: {hyp}")

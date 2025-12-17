from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import lightning.pytorch as pl
import torch
from lightning.pytorch.utilities.types import OptimizerLRSchedulerConfig
from loguru import logger
from sentencepiece import SentencePieceProcessor

from SpeechToText.models.common import ExamplesBuffer, wer_cer_by_lang
from SpeechToText.models.common.optimizers import configure_adamw_noam
from SpeechToText.models.common.rnnt import greedy_rnnt_path_decode_one, rnnt_loss_mean
from SpeechToText.models.tdt.model import FastConformerTDT
from SpeechToText.models.typing import TDTOutput, TrainBatch, ValBatch

if TYPE_CHECKING:
    from SpeechToText.models.tdt.train import TrainConfig


class LitFastConformerTDT(pl.LightningModule):
    def __init__(
        self,
        config: TrainConfig,
        *,
        sp: SentencePieceProcessor,
        vocab_size: int,
    ) -> None:
        super().__init__()
        self.config = config
        self.sp = sp

        model_config = cast(Any, config.model)

        # Konwencja w repo: blank=0, token_id = sp_id+1 => vocab = sp_vocab + 1
        self.blank_id = int(config.blank_id)
        model_config.blank_id = int(self.blank_id)

        model_config.decoder.vocab_size = int(vocab_size)
        model_config.joint.vocab_size = int(vocab_size)

        self.net = FastConformerTDT(model_config)

        self.examples = ExamplesBuffer(per_lang=2)
        self.save_hyperparameters(ignore=["sp"])

    def forward(
        self,
        feats: torch.Tensor,
        feat_lengths: torch.Tensor,
        targets: torch.Tensor,
        target_lengths: torch.Tensor,
    ) -> TDTOutput:
        out = self.net(
            feats=feats,
            feat_lengths=feat_lengths,
            targets_concat=targets,
            target_lengths=target_lengths,
        )
        return cast(TDTOutput, out)

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

    def _rnnt_loss(
        self,
        logits_or_log_probs: torch.Tensor,
        out_lengths: torch.Tensor,
        targets_1d_or_2d: torch.Tensor,
        target_lengths: torch.Tensor,
    ) -> torch.Tensor:
        return rnnt_loss_mean(
            logits=logits_or_log_probs,
            out_lengths=out_lengths,
            targets_1d_or_2d=targets_1d_or_2d,
            target_lengths=target_lengths,
            blank_id=self.blank_id,
            clamp=float(self.config.rnnt_clamp),
            fused_log_softmax=bool(self.config.fused_log_softmax),
        )

    def training_step(self, batch: TrainBatch, batch_idx: int) -> torch.Tensor:
        feats = batch["features"]
        feat_lens = batch["feature_lengths"]
        targets = batch["targets"]
        target_lens = batch["target_lengths"]

        out = self.forward(feats, feat_lens, targets=targets, target_lengths=target_lens)
        loss = self._rnnt_loss(out.log_probs, out.out_lengths, targets, target_lens)

        self.log(
            "train/loss",
            loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            sync_dist=False,
            batch_size=feats.size(0),
        )
        return loss

    def validation_step(self, batch: ValBatch, batch_idx: int) -> dict[str, float]:
        feats = batch["features"]
        feat_lens = batch["feature_lengths"]
        targets = batch["targets"]
        target_lens = batch["target_lengths"]
        texts = batch["text"]
        langs = batch.get("language")

        out = self.forward(feats, feat_lens, targets=targets, target_lengths=target_lens)
        loss = self._rnnt_loss(out.log_probs, out.out_lengths, targets, target_lens)

        bs = feats.size(0)
        self.log("val/loss", loss, prog_bar=True, on_epoch=True, sync_dist=False, batch_size=bs)

        pred_texts: list[str] = []
        for b in range(bs):
            ids = greedy_rnnt_path_decode_one(
                out.log_probs[b : b + 1],
                out_length=int(out.out_lengths[b].item()),
                max_symbols_per_t=int(self.config.val_max_symbols_per_t),
                blank_id=self.blank_id,
            )
            sp_ids = [i - 1 for i in ids if i != self.blank_id and i > 0]
            pred_texts.append("" if not sp_ids else self.sp.decode_ids(sp_ids))

        m = wer_cer_by_lang(texts, pred_texts, langs)
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

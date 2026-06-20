from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import lightning.pytorch as pl
import torch
from lightning.pytorch.utilities.types import OptimizerLRSchedulerConfig
from loguru import logger
from sentencepiece import SentencePieceProcessor

from SpeechToText.augmentation import SpecAugment
from SpeechToText.features import WaveformFeaturizer
from SpeechToText.models.common import ExamplesBuffer, wer_cer_by_lang
from SpeechToText.models.common.optimizers import configure_adamw_noam
from SpeechToText.models.common.rnnt import greedy_rnnt_path_decode_one, rnnt_loss_mean
from SpeechToText.models.common.validation_logging import (
    WorstValExamplesCollector,
    log_wandb_worst_val_examples,
)
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

        self.blank_id = int(config.blank_id)
        model_config.blank_id = int(self.blank_id)
        model_config.decoder.vocab_size = int(vocab_size)
        model_config.joint.vocab_size = int(vocab_size)
        model_config.decoder.d_model = int(model_config.encoder.d_model)
        model_config.joint.enc_d = int(model_config.encoder.d_model)
        model_config.joint.pred_d = int(model_config.encoder.d_model)

        feat_cfg = config.data.features
        self.sample_rate = feat_cfg.sample_rate

        spec_augment = SpecAugment(
            config.spec_augment, augment_start_epoch=config.spec_augment_start_epoch
        )
        self.featurizer = WaveformFeaturizer(
            config.data.features,
            spec_augment=spec_augment,
        )

        self.net = FastConformerTDT(model_config)
        self.examples = ExamplesBuffer(per_lang=2)
        self._val_examples = WorstValExamplesCollector(max_examples=50)
        self.save_hyperparameters(ignore=["sp"])

    def on_fit_start(self) -> None:
        self.featurizer = self.featurizer.to(self.device)

    def _encode_batch(
        self, audio: torch.Tensor, audio_lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.featurizer.set_current_epoch(self.current_epoch)
        return cast(
            tuple[torch.Tensor, torch.Tensor], self.featurizer(audio.to(self.device), audio_lengths)
        )

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
        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = int(total_steps * opt_cfg.warmup_ratio)

        return configure_adamw_noam(
            self,
            lr=opt_cfg.lr,
            betas=opt_cfg.betas,
            weight_decay=opt_cfg.weight_decay,
            warmup_steps=max(1, warmup_steps),
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
        audio = batch["audio"]
        audio_lengths = batch["audio_length"]
        targets = batch["targets"]
        target_lengths = batch["target_length"]

        feats, feat_lens = self._encode_batch(audio, audio_lengths)
        out = self.forward(feats, feat_lens, targets=targets, target_lengths=target_lengths)
        loss = self._rnnt_loss(out.log_probs, out.out_lengths, targets, target_lengths)

        self.log(
            "train/loss",
            loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            sync_dist=False,
            batch_size=audio.size(0),
        )
        return loss

    def validation_step(self, batch: ValBatch, batch_idx: int) -> dict[str, float]:
        audio = batch["audio"]
        audio_lengths = batch["audio_length"]
        targets = batch["targets"]
        target_lengths = batch["target_length"]
        texts = batch["text"]
        langs = batch.get("language")

        feats, feat_lens = self._encode_batch(audio, audio_lengths)
        out = self.forward(feats, feat_lens, targets=targets, target_lengths=target_lengths)
        loss = self._rnnt_loss(out.log_probs, out.out_lengths, targets, target_lengths)

        bs = audio.size(0)
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

        datasets = batch.get("dataset", ["unknown"] * bs)
        for index in range(bs):
            self._val_examples.add(
                dataset=datasets[index] if index < len(datasets) else "unknown",
                language=(langs or ["unknown"] * bs)[index],
                reference=texts[index],
                hypothesis=pred_texts[index],
                audio=audio[index, : int(audio_lengths[index].item())],
            )

        return {"wer": m["wer/overall"], "cer": m["cer/overall"]}

    def on_validation_epoch_start(self) -> None:
        self._val_examples.reset()

    def on_validation_epoch_end(self) -> None:
        log_wandb_worst_val_examples(
            self.logger,
            self._val_examples.worst_first(),
            sample_rate=int(self.sample_rate),
            epoch=int(self.current_epoch),
        )

        for lang, pairs in self.examples.pop_all().items():
            if not pairs:
                continue
            logger.info(f"[VAL][{lang}] --- examples epoch {self.current_epoch} ---")
            for ref, hyp in pairs:
                logger.info(f"[VAL][{lang}] REF: {ref}")
                logger.info(f"[VAL][{lang}] HYP: {hyp}")

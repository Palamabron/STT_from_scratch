from __future__ import annotations

from typing import Any, Final, cast

import lightning.pytorch as pl
import torch
import torch.nn as nn
from lightning.pytorch.utilities.types import OptimizerLRSchedulerConfig
from loguru import logger
from sentencepiece import SentencePieceProcessor

from SpeechToText.augmentation import GPUAudioAugmentation, SpecAugment
from SpeechToText.features import WaveformFeaturizer
from SpeechToText.models.common import (
    ExamplesBuffer,
    ctc_ids_to_texts_spm,
    greedy_ctc_decode,
)
from SpeechToText.models.common.batch_filter import (
    filter_batch_by_encoder_length,
    warn_empty_training_batch,
)
from SpeechToText.models.common.optimizer_factory import configure_adamw_scheduler
from SpeechToText.models.common.validation_logging import (
    WorstValExamplesCollector,
    log_wandb_worst_val_examples,
)
from SpeechToText.models.ctc.model import FastConformerCTC
from SpeechToText.models.ctc.steps import compute_ctc_losses
from SpeechToText.models.typing import ValBatch


class LitFastConformerCTC(pl.LightningModule):
    """Lightning module for Fast-Conformer CTC training and validation."""

    def __init__(
        self,
        config: Any,
        sp: SentencePieceProcessor,
        rir_bank: tuple[torch.Tensor, ...] | None = None,
        noise_bank: tuple[torch.Tensor, ...] | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.sp: Final[SentencePieceProcessor] = sp

        self.blank_id: Final[int] = 0
        self.pad_id: Final[int] = self.blank_id

        self.ctc_label_smoothing: Final[float] = float(config.ctc_label_smoothing)
        self.aux_ctc_weight: Final[float] = float(config.aux_ctc_weight)

        gpu_augment = GPUAudioAugmentation(
            config.audio_augment,
            rir_bank,
            noise_bank,
            augment_start_epoch=config.audio_augment_start_epoch,
        )
        spec_augment = SpecAugment(
            config.spec_augment, augment_start_epoch=config.spec_augment_start_epoch
        )
        self.featurizer = WaveformFeaturizer(
            config.data.features,
            spec_augment=spec_augment,
            gpu_augment=gpu_augment,
        )

        vocab_size: int = int(sp.get_piece_size())
        self.net: Final[FastConformerCTC] = FastConformerCTC(
            config.model, vocab_size=vocab_size + 1, blank_id=self.blank_id
        )
        self.ctc_loss: Final[nn.CTCLoss] = nn.CTCLoss(
            blank=self.blank_id, zero_infinity=True, reduction="mean"
        )
        self.examples: Final[ExamplesBuffer] = ExamplesBuffer(per_lang=2)
        self._val_examples = WorstValExamplesCollector(max_examples=50)

        self.save_hyperparameters(ignore=["sp", "rir_bank", "noise_bank"])

        self._val_texts_pred: list[str] = []
        self._val_texts_ref: list[str] = []
        self._val_langs: list[str] = []

    def on_fit_start(self) -> None:
        self.featurizer = self.featurizer.to(self.device)

    def _encode_batch(
        self,
        audio: torch.Tensor,
        audio_lengths: torch.Tensor,
        clean_pass: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.featurizer.set_current_epoch(self.current_epoch)
        return cast(
            tuple[torch.Tensor, torch.Tensor],
            self.featurizer(audio, audio_lengths, clean_pass=clean_pass),
        )

    def forward(self, feats: torch.Tensor, feat_lengths: torch.Tensor) -> torch.Tensor:
        out = self.net(feats, feat_lengths)
        return cast(torch.Tensor, out.log_probs)

    def configure_optimizers(self) -> OptimizerLRSchedulerConfig:
        opt_cfg = self.config.optimizer
        total_steps = int(self.trainer.estimated_stepping_batches)

        return configure_adamw_scheduler(
            self,
            optimizer_cfg=opt_cfg,
            total_steps=total_steps,
        )

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor | None:
        audio = batch["audio"]
        audio_lengths = batch["audio_length"]
        targets = batch["targets"]
        target_lengths = batch["target_length"]
        clean_pass = batch.get("clean_pass")
        if clean_pass is not None:
            clean_pass = clean_pass.to(self.device)

        feats, feat_lens = self._encode_batch(audio, audio_lengths, clean_pass=clean_pass)
        out = self.net(feats, feat_lens)

        log_probs = cast(torch.Tensor, out.log_probs)
        aux_log_probs = cast(torch.Tensor, out.aux_log_probs)
        if aux_log_probs is None:
            aux_log_probs = torch.empty(0, device=self.device)

        time_steps = int(log_probs.size(1))
        ctc_in_lens = out.out_lengths.clamp(max=time_steps)

        filtered = filter_batch_by_encoder_length(batch, ctc_in_lens, target_lengths)
        if filtered is None:
            warn_empty_training_batch(batch_idx, audio.size(0))
            return None
        if filtered[0] is not batch:
            batch = filtered[0]
            audio = batch["audio"]
            audio_lengths = batch["audio_length"]
            targets = batch["targets"]
            target_lengths = batch["target_length"]
            clean_pass = batch.get("clean_pass")
            if clean_pass is not None:
                clean_pass = clean_pass.to(self.device)
            feats, feat_lens = self._encode_batch(audio, audio_lengths, clean_pass=clean_pass)
            out = self.net(feats, feat_lens)
            log_probs = cast(torch.Tensor, out.log_probs)
            aux_log_probs = cast(torch.Tensor, out.aux_log_probs)
            if aux_log_probs is None:
                aux_log_probs = torch.empty(0, device=self.device)
            ctc_in_lens = out.out_lengths.clamp(max=int(log_probs.size(1)))
        else:
            _, ctc_in_lens, target_lengths, _ = filtered

        losses = compute_ctc_losses(
            log_probs=log_probs,
            out_lengths=ctc_in_lens,
            aux_log_probs=aux_log_probs,
            targets=targets,
            target_lengths=target_lengths,
            blank_id=self.blank_id,
            lsm_weight=self.ctc_label_smoothing,
            aux_weight=self.aux_ctc_weight,
        )

        self.log(
            "train/loss",
            losses.total,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            batch_size=audio.size(0),
        )
        if clean_pass is not None:
            self.log(
                "train/aug/clean_pass_frac",
                clean_pass.float().mean(),
                on_step=True,
                on_epoch=True,
                batch_size=audio.size(0),
            )
        return losses.total

    def validation_step(self, batch: ValBatch, batch_idx: int) -> None:
        audio = batch["audio"]
        audio_lengths = batch["audio_length"]
        targets = batch["targets"]
        target_lengths = batch["target_length"]

        feats, feat_lens = self._encode_batch(audio, audio_lengths)
        out = self.net(feats, feat_lens)

        log_probs_btv = cast(torch.Tensor, out.log_probs).float()
        time_steps = int(log_probs_btv.size(1))
        if time_steps == 0:
            return

        ctc_in_lens = out.out_lengths.clamp(max=time_steps)
        loss = self.ctc_loss(log_probs_btv.transpose(0, 1), targets, ctc_in_lens, target_lengths)
        self.log("val/loss", loss, prog_bar=True, on_epoch=True, batch_size=audio.size(0))

        greedy_preds = greedy_ctc_decode(
            log_probs_btv.detach(), ctc_in_lens, blank_id=self.blank_id
        )
        self._val_examples.accumulate_blank_stats(log_probs_btv, ctc_in_lens, self.blank_id)
        texts_pred = ctc_ids_to_texts_spm(self.sp, greedy_preds)
        texts_ref = batch["text"]
        langs = batch.get("language", ["unknown"] * len(texts_pred))
        datasets = batch.get("dataset", ["unknown"] * len(texts_pred))

        self._val_texts_pred.extend(texts_pred)
        self._val_texts_ref.extend(texts_ref)
        self._val_langs.extend(langs)

        for index, (ref, hyp, lang) in enumerate(zip(texts_ref, texts_pred, langs, strict=True)):
            self.examples.add(lang, ref, hyp)
            self._val_examples.add(
                dataset=datasets[index] if index < len(datasets) else "unknown",
                language=lang,
                reference=ref,
                hypothesis=hyp,
                audio=audio[index, : int(audio_lengths[index].item())],
            )

    def on_validation_epoch_start(self) -> None:
        self._val_texts_pred.clear()
        self._val_texts_ref.clear()
        self._val_langs.clear()
        self._val_examples.reset()

    def on_validation_epoch_end(self) -> None:
        from SpeechToText.models.common import wer_cer_by_lang

        log_wandb_worst_val_examples(
            self.logger,
            self._val_examples.worst_first(),
            sample_rate=int(self.config.data.features.sample_rate),
            epoch=int(self.current_epoch),
            step=int(self.trainer.global_step),
        )

        if not self._val_texts_ref:
            self.log("val/wer/overall", 1.0, prog_bar=True, on_epoch=True)
        else:
            metrics = wer_cer_by_lang(self._val_texts_ref, self._val_texts_pred, self._val_langs)
            for name, value in metrics.items():
                self.log(f"val/{name}", value, prog_bar=True, on_epoch=True)

        self.log(
            "val/blank_fraction",
            self._val_examples.blank_fraction(),
            prog_bar=True,
            on_epoch=True,
        )

        examples = self.examples.pop_all()
        for lang, pairs in examples.items():
            if pairs:
                ref, hyp = pairs[0]
                logger.info(f"[VAL][{lang}] REF: {ref}")
                logger.info(f"[VAL][{lang}] HYP: {hyp}")

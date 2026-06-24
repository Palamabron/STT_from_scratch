from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import lightning.pytorch as pl
import torch
from lightning.pytorch.utilities.types import OptimizerLRSchedulerConfig
from loguru import logger
from sentencepiece import SentencePieceProcessor

from SpeechToText.augmentation import GPUAudioAugmentation, SpecAugment
from SpeechToText.features import WaveformFeaturizer
from SpeechToText.models.common import ExamplesBuffer, wer_cer_by_lang
from SpeechToText.models.common.batch_filter import filter_batch_by_encoder_length
from SpeechToText.models.common.optimizer_factory import configure_adamw_scheduler
from SpeechToText.models.common.rnnt import transducer_greedy_decode_one
from SpeechToText.models.common.validation_logging import (
    WorstValExamplesCollector,
    log_wandb_worst_val_examples,
)
from SpeechToText.models.tdt.loss import compute_tdt_losses
from SpeechToText.models.tdt.model import FastConformerTDT
from SpeechToText.models.typing import TDTOutput, TrainBatch, ValBatch

if TYPE_CHECKING:
    from SpeechToText.models.tdt.config import TrainConfig


class LitFastConformerTDT(pl.LightningModule):
    """Lightning module for Fast-Conformer RNN-T / TDT training."""

    def __init__(
        self,
        config: TrainConfig,
        *,
        sp: SentencePieceProcessor,
        vocab_size: int,
        rir_bank: tuple[torch.Tensor, ...] | None = None,
        noise_bank: tuple[torch.Tensor, ...] | None = None,
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
        model_config.joint.use_tdt = bool(config.use_tdt)
        model_config.joint_fused_batch_size = config.joint_fused_batch_size

        feat_cfg = config.data.features
        self.sample_rate = feat_cfg.sample_rate

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

        self.net = FastConformerTDT(model_config)
        self.examples = ExamplesBuffer(per_lang=2)
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
            self.featurizer(
                audio.to(self.device),
                audio_lengths,
                clean_pass=clean_pass.to(self.device) if clean_pass is not None else None,
            ),
        )

    def forward(
        self,
        feats: torch.Tensor,
        feat_lengths: torch.Tensor,
        targets: torch.Tensor,
        target_lengths: torch.Tensor,
    ) -> TDTOutput:
        return cast(
            TDTOutput,
            self.net(
                feats=feats,
                feat_lengths=feat_lengths,
                targets_concat=targets,
                target_lengths=target_lengths,
            ),
        )

    def configure_optimizers(self) -> OptimizerLRSchedulerConfig:
        opt_cfg = self.config.optimizer
        total_steps = int(self.trainer.estimated_stepping_batches)

        return configure_adamw_scheduler(
            self,
            optimizer_cfg=opt_cfg,
            total_steps=total_steps,
        )

    def _transducer_loss(
        self,
        out: TDTOutput,
        targets: torch.Tensor,
        target_lengths: torch.Tensor,
    ) -> torch.Tensor:
        assert out.token_logits is not None

        targets_padded = FastConformerTDT.pad_targets_from_concat(
            targets, target_lengths, pad_id=self.blank_id
        )
        losses = compute_tdt_losses(
            token_logits=out.token_logits,
            duration_logits=out.duration_logits,
            out_lengths=out.out_lengths,
            targets_padded=targets_padded,
            target_lengths=target_lengths,
            blank_id=self.blank_id,
            label_smoothing=float(self.config.label_smoothing),
            rnnt_clamp=float(self.config.rnnt_clamp),
            fused_log_softmax=bool(self.config.fused_log_softmax),
            use_tdt=bool(self.config.use_tdt),
            tdt_sigma=float(self.config.tdt_sigma),
            tdt_omega=float(self.config.tdt_omega),
        )
        return losses.total

    def _maybe_filter_batch(
        self,
        batch: TrainBatch | ValBatch,
        out_lengths: torch.Tensor,
        target_lengths: torch.Tensor,
    ) -> tuple[TrainBatch | ValBatch, torch.Tensor, torch.Tensor] | None:
        if (out_lengths >= target_lengths).all():
            return batch, out_lengths, target_lengths
        filtered = filter_batch_by_encoder_length(batch, out_lengths, target_lengths)
        if filtered is None:
            return None
        batch_f, out_lengths_f, target_lengths_f, _ = filtered
        return batch_f, out_lengths_f, target_lengths_f

    def training_step(self, batch: TrainBatch, batch_idx: int) -> torch.Tensor | None:
        audio = batch["audio"]
        audio_lengths = batch["audio_length"]
        targets = batch["targets"]
        target_lengths = batch["target_length"]
        clean_pass = batch.get("clean_pass")
        if clean_pass is not None:
            clean_pass = clean_pass.to(self.device)

        feats, feat_lens = self._encode_batch(audio, audio_lengths, clean_pass=clean_pass)
        out = self.forward(feats, feat_lens, targets=targets, target_lengths=target_lengths)

        filtered = self._maybe_filter_batch(batch, out.out_lengths, target_lengths)
        if filtered is None:
            return None
        batch_f, _, _ = filtered
        if batch_f is not batch:
            batch = batch_f
            audio = batch["audio"]
            audio_lengths = batch["audio_length"]
            targets = batch["targets"]
            target_lengths = batch["target_length"]
            clean_pass = batch.get("clean_pass")
            if clean_pass is not None:
                clean_pass = clean_pass.to(self.device)
            feats, feat_lens = self._encode_batch(audio, audio_lengths, clean_pass=clean_pass)
            out = self.forward(feats, feat_lens, targets=targets, target_lengths=target_lengths)

        loss = self._transducer_loss(out, targets, target_lengths)
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

    def _decode_one_from_encoder(self, enc: torch.Tensor, out_len: int) -> list[int]:
        return transducer_greedy_decode_one(
            enc.unsqueeze(0) if enc.dim() == 2 else enc,
            out_len,
            decoder=self.net.decoder,
            joint=self.net.joint,
            blank_id=self.blank_id,
            max_symbols_per_t=int(self.config.val_max_symbols_per_t),
        )

    def validation_step(self, batch: ValBatch, batch_idx: int) -> None:
        audio = batch["audio"]
        audio_lengths = batch["audio_length"]
        targets = batch["targets"]
        target_lengths = batch["target_length"]
        texts = batch["text"]
        langs = batch.get("language", ["unknown"] * len(texts))

        feats, feat_lens = self._encode_batch(audio, audio_lengths)
        enc, out_lengths = self.net.encoder(feats, feat_lens)

        if self.config.compute_eval_loss:
            out = self.forward(feats, feat_lens, targets=targets, target_lengths=target_lengths)
            loss = self._transducer_loss(out, targets, target_lengths)
            self.log(
                "val/loss",
                loss,
                prog_bar=True,
                on_epoch=True,
                sync_dist=False,
                batch_size=audio.size(0),
            )

        bs = audio.size(0)
        pred_texts: list[str] = []
        for index in range(bs):
            out_len = int(out_lengths[index].item())
            ids = self._decode_one_from_encoder(enc[index], out_len)
            sp_ids = [
                token_id - 1 for token_id in ids if token_id != self.blank_id and token_id > 0
            ]
            pred_texts.append("" if not sp_ids else self.sp.decode_ids(sp_ids))

        self._val_texts_pred.extend(pred_texts)
        self._val_texts_ref.extend(texts)
        self._val_langs.extend(langs)

        for lang, ref, hyp in zip(langs, texts, pred_texts, strict=True):
            self.examples.add(lang, ref, hyp)

        datasets = batch.get("dataset", ["unknown"] * bs)
        for index in range(bs):
            self._val_examples.add(
                dataset=datasets[index] if index < len(datasets) else "unknown",
                language=langs[index],
                reference=texts[index],
                hypothesis=pred_texts[index],
                audio=audio[index, : int(audio_lengths[index].item())],
            )

    def on_validation_epoch_start(self) -> None:
        self._val_texts_pred.clear()
        self._val_texts_ref.clear()
        self._val_langs.clear()
        self._val_examples.reset()

    def on_validation_epoch_end(self) -> None:
        log_wandb_worst_val_examples(
            self.logger,
            self._val_examples.worst_first(),
            sample_rate=int(self.sample_rate),
            epoch=int(self.current_epoch),
            step=int(self.trainer.global_step),
        )

        if not self._val_texts_ref:
            self.log("val/wer/overall", 1.0, prog_bar=True, on_epoch=True)
        else:
            metrics = wer_cer_by_lang(self._val_texts_ref, self._val_texts_pred, self._val_langs)
            for name, value in metrics.items():
                self.log(f"val/{name}", value, prog_bar=True, on_epoch=True)

        for lang, pairs in self.examples.pop_all().items():
            if not pairs:
                continue
            logger.info(f"[VAL][{lang}] --- examples epoch {self.current_epoch} ---")
            for ref, hyp in pairs:
                logger.info(f"[VAL][{lang}] REF: {ref}")
                logger.info(f"[VAL][{lang}] HYP: {hyp}")

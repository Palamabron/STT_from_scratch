from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

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
    wer_cer_by_lang,
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
from SpeechToText.models.ctc_attention.model import FastConformerCTCAttention
from SpeechToText.models.ctc_attention.steps import compute_ctc_attn_losses
from SpeechToText.models.typing import CTCAttnOutput, TrainBatch, ValBatch


@dataclass(frozen=True)
class TrainingStage:
    """Effective training behaviour for the current epoch."""

    effective_ctc_weight: float
    effective_aux_ctc_weight: float
    include_attn: bool
    freeze_encoder: bool
    freeze_ctc_heads: bool
    freeze_decoder: bool
    name: str


class LitFastConformerCTCAttention(pl.LightningModule):
    """Lightning module for joint CTC and attention-decoder training."""

    def __init__(
        self,
        config: Any,
        sp: SentencePieceProcessor,
        rir_bank: tuple[torch.Tensor, ...] | None = None,
        noise_bank: tuple[torch.Tensor, ...] | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.sp = sp

        sp_vocab = int(sp.get_piece_size())
        self.vocab_size = sp_vocab + 1
        self.blank_id = 0
        self.pad_id = sp.pad_id() + 1
        self.bos_id = sp.bos_id() + 1
        self.eos_id = sp.eos_id() + 1

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

        self.net = FastConformerCTCAttention(
            config.model,
            vocab_size=self.vocab_size,
            blank_id=self.blank_id,
            pad_id=self.pad_id,
            bos_id=self.bos_id,
            eos_id=self.eos_id,
        )

        self.ctc_loss = nn.CTCLoss(blank=self.blank_id, zero_infinity=True, reduction="mean")
        self.attn_loss = nn.NLLLoss(ignore_index=self.pad_id, reduction="mean")
        self.examples = ExamplesBuffer(per_lang=2)
        self._val_examples = WorstValExamplesCollector(max_examples=50)
        self.save_hyperparameters(ignore=["sp", "rir_bank", "noise_bank"])

        self._val_texts_pred: list[str] = []
        self._val_texts_ref: list[str] = []
        self._val_langs: list[str] = []
        self._training_stage: TrainingStage | None = None

    def on_fit_start(self) -> None:
        self.featurizer = self.featurizer.to(self.device)

    def _encoder_modules(self) -> list[nn.Module]:
        return [self.net.encoder]

    def _ctc_head_modules(self) -> list[nn.Module]:
        return [self.net.ctc_proj, self.net.aux_projs]

    def _decoder_modules(self) -> list[nn.Module]:
        return [self.net.tok_embed, self.net.pos_embed, self.net.decoder, self.net.dec_proj]

    @staticmethod
    def _set_trainable(modules: list[nn.Module], trainable: bool) -> None:
        for module in modules:
            for parameter in module.parameters():
                parameter.requires_grad = trainable

    @staticmethod
    def _set_module_mode(modules: list[nn.Module], train: bool) -> None:
        for module in modules:
            if train:
                module.train()
            else:
                module.eval()

    def _training_stage_for_epoch(self, epoch: int) -> TrainingStage:
        cfg = self.config
        if cfg.ctc_calibration_epochs > 0 and epoch < cfg.ctc_calibration_epochs:
            return TrainingStage(
                effective_ctc_weight=1.0,
                effective_aux_ctc_weight=float(cfg.aux_ctc_weight),
                include_attn=False,
                freeze_encoder=True,
                freeze_ctc_heads=False,
                freeze_decoder=True,
                name="ctc_calibration",
            )
        if epoch < cfg.decoder_warmup_epochs:
            return TrainingStage(
                effective_ctc_weight=0.0,
                effective_aux_ctc_weight=0.0,
                include_attn=True,
                freeze_encoder=True,
                freeze_ctc_heads=True,
                freeze_decoder=False,
                name="decoder_warmup",
            )

        freeze_encoder = epoch < cfg.freeze_encoder_epochs
        freeze_decoder = (
            cfg.freeze_decoder_after_epoch is not None and epoch >= cfg.freeze_decoder_after_epoch
        )
        return TrainingStage(
            effective_ctc_weight=float(cfg.ctc_weight),
            effective_aux_ctc_weight=float(cfg.aux_ctc_weight),
            include_attn=not freeze_decoder,
            freeze_encoder=freeze_encoder,
            freeze_ctc_heads=freeze_encoder,
            freeze_decoder=freeze_decoder,
            name="joint",
        )

    def _apply_training_stage(self, stage: TrainingStage) -> None:
        self._set_trainable(self._encoder_modules(), not stage.freeze_encoder)
        self._set_trainable(self._ctc_head_modules(), not stage.freeze_ctc_heads)
        self._set_trainable(self._decoder_modules(), not stage.freeze_decoder)
        self._set_module_mode(self._encoder_modules(), not stage.freeze_encoder)
        self._set_module_mode(self._ctc_head_modules(), not stage.freeze_ctc_heads)
        self._set_module_mode(self._decoder_modules(), not stage.freeze_decoder)

    def on_train_epoch_start(self) -> None:
        stage = self._training_stage_for_epoch(int(self.current_epoch))
        self._training_stage = stage
        self._apply_training_stage(stage)
        logger.info(
            "Epoch {} training stage={} ctc_weight={} aux_ctc_weight={} include_attn={}",
            self.current_epoch,
            stage.name,
            stage.effective_ctc_weight,
            stage.effective_aux_ctc_weight,
            stage.include_attn,
        )

    def _encode_batch(
        self,
        audio: torch.Tensor,
        audio_lengths: torch.Tensor,
        clean_pass: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.featurizer.set_current_epoch(self.current_epoch)
        return cast(
            tuple[torch.Tensor, torch.Tensor],
            self.featurizer(audio.to(self.device), audio_lengths, clean_pass=clean_pass),
        )

    def build_decoder_sequences(
        self,
        targets_concat: torch.Tensor,
        target_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build teacher-forced decoder input and target sequences."""
        device = targets_concat.device
        batch_size = int(target_lengths.shape[0])
        max_len = int(target_lengths.max().item()) if batch_size > 0 else 0

        dec_in = torch.full((batch_size, max_len + 1), self.pad_id, dtype=torch.long, device=device)
        dec_out = torch.full(
            (batch_size, max_len + 1), self.pad_id, dtype=torch.long, device=device
        )

        offset = 0
        for index in range(batch_size):
            target_len = int(target_lengths[index].item())
            if target_len == 0:
                dec_in[index, 0] = self.bos_id
                dec_out[index, 0] = self.eos_id
                continue
            sequence = targets_concat[offset : offset + target_len]
            offset += target_len
            dec_in[index, 0] = self.bos_id
            dec_in[index, 1 : target_len + 1] = sequence
            dec_out[index, 0:target_len] = sequence
            dec_out[index, target_len] = self.eos_id
        return dec_in, dec_out

    def forward(
        self,
        feats: torch.Tensor,
        feat_lengths: torch.Tensor,
        decoder_input: torch.Tensor | None = None,
    ) -> CTCAttnOutput:
        return cast(CTCAttnOutput, self.net(feats, feat_lengths, decoder_input=decoder_input))

    def configure_optimizers(self) -> OptimizerLRSchedulerConfig:
        opt_cfg = self.config.optimizer
        total_steps = int(self.trainer.estimated_stepping_batches)

        return configure_adamw_scheduler(
            self,
            optimizer_cfg=opt_cfg,
            total_steps=total_steps,
        )

    def training_step(self, batch: TrainBatch, batch_idx: int) -> torch.Tensor | None:
        stage = self._training_stage or self._training_stage_for_epoch(int(self.current_epoch))
        audio = batch["audio"]
        audio_lengths = batch["audio_length"]
        targets = batch["targets"]
        target_lengths = batch["target_length"]
        clean_pass = batch.get("clean_pass")
        if clean_pass is not None:
            clean_pass = clean_pass.to(self.device)

        feats, feat_lens = self._encode_batch(audio, audio_lengths, clean_pass=clean_pass)
        dec_in: torch.Tensor | None = None
        dec_out: torch.Tensor | None = None
        if stage.include_attn:
            dec_in, dec_out = self.build_decoder_sequences(targets, target_lengths)
        out = self.forward(feats, feat_lens, decoder_input=dec_in)
        if stage.include_attn:
            assert out.dec_log_probs is not None

        filtered = filter_batch_by_encoder_length(batch, out.out_lengths, target_lengths)
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
            if stage.include_attn:
                dec_in, dec_out = self.build_decoder_sequences(targets, target_lengths)
            out = self.forward(feats, feat_lens, decoder_input=dec_in)
            if stage.include_attn:
                assert out.dec_log_probs is not None

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
            aux_ctc_weight=stage.effective_aux_ctc_weight,
            ctc_weight=stage.effective_ctc_weight,
            autocast_device_type="cuda" if out.ctc_log_probs.is_cuda else "cpu",
            attn_loss_fn=self.attn_loss,
            include_attn=stage.include_attn,
        )

        self.log_dict(
            {
                "train/loss": losses.total,
                "train/loss_ctc": losses.ctc_main,
                "train/loss_attn": losses.attn,
            },
            prog_bar=True,
            on_step=True,
            batch_size=audio.size(0),
        )

        return losses.total

    def validation_step(self, batch: ValBatch, batch_idx: int) -> None:
        audio = batch["audio"]
        audio_lengths = batch["audio_length"]
        targets = batch["targets"]
        target_lengths = batch["target_length"]

        stage = self._training_stage or self._training_stage_for_epoch(int(self.current_epoch))

        feats, feat_lens = self._encode_batch(audio, audio_lengths)
        dec_in: torch.Tensor | None = None
        dec_out: torch.Tensor | None = None
        if stage.include_attn:
            dec_in, dec_out = self.build_decoder_sequences(targets, target_lengths)
        out = self.forward(feats, feat_lens, decoder_input=dec_in)
        if stage.include_attn:
            assert out.dec_log_probs is not None

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
            aux_ctc_weight=stage.effective_aux_ctc_weight,
            ctc_weight=stage.effective_ctc_weight,
            autocast_device_type="cuda" if out.ctc_log_probs.is_cuda else "cpu",
            attn_loss_fn=self.attn_loss,
            include_attn=stage.include_attn,
        )

        self.log_dict(
            {
                "val/loss": losses.total,
                "val/loss_ctc": losses.ctc_main,
                "val/loss_attn": losses.attn,
            },
            prog_bar=True,
            on_epoch=True,
            batch_size=audio.size(0),
        )

        out_infer = self.forward(feats, feat_lens, decoder_input=None)

        val_mode = getattr(self.config, "val_decode_mode", "ctc_greedy")
        if val_mode == "attention_greedy":
            from SpeechToText.models.common.inference import (
                ctc_attention_special_tokens,
                decode_ctc_attention_attention_greedy,
            )

            enc, out_lengths, _aux = self.net.encode(feats, feat_lens)
            tokens = ctc_attention_special_tokens(self)
            pred_texts = [
                decode_ctc_attention_attention_greedy(
                    self,
                    enc,
                    out_lengths,
                    sample_index=index,
                    sp=self.sp,
                    tokens=tokens,
                )
                for index in range(audio.size(0))
            ]
        else:
            decoded = greedy_ctc_decode(
                out_infer.ctc_log_probs, out_infer.out_lengths, blank_id=self.blank_id
            )
            pred_texts = ctc_ids_to_texts_spm(self.sp, decoded)

        self._val_examples.accumulate_blank_stats(
            out_infer.ctc_log_probs, out_infer.out_lengths, self.blank_id
        )

        texts_ref = batch["text"]
        langs = batch.get("language", ["unknown"] * len(pred_texts))
        datasets = batch.get("dataset", ["unknown"] * len(pred_texts))

        self._val_texts_pred.extend(pred_texts)
        self._val_texts_ref.extend(texts_ref)
        self._val_langs.extend(langs)

        for index, (ref, hyp, lang) in enumerate(zip(texts_ref, pred_texts, langs, strict=True)):
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

        for lang, pairs in self.examples.pop_all().items():
            if pairs:
                ref, hyp = pairs[0]
                logger.info("[VAL][{}] REF: {} | HYP: {}", lang, ref, hyp)

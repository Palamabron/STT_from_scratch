from __future__ import annotations

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
    wer_cer_by_lang_with_mer,
)
from SpeechToText.models.common.batch_filter import filter_batch_by_encoder_length
from SpeechToText.models.common.optimizers import configure_adamw_noam
from SpeechToText.models.common.validation_logging import (
    WorstValExamplesCollector,
    log_wandb_worst_val_examples,
)
from SpeechToText.models.ctc_attention.model import FastConformerCTCAttention
from SpeechToText.models.ctc_attention.steps import compute_ctc_attn_losses
from SpeechToText.models.typing import CTCAttnOutput, TrainBatch, ValBatch


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

        gpu_augment = GPUAudioAugmentation(config.audio_augment, rir_bank, noise_bank)
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

    def on_fit_start(self) -> None:
        self.featurizer = self.featurizer.to(self.device)

    def _encode_batch(
        self, audio: torch.Tensor, audio_lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.featurizer.set_current_epoch(self.current_epoch)
        return cast(
            tuple[torch.Tensor, torch.Tensor], self.featurizer(audio.to(self.device), audio_lengths)
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
        d_model = int(self.config.model.encoder.d_model)
        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = int(total_steps * opt_cfg.warmup_ratio)

        return configure_adamw_noam(
            self,
            lr=opt_cfg.lr,
            betas=opt_cfg.betas,
            weight_decay=getattr(opt_cfg, "weight_decay", 0.01),
            warmup_steps=max(1, warmup_steps),
            d_model=d_model,
        )

    def training_step(self, batch: TrainBatch, batch_idx: int) -> torch.Tensor | None:
        audio = batch["audio"]
        audio_lengths = batch["audio_length"]
        targets = batch["targets"]
        target_lengths = batch["target_length"]

        feats, feat_lens = self._encode_batch(audio, audio_lengths)
        dec_in, dec_out = self.build_decoder_sequences(targets, target_lengths)
        out = self.forward(feats, feat_lens, decoder_input=dec_in)
        assert out.dec_log_probs is not None

        filtered = filter_batch_by_encoder_length(batch, out.out_lengths, target_lengths)
        if filtered is None:
            return None
        if filtered[0] is not batch:
            batch = filtered[0]
            audio = batch["audio"]
            audio_lengths = batch["audio_length"]
            targets = batch["targets"]
            target_lengths = batch["target_length"]
            feats, feat_lens = self._encode_batch(audio, audio_lengths)
            dec_in, dec_out = self.build_decoder_sequences(targets, target_lengths)
            out = self.forward(feats, feat_lens, decoder_input=dec_in)
            assert out.dec_log_probs is not None

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
            aux_ctc_weight=self.config.aux_ctc_weight,
            ctc_weight=self.config.ctc_weight,
            autocast_device_type="cuda" if out.ctc_log_probs.is_cuda else "cpu",
            attn_loss_fn=self.attn_loss,
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

        feats, feat_lens = self._encode_batch(audio, audio_lengths)
        out = self.forward(feats, feat_lens, decoder_input=None)

        loss = self.ctc_loss(
            out.ctc_log_probs.transpose(0, 1), targets, out.out_lengths, target_lengths
        )
        self.log("val/loss", loss, prog_bar=True, on_epoch=True, batch_size=audio.size(0))

        decoded = greedy_ctc_decode(out.ctc_log_probs, out.out_lengths, blank_id=self.blank_id)
        self._val_examples.accumulate_blank_stats(out.ctc_log_probs, out.out_lengths, self.blank_id)
        pred_texts = ctc_ids_to_texts_spm(self.sp, decoded)

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
        if not self._val_texts_ref:
            self.log("val/wer/overall", 1.0, prog_bar=True, on_epoch=True)
        else:
            metrics = wer_cer_by_lang_with_mer(
                self._val_texts_ref, self._val_texts_pred, self._val_langs
            )
            for name, value in metrics.items():
                self.log(f"val/{name}", value, prog_bar=True, on_epoch=True)

        self.log(
            "val/blank_fraction",
            self._val_examples.blank_fraction(),
            prog_bar=True,
            on_epoch=True,
        )
        log_wandb_worst_val_examples(
            self.logger,
            self._val_examples.worst_first(),
            sample_rate=int(self.config.data.features.sample_rate),
            epoch=int(self.current_epoch),
        )

        for lang, pairs in self.examples.pop_all().items():
            if pairs:
                ref, hyp = pairs[0]
                logger.info("[VAL][{}] REF: {} | HYP: {}", lang, ref, hyp)

from __future__ import annotations

from typing import Final

import torch
import torch.nn as nn
import torchaudio.transforms as AT

from .augmentation import GPUAudioAugmentation, SpecAugment
from .dataset import FeatureConfig

FEATURE_NORM_EPS = 1e-5


class WaveformFeaturizer(nn.Module):
    """Convert raw waveforms to normalized log-mel features."""

    def __init__(
        self,
        feat_cfg: FeatureConfig,
        *,
        spec_augment: SpecAugment | None = None,
        gpu_augment: GPUAudioAugmentation | None = None,
    ) -> None:
        super().__init__()
        self.sample_rate: Final[int] = feat_cfg.sample_rate
        self.hop_length: Final[int] = int(self.sample_rate * feat_cfg.hop_length_ms / 1000.0)

        self.mel_spec: Final[AT.MelSpectrogram] = AT.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=feat_cfg.n_fft,
            win_length=int(self.sample_rate * feat_cfg.win_length_ms / 1000.0),
            hop_length=self.hop_length,
            n_mels=feat_cfg.n_mels,
            power=2.0,
            center=True,
            normalized=False,
        )
        self.amplitude_to_db: Final[AT.AmplitudeToDB] = AT.AmplitudeToDB(top_db=feat_cfg.top_db)
        self.spec_augment = spec_augment
        self.gpu_augment = gpu_augment
        self._current_epoch = 0

    def set_current_epoch(self, epoch: int) -> None:
        self._current_epoch = int(epoch)
        if self.spec_augment is not None:
            self.spec_augment.set_current_epoch(self._current_epoch)
        if self.gpu_augment is not None:
            self.gpu_augment.set_current_epoch(self._current_epoch)

    def audio_lengths_to_feat_lengths(self, audio_lengths: torch.Tensor) -> torch.Tensor:
        return (audio_lengths // self.hop_length) + 1

    def forward(
        self,
        audio: torch.Tensor,
        audio_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        wav = audio
        if self.training and self.gpu_augment is not None:
            wav = self.gpu_augment(wav)

        mel = self.mel_spec(wav)
        mel = self.amplitude_to_db(mel)
        feats = mel.transpose(1, 2).contiguous()
        feat_lens = self.audio_lengths_to_feat_lengths(audio_lengths.to(wav.device))

        if self.training and self.spec_augment is not None:
            feats = self.spec_augment(feats)

        batch_size, time_steps, _ = feats.shape
        mask = torch.arange(time_steps, device=feats.device).unsqueeze(0) < feat_lens.unsqueeze(1)
        mask = mask.unsqueeze(-1)

        masked_feats = feats.masked_fill(~mask, 0.0)
        denom = feat_lens.clamp_min(1).to(feats.dtype).view(-1, 1, 1)
        mean = masked_feats.sum(dim=1, keepdim=True) / denom
        variance = ((masked_feats - mean).masked_fill(~mask, 0.0) ** 2).sum(
            dim=1, keepdim=True
        ) / denom
        feats = (feats - mean) / variance.sqrt().clamp_min(FEATURE_NORM_EPS)
        return feats, feat_lens

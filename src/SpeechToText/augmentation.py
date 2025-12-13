from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

import torch
import torchaudio.transforms as T
from loguru import logger


@dataclass
class SpecAugmentConfig:
    freq_masks: int = 2
    time_masks: int = 10
    freq_width: int = 30
    time_width_fraction: float = 0.1


class SpecAugment(torch.nn.Module):
    def __init__(self, cfg: SpecAugmentConfig, augment_start_epoch: int = 3):
        super().__init__()
        self.cfg = cfg
        self.augment_start_epoch = augment_start_epoch
        self.current_epoch = 0

    def set_current_epoch(self, epoch: int) -> None:
        self.current_epoch = epoch

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return feats
        if self.current_epoch < self.augment_start_epoch:
            return feats

        b, t, f = feats.shape
        feats = feats.transpose(1, 2)

        for _ in range(self.cfg.freq_masks):
            fm = T.FrequencyMasking(freq_mask_param=self.cfg.freq_width)
            feats = fm(feats)

        max_time_width = max(1, int(t * self.cfg.time_width_fraction))
        for _ in range(self.cfg.time_masks):
            tm = T.TimeMasking(time_mask_param=max_time_width)
            feats = tm(feats)

        feats = feats.transpose(1, 2)
        return feats


@dataclass
class AudioAugmentConfig:
    speed_factors: Sequence[float] = (0.95, 0.98, 1.0, 1.02, 1.05)
    prob_apply: float = 0.7
    add_noise: bool = True
    noise_prob: float = 0.3
    noise_scale: float = 0.005


class AudioAugmentation:
    def __init__(
        self, cfg: AudioAugmentConfig, sample_rate: int = 16_000, augment_start_epoch: int = 3
    ):
        self.cfg = cfg
        self.sample_rate = sample_rate
        self.augment_start_epoch = augment_start_epoch
        self.current_epoch = 0

    def set_current_epoch(self, epoch: int) -> None:
        self.current_epoch = epoch

    def speed_perturb(self, audio: torch.Tensor, factor: float) -> torch.Tensor:
        if factor == 1.0:
            return audio
        new_sr = int(self.sample_rate * factor)
        resample1 = T.Resample(orig_freq=self.sample_rate, new_freq=new_sr)
        resample2 = T.Resample(orig_freq=new_sr, new_freq=self.sample_rate)
        x = cast(torch.Tensor, resample1(audio))
        x = cast(torch.Tensor, resample2(x))
        return x

    def add_gaussian_noise(self, audio: torch.Tensor) -> torch.Tensor:
        if not self.cfg.add_noise or torch.rand(1).item() > self.cfg.noise_prob:
            return audio
        noise = torch.randn_like(audio) * self.cfg.noise_scale
        return audio + noise

    def __call__(self, audio: torch.Tensor) -> torch.Tensor:
        if self.current_epoch < self.augment_start_epoch:
            return audio

        if torch.rand(1).item() < self.cfg.prob_apply:
            idx = int(torch.randint(0, len(self.cfg.speed_factors), (1,)).item())
            factor = self.cfg.speed_factors[idx]
            try:
                audio = self.speed_perturb(audio, factor)
            except Exception as e:
                logger.warning(f"Audio speed perturbation failed: {e}")

        audio = self.add_gaussian_noise(audio)
        return audio

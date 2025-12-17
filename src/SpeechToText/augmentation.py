from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final, cast

import torch
import torch.nn.functional as F
import torchaudio.functional as AF
import torchaudio.transforms as T


@dataclass(slots=True)
class SpecAugmentConfig:
    freq_masks: int = 2
    time_masks: int = 10
    freq_width: int = 30
    time_width_fraction: float = 0.1


class SpecAugment(torch.nn.Module):
    def __init__(self, cfg: SpecAugmentConfig, augment_start_epoch: int = 3) -> None:
        super().__init__()
        self.cfg: Final = cfg
        self.augment_start_epoch: Final = augment_start_epoch
        self.current_epoch = 0

        self._freq_mask = T.FrequencyMasking(freq_mask_param=cfg.freq_width)

    def set_current_epoch(self, epoch: int) -> None:
        self.current_epoch = int(epoch)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        if (not self.training) or (self.current_epoch < self.augment_start_epoch):
            return feats

        if feats.dim() != 3:
            raise ValueError(f"Expected (B,T,F); got {tuple(feats.shape)}")

        b, t, f = feats.shape
        x = feats.transpose(1, 2)

        for _ in range(self.cfg.freq_masks):
            x = self._freq_mask(x)

        max_time_width = max(1, int(t * self.cfg.time_width_fraction))
        time_mask = T.TimeMasking(time_mask_param=max_time_width)
        for _ in range(self.cfg.time_masks):
            x = time_mask(x)

        return x.transpose(1, 2)


@dataclass(slots=True)
class AudioAugmentConfig:
    prob_apply: float = 0.7

    speed_factors: tuple[float, ...] = (0.95, 0.98, 1.0, 1.02, 1.05)
    speed_prob: float = 0.7

    gain_prob: float = 0.3
    gain_db_min: float = -6.0
    gain_db_max: float = 6.0

    phone_prob: float = 0.2
    phone_down_sr: int = 8000
    phone_highpass_hz: float = 300.0
    phone_lowpass_hz: float = 3400.0

    rir_prob: float = 0.2
    rir_bank: tuple[torch.Tensor, ...] | None = None

    bg_noise_prob: float = 0.3
    snr_db_min: float = 5.0
    snr_db_max: float = 20.0
    noise_bank: tuple[torch.Tensor, ...] | None = None

    gaussian_prob: float = 0.0
    gaussian_scale: float = 0.005


class AudioAugmentation:
    def __init__(
        self, cfg: AudioAugmentConfig, sample_rate: int = 16_000, augment_start_epoch: int = 3
    ) -> None:
        self.cfg: Final = cfg
        self.sample_rate: Final = sample_rate
        self.augment_start_epoch: Final = augment_start_epoch
        self.current_epoch = 0

        pairs: list[tuple[T.Resample, T.Resample] | None] = []
        for f in cfg.speed_factors:
            if f == 1.0:
                pairs.append(None)
            else:
                mid_sr = int(round(sample_rate * f))
                pairs.append((T.Resample(sample_rate, mid_sr), T.Resample(mid_sr, sample_rate)))
        self._speed_pairs: Final[tuple[tuple[T.Resample, T.Resample] | None, ...]] = tuple(pairs)

        self._phone_down: Final = T.Resample(sample_rate, cfg.phone_down_sr)
        self._phone_up: Final = T.Resample(cfg.phone_down_sr, sample_rate)

    def set_current_epoch(self, epoch: int) -> None:
        self.current_epoch = int(epoch)

    @staticmethod
    def _as_1d(x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            return x
        if x.dim() == 2 and x.size(0) == 1:
            return x.squeeze(0)
        raise ValueError(f"Expected (T,) or (1,T); got {tuple(x.shape)}")

    @staticmethod
    def _fix_len(x: torch.Tensor, target_len: int) -> torch.Tensor:
        n = x.numel()
        if n == target_len:
            return x
        if n > target_len:
            return x[:target_len]
        return F.pad(x, (0, target_len - n))

    @staticmethod
    def _rand_uniform_like(x: torch.Tensor, a: float, b: float) -> float:
        return float(x.new_empty(()).uniform_(a, b).item())

    def _speed(self, x: torch.Tensor) -> torch.Tensor:
        if float(torch.rand(()).item()) > self.cfg.speed_prob:
            return x
        idx = int(torch.randint(0, len(self.cfg.speed_factors), ()).item())
        pair = self._speed_pairs[idx]
        if pair is None:
            return x
        down, up = pair
        t0 = x.numel()
        y = up(down(x.unsqueeze(0))).squeeze(0)
        y = self._fix_len(y, t0)
        return y

    def _gain_db(self, x: torch.Tensor) -> torch.Tensor:
        if float(torch.rand(()).item()) > self.cfg.gain_prob:
            return x
        gain_db = self._rand_uniform_like(x, self.cfg.gain_db_min, self.cfg.gain_db_max)
        gain = math.pow(10.0, gain_db / 20.0)
        return x * gain

    def _telephone(self, x: torch.Tensor) -> torch.Tensor:
        if float(torch.rand(()).item()) > self.cfg.phone_prob:
            return x
        t0 = x.numel()
        y = self._phone_up(self._phone_down(x.unsqueeze(0))).squeeze(0)
        y = self._fix_len(y, t0)
        y = cast(
            torch.Tensor,
            AF.highpass_biquad(
                y,
                sample_rate=self.sample_rate,
                cutoff_freq=self.cfg.phone_highpass_hz,
            ),
        )
        y = cast(
            torch.Tensor,
            AF.lowpass_biquad(
                y,
                sample_rate=self.sample_rate,
                cutoff_freq=self.cfg.phone_lowpass_hz,
            ),
        )
        return y

    def _rir(self, x: torch.Tensor) -> torch.Tensor:
        bank = self.cfg.rir_bank
        if float(torch.rand(()).item()) > self.cfg.rir_prob or not bank:
            return x
        rir = self._as_1d(bank[int(torch.randint(0, len(bank), ()).item())]).to(
            device=x.device, dtype=x.dtype
        )
        rir = rir / rir.abs().sum().clamp_min(1e-6)
        w = rir.flip(0).view(1, 1, -1)
        y = F.conv1d(x.view(1, 1, -1), w, padding=rir.numel() - 1).view(-1)
        return y[: x.numel()]

    def _bg_noise_snr(self, x: torch.Tensor) -> torch.Tensor:
        bank = self.cfg.noise_bank
        if float(torch.rand(()).item()) > self.cfg.bg_noise_prob or not bank:
            return x

        noise = self._as_1d(bank[int(torch.randint(0, len(bank), ()).item())]).to(
            device=x.device, dtype=x.dtype
        )
        t0 = x.numel()

        if noise.numel() < t0:
            reps = (t0 + noise.numel() - 1) // noise.numel()
            noise = noise.repeat(reps)[:t0]
        else:
            start = int(torch.randint(0, max(1, noise.numel() - t0 + 1), ()).item())
            noise = noise[start : start + t0]

        snr_db = self._rand_uniform_like(x, self.cfg.snr_db_min, self.cfg.snr_db_max)
        return cast(torch.Tensor, AF.add_noise(x, noise, snr=x.new_tensor(snr_db)))

    def _gaussian(self, x: torch.Tensor) -> torch.Tensor:
        if self.cfg.gaussian_prob <= 0.0 or float(torch.rand(()).item()) > self.cfg.gaussian_prob:
            return x
        x_noised = x.add(torch.randn_like(x), alpha=self.cfg.gaussian_scale)
        return x_noised

    @torch.inference_mode()
    def __call__(self, audio: torch.Tensor) -> torch.Tensor:
        x = self._as_1d(audio).contiguous()

        if self.current_epoch < self.augment_start_epoch:
            return x
        if float(torch.rand(()).item()) > self.cfg.prob_apply:
            return x

        x = self._speed(x)
        x = self._gain_db(x)
        x = self._telephone(x)
        x = self._rir(x)
        x = self._bg_noise_snr(x)
        x = self._gaussian(x)
        x = x.clamp(-1.0, 1.0)
        return x

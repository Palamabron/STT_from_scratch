from __future__ import annotations

import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Final, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.functional as AF
import torchaudio.transforms as T
from loguru import logger
from tqdm import tqdm

DEFAULT_NOISE_BANK_PATH: Final[str] = "data/augment/noise_bank.pt"
DEFAULT_RIR_BANK_PATH: Final[str] = "data/augment/rir_bank.pt"


@dataclass(slots=True)
class SpecAugmentConfig:
    """SpecAugment mask counts and widths."""

    freq_masks: int = 2
    time_masks: int = 10
    freq_width: int = 30
    time_width_fraction: float = 0.1


class SpecAugment(nn.Module):
    """Apply frequency and time masking to log-mel features."""

    def __init__(self, cfg: SpecAugmentConfig, augment_start_epoch: int = 3) -> None:
        super().__init__()
        self.cfg: Final = cfg
        self.augment_start_epoch: Final[int] = augment_start_epoch
        self.current_epoch: int = 0
        self._freq_mask: Final[T.FrequencyMasking] = T.FrequencyMasking(
            freq_mask_param=cfg.freq_width
        )

    def set_current_epoch(self, epoch: int) -> None:
        self.current_epoch = int(epoch)

    @torch.inference_mode()
    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        """Mask log-mel features with shape ``[batch, time, freq]``."""
        if not self.training or self.current_epoch < self.augment_start_epoch:
            return feats

        masked = feats.transpose(1, 2).contiguous()

        for _ in range(self.cfg.freq_masks):
            masked = self._freq_mask(masked)

        _, _, time_steps = masked.shape
        max_width = max(1, int(time_steps * self.cfg.time_width_fraction))
        for _ in range(self.cfg.time_masks):
            width = int(torch.randint(0, max_width + 1, ()).item())
            if width <= 0:
                continue
            start = int(torch.randint(0, max(1, time_steps - width + 1), ()).item())
            masked[:, :, start : start + width] = 0.0

        return masked.transpose(1, 2).contiguous()


@dataclass(slots=True)
class AudioAugmentConfig:
    """Audio-domain augmentation probabilities and ranges."""

    augment_start_epoch: int = 0
    heavy_augment_start_epoch: int = 0
    prob_apply: float = 0.5  # deprecated, ignored; kept for checkpoint/worker pickle compat
    speed_factors: tuple[float, ...] = (0.95, 0.98, 1.02, 1.05)
    speed_prob: float = 0.5
    gain_prob: float = 0.3
    gain_db_min: float = -6.0
    gain_db_max: float = 6.0
    gain_db_mean: float = 0.0
    gain_db_std: float = 4.0
    phone_prob: float = 0.0
    phone_down_sr: int = 8000
    phone_highpass_hz: float = 300.0
    phone_lowpass_hz: float = 3400.0
    rir_prob: float = 0.1
    rir_bank: tuple[torch.Tensor, ...] | None = None
    bg_noise_prob: float = 0.3
    snr_db_min: float = 5.0
    snr_db_max: float = 20.0
    snr_db_mean: float = 10.0
    snr_db_std: float = 5.0
    noise_bank: tuple[torch.Tensor, ...] | None = None
    gaussian_prob: float = 0.0
    gaussian_scale: float = 0.005
    clean_pass_prob: float = 0.08

    def __getattr__(self, name: str) -> float:
        if name == "clean_pass_prob":
            return 0.08
        raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")


def _postprocess_bank_clip(
    waveform: torch.Tensor,
    *,
    sample_rate: int,
    min_len_sec: float,
    normalize_rirs: bool,
    max_rir_len_sec: float | None,
) -> torch.Tensor | None:
    waveform = waveform.detach().float().reshape(-1)
    if waveform.numel() < int(min_len_sec * sample_rate):
        return None

    if normalize_rirs and max_rir_len_sec is not None:
        max_samples = int(max_rir_len_sec * sample_rate)
        if waveform.size(-1) > max_samples:
            waveform = waveform[:max_samples]

    if normalize_rirs:
        energy = waveform.abs().sum()
        if energy > 1e-6:
            waveform = waveform / energy

    return waveform


def _load_bank_from_pt(
    pt_path: Path,
    *,
    sample_rate: int,
    min_len_sec: float,
    normalize_rirs: bool,
    max_rir_len_sec: float | None,
    max_bank_items: int | None = None,
) -> tuple[torch.Tensor, ...] | None:
    raw = torch.load(str(pt_path), map_location="cpu", weights_only=False)
    if isinstance(raw, tuple):
        clips = list(raw)
    elif isinstance(raw, list):
        clips = raw
    else:
        logger.warning("Unsupported bank format in {}: {}", pt_path, type(raw))
        return None

    loaded: list[torch.Tensor] = []
    for clip in clips:
        if not isinstance(clip, torch.Tensor):
            continue
        processed = _postprocess_bank_clip(
            clip,
            sample_rate=sample_rate,
            min_len_sec=min_len_sec,
            normalize_rirs=normalize_rirs,
            max_rir_len_sec=max_rir_len_sec,
        )
        if processed is not None:
            loaded.append(processed)

    if not loaded:
        return None

    if max_bank_items is not None and len(loaded) > max_bank_items:
        rng = random.Random(42)
        loaded = rng.sample(loaded, max_bank_items)
        logger.info(
            "Subsampled bank {} to {} items (max_bank_items={})",
            pt_path,
            len(loaded),
            max_bank_items,
        )

    logger.info("Loaded {} clips from prebuilt bank {}", len(loaded), pt_path)
    return tuple(loaded)


def _load_bank_from_directory(
    root_path: Path,
    *,
    sample_rate: int,
    max_files: int,
    min_len_sec: float,
    normalize_rirs: bool,
    max_size_bytes: int,
    max_rir_len_sec: float | None,
) -> tuple[torch.Tensor, ...] | None:
    import soundfile as sf

    logger.info("Scanning for audio files in {}", root_path)
    all_files: list[Path] = []
    scan_limit = max_files * 3
    scanned = 0
    for extension in ("*.wav", "*.flac"):
        for path in root_path.rglob(extension):
            all_files.append(path)
            scanned += 1
            if scanned >= scan_limit:
                break
        if scanned >= scan_limit:
            break

    if not all_files:
        return None

    random.shuffle(all_files)
    files_to_load = all_files[:max_files]
    loaded: list[torch.Tensor] = []

    for path in tqdm(files_to_load, desc=f"Loading bank from {root_path.name}"):
        try:
            if os.path.getsize(path) > max_size_bytes:
                continue

            data, sr = sf.read(str(path))
            wav = torch.from_numpy(data).float()

            if wav.ndim == 1:
                wav = wav.unsqueeze(0)
            else:
                wav = wav.transpose(0, 1)

            if wav.shape[0] > 1:
                wav = wav.mean(dim=0, keepdim=True)
            if sr != sample_rate:
                wav = AF.resample(wav, sr, sample_rate)

            processed = _postprocess_bank_clip(
                wav.squeeze(0),
                sample_rate=sample_rate,
                min_len_sec=min_len_sec,
                normalize_rirs=normalize_rirs,
                max_rir_len_sec=max_rir_len_sec,
            )
            if processed is not None:
                loaded.append(processed)
        except OSError:
            continue

    if not loaded:
        return None

    logger.info("Loaded {} clips into memory from {}", len(loaded), root_path)
    return tuple(loaded)


def load_audio_bank(
    root_path: str | None,
    sample_rate: int = 16000,
    max_files: int = 2000,
    min_len_sec: float = 1.0,
    normalize_rirs: bool = False,
    max_size_bytes: int = 20 * 1024 * 1024,
    max_rir_len_sec: float | None = 0.5,
    max_bank_items: int | None = None,
) -> tuple[torch.Tensor, ...] | None:
    """Load a bank of mono waveforms from a ``.pt`` archive or directory tree.

    Args:
        root_path: Path to a ``torch.save`` tuple of 1-D tensors, or a directory
            containing ``.wav`` / ``.flac`` files.
        sample_rate: Target sample rate for directory scans.
        max_files: Maximum number of clips to keep when scanning directories.
        min_len_sec: Minimum clip duration in seconds.
        normalize_rirs: Whether to peak-normalize impulse responses.
        max_size_bytes: Skip files larger than this size during directory scans.
        max_rir_len_sec: Truncate RIR clips to this duration when set.
        max_bank_items: Randomly subsample large ``.pt`` banks to this many clips.

    Returns:
        Tuple of 1-D waveform tensors, or ``None`` when no clips were loaded.
    """
    if not root_path or not os.path.exists(root_path):
        return None

    path = Path(root_path)
    if path.is_file() and path.suffix == ".pt":
        return _load_bank_from_pt(
            path,
            sample_rate=sample_rate,
            min_len_sec=min_len_sec,
            normalize_rirs=normalize_rirs,
            max_rir_len_sec=max_rir_len_sec,
            max_bank_items=max_bank_items,
        )

    if path.is_dir():
        return _load_bank_from_directory(
            path,
            sample_rate=sample_rate,
            max_files=max_files,
            min_len_sec=min_len_sec,
            normalize_rirs=normalize_rirs,
            max_size_bytes=max_size_bytes,
            max_rir_len_sec=max_rir_len_sec,
        )

    logger.warning("Audio bank path exists but is not a .pt file or directory: {}", path)
    return None


class AudioAugmentation:
    """CPU-side waveform augmentations applied before featurization."""

    def __init__(
        self, cfg: AudioAugmentConfig, sample_rate: int = 16_000, augment_start_epoch: int = 3
    ) -> None:
        self.cfg = cfg
        self.sample_rate = sample_rate
        self.augment_start_epoch = augment_start_epoch
        self.current_epoch = 0
        self._phone_down = T.Resample(sample_rate, cfg.phone_down_sr)
        self._phone_up = T.Resample(cfg.phone_down_sr, sample_rate)

    def set_current_epoch(self, epoch: int) -> None:
        self.current_epoch = int(epoch)

    @staticmethod
    def _speed(x: torch.Tensor, cfg: AudioAugmentConfig) -> torch.Tensor:
        if cfg.speed_prob <= 0.0 or float(torch.rand(()).item()) > cfg.speed_prob:
            return x
        factor_index = int(torch.randint(0, len(cfg.speed_factors), ()).item())
        factor = float(cfg.speed_factors[factor_index])
        if factor == 1.0:
            return x

        new_len = max(1, int(round(x.numel() / factor)))
        return F.interpolate(
            x.view(1, 1, -1), size=new_len, mode="linear", align_corners=False
        ).view(-1)

    @staticmethod
    def _telephone(
        x: torch.Tensor, cfg: AudioAugmentConfig, down: T.Resample, up: T.Resample
    ) -> torch.Tensor:
        if float(torch.rand(()).item()) > cfg.phone_prob:
            return x
        original_len = x.numel()
        y = up(down(x.unsqueeze(0))).squeeze(0)
        if y.numel() > original_len:
            y = y[:original_len]
        elif y.numel() < original_len:
            y = F.pad(y, (0, original_len - y.numel()))
        return cast(torch.Tensor, y)

    def __call__(self, audio: torch.Tensor) -> torch.Tensor:
        if self.current_epoch < self.augment_start_epoch:
            return audio

        waveform = audio
        if self.cfg.speed_prob > 0.0:
            waveform = self._speed(waveform, self.cfg)
        if self.cfg.phone_prob > 0.0:
            waveform = self._telephone(waveform, self.cfg, self._phone_down, self._phone_up)
        return waveform


def _pack_waveform_bank(
    bank: tuple[torch.Tensor, ...] | None,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Pack variable-length 1-D clips into ``[n_clips, max_len]`` for fast GPU indexing."""
    if not bank:
        return None
    lengths = torch.tensor(
        [int(clip.reshape(-1).numel()) for clip in bank],
        dtype=torch.long,
    )
    max_len = int(lengths.max().item())
    packed = torch.zeros(len(bank), max_len, dtype=torch.float32)
    for index, clip in enumerate(bank):
        flat = clip.reshape(-1)
        packed[index, : flat.numel()] = flat
    return packed, lengths


class GPUAudioAugmentation(nn.Module):
    """GPU-side augmentations for batched waveforms during training."""

    def __init__(
        self,
        cfg: AudioAugmentConfig,
        rir_bank: tuple[torch.Tensor, ...] | None,
        noise_bank: tuple[torch.Tensor, ...] | None,
        augment_start_epoch: int = 0,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.augment_start_epoch = int(augment_start_epoch)
        rir_packed = _pack_waveform_bank(rir_bank)
        noise_packed = _pack_waveform_bank(noise_bank)
        if rir_packed is not None:
            self.register_buffer("rir_packed", rir_packed[0], persistent=False)
        else:
            self.rir_packed = None
        if noise_packed is not None:
            self.register_buffer("noise_packed", noise_packed[0], persistent=False)
            self.register_buffer("noise_lengths", noise_packed[1], persistent=False)
        else:
            self.noise_packed = None
            self.noise_lengths = None
        self.current_epoch = 0

    def set_current_epoch(self, epoch: int) -> None:
        self.current_epoch = int(epoch)

    @torch.inference_mode()
    def forward(
        self,
        audio: torch.Tensor,
        clean_pass: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply gain, noise, and reverberation augmentations to ``[batch, time]``."""
        if not self.training:
            return audio

        if self.current_epoch < self.augment_start_epoch:
            return audio

        batch_size, time_steps = audio.shape
        device = audio.device

        if self.cfg.gain_prob > 0.0:
            gain_mask = torch.rand(batch_size, device=device) < self.cfg.gain_prob
            if clean_pass is not None:
                gain_mask = gain_mask & ~clean_pass
            if gain_mask.any():
                gains_db = torch.zeros(batch_size, device=device).normal_(
                    self.cfg.gain_db_mean, self.cfg.gain_db_std
                )
                gains_db = gains_db.clamp(self.cfg.gain_db_min, self.cfg.gain_db_max)
                gains = torch.pow(10.0, gains_db / 20.0)
                gains = torch.where(gain_mask, gains, torch.ones_like(gains))
                audio = audio * gains.unsqueeze(1)

        if self.cfg.gaussian_prob > 0.0:
            noise_mask = torch.rand(batch_size, device=device) < self.cfg.gaussian_prob
            if clean_pass is not None:
                noise_mask = noise_mask & ~clean_pass
            if noise_mask.any():
                noise = torch.randn_like(audio) * self.cfg.gaussian_scale
                audio = torch.where(noise_mask.unsqueeze(1), audio + noise, audio)

        if self.current_epoch < self.cfg.heavy_augment_start_epoch:
            return audio.clamp(-1.0, 1.0)

        if self.cfg.rir_prob > 0.0 and self.rir_packed is not None:
            rir_mask = torch.rand(batch_size, device=device) < self.cfg.rir_prob
            if clean_pass is not None:
                rir_mask = rir_mask & ~clean_pass
            if rir_mask.any():
                indices = torch.randint(
                    0, int(self.rir_packed.size(0)), (batch_size,), device=device
                )
                selected_rirs = self.rir_packed[indices]
                max_rir_len = int(selected_rirs.size(1))
                rir_batch = selected_rirs.unsqueeze(1).flip(2)

                padding = max_rir_len - 1
                convolved = F.conv1d(
                    audio.unsqueeze(0),
                    rir_batch,
                    padding=padding,
                    groups=batch_size,
                ).squeeze(0)[:, :time_steps]
                audio = torch.where(rir_mask.unsqueeze(1), convolved, audio)

        if (
            self.cfg.bg_noise_prob > 0.0
            and self.noise_packed is not None
            and self.noise_lengths is not None
        ):
            noise_mask = torch.rand(batch_size, device=device) < self.cfg.bg_noise_prob
            if clean_pass is not None:
                noise_mask = noise_mask & ~clean_pass
            if noise_mask.any():
                indices = torch.randint(
                    0, int(self.noise_packed.size(0)), (batch_size,), device=device
                )
                snrs_db = torch.zeros(batch_size, device=device).normal_(
                    self.cfg.snr_db_mean, self.cfg.snr_db_std
                )
                snrs_db = snrs_db.clamp(self.cfg.snr_db_min, self.cfg.snr_db_max)

                max_starts = (self.noise_lengths[indices] - time_steps).clamp(min=0)
                starts = (
                    torch.rand(batch_size, device=device) * max_starts.to(torch.float32)
                ).long()

                for index in range(batch_size):
                    if not noise_mask[index]:
                        continue
                    noise_len = int(self.noise_lengths[indices[index]].item())
                    noise = self.noise_packed[indices[index], :noise_len]
                    if noise_len < time_steps:
                        repeats = (time_steps + noise_len - 1) // noise_len
                        noise = noise.repeat(repeats)[:time_steps]
                    else:
                        start = int(starts[index].item())
                        noise = noise[start : start + time_steps]

                    signal_power = audio[index].pow(2).mean()
                    noise_power = noise.pow(2).mean()
                    if noise_power > 0:
                        target_noise_power = signal_power / (10.0 ** (snrs_db[index] / 10.0))
                        scale = (target_noise_power / noise_power).sqrt()
                        audio[index] = audio[index] + noise * scale

        return audio.clamp(-1.0, 1.0)

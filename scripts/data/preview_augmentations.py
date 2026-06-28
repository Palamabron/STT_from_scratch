#!/usr/bin/env python3
"""Generate listenable augmentation previews (clean / SpecAugment phase / full pipeline).

SpecAugment masks log-mel during training (not directly on waveforms). For listening we
approximate the time-masking part as short muted gaps plus speed perturb; frequency
masking is feature-only. RIR and MUSAN are applied on waveforms as in training.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio
import tyro
from loguru import logger

from SpeechToText.augmentation import AudioAugmentConfig, SpecAugmentConfig, load_audio_bank
from SpeechToText.dataset import FeatureConfig

SAMPLE_RATE = 16_000


@dataclass(slots=True)
class PreviewConfig:
    output_dir: str = "results/augment_previews"
    val_manifest: str = "data/manifests/final/val_final.jsonl"
    noise_bank: str = "data/augment/noise_bank.pt"
    rir_bank: str = "data/augment/rir_bank.pt"
    seed: int = 42


DEFAULT_SAMPLES: tuple[tuple[str, int], ...] = (
    ("pl", 0),
    ("pl", 13),
    ("en", 3334),
    ("en", 3336),
)


def _load_manifest_row(manifest: Path, line_index: int) -> dict[str, str | float]:
    with manifest.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            if idx == line_index:
                return json.loads(line)
    raise IndexError(f"Line {line_index} not found in {manifest}")


def _load_mono_audio(path: Path, sample_rate: int = SAMPLE_RATE) -> torch.Tensor:
    wav, sr = torchaudio.load(str(path))
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != sample_rate:
        wav = torchaudio.functional.resample(wav, sr, sample_rate)
    return wav.squeeze(0).contiguous()


def _save_wav(path: Path, wav: torch.Tensor, sample_rate: int = SAMPLE_RATE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    audio = wav.detach().cpu().float().numpy()
    peak = max(float(abs(audio).max()), 1e-8)
    if peak > 1.0:
        audio = audio / peak
    sf.write(str(path), audio, sample_rate)


def _apply_speed(wav: torch.Tensor, factor: float) -> torch.Tensor:
    if factor == 1.0:
        return wav
    new_len = max(1, int(round(wav.numel() / factor)))
    return F.interpolate(wav.view(1, 1, -1), size=new_len, mode="linear", align_corners=False).view(
        -1
    )


def _apply_time_mask_waveform(
    wav: torch.Tensor,
    feat_cfg: FeatureConfig,
    spec_cfg: SpecAugmentConfig,
    *,
    seed: int,
) -> torch.Tensor:
    """Rough listenability proxy for SpecAugment time masks (feature masking → muted gaps)."""
    rng = random.Random(seed)
    hop = int(feat_cfg.sample_rate * feat_cfg.hop_length_ms / 1000.0)
    n_frames = max(1, wav.numel() // hop)
    max_width = max(1, int(n_frames * spec_cfg.time_width_fraction))
    out = wav.clone()
    for _ in range(spec_cfg.time_masks):
        width = rng.randint(0, max_width)
        if width <= 0:
            continue
        start = rng.randint(0, max(1, n_frames - width))
        s_sample = start * hop
        e_sample = min(wav.numel(), (start + width) * hop)
        out[s_sample:e_sample] *= 0.02
    return out


def _apply_rir(wav: torch.Tensor, rir: torch.Tensor) -> torch.Tensor:
    rir = rir.reshape(-1).float()
    convolved = F.conv1d(
        wav.view(1, 1, -1),
        rir.flip(0).view(1, 1, -1),
        padding=rir.numel() - 1,
    ).view(-1)
    return convolved[: wav.numel()]


def _apply_background_noise(wav: torch.Tensor, noise: torch.Tensor, snr_db: float) -> torch.Tensor:
    noise = noise.reshape(-1).float()
    if noise.numel() < wav.numel():
        repeats = (wav.numel() + noise.numel() - 1) // noise.numel()
        noise = noise.repeat(repeats)[: wav.numel()]
    else:
        noise = noise[: wav.numel()]

    signal_power = wav.pow(2).mean()
    noise_power = noise.pow(2).mean()
    if noise_power <= 0:
        return wav
    target_noise_power = signal_power / (10.0 ** (snr_db / 10.0))
    scale = (target_noise_power / noise_power).sqrt()
    return (wav + noise * scale).clamp(-1.0, 1.0)


def _slug(text: str, max_len: int = 40) -> str:
    slug = "".join(ch if ch.isalnum() else "_" for ch in text.lower())
    slug = "_".join(part for part in slug.split("_") if part)
    return slug[:max_len].strip("_") or "sample"


def _process_sample(
    *,
    lang: str,
    row: dict[str, str | float],
    out_dir: Path,
    feat_cfg: FeatureConfig,
    spec_cfg: SpecAugmentConfig,
    rir_bank: tuple[torch.Tensor, ...] | None,
    noise_bank: tuple[torch.Tensor, ...] | None,
    seed: int,
    sample_idx: int,
) -> None:
    audio_path = Path(str(row["audio_filepath"]))
    text = str(row["text"])
    duration = float(row["duration"])
    name = f"{lang}_{sample_idx:02d}_{_slug(text)}"

    wav = _load_mono_audio(audio_path)

    sample_seed = seed + sample_idx * 997
    rng = random.Random(sample_seed)
    speed_factor = 0.98

    rir_idx = rng.randrange(len(rir_bank)) if rir_bank else 0
    noise_idx = rng.randrange(len(noise_bank)) if noise_bank else 0
    snr_db = 10.0

    sample_dir = out_dir / name
    sample_dir.mkdir(parents=True, exist_ok=True)

    _save_wav(sample_dir / "0_clean.wav", wav)

    wav_spec = _apply_speed(wav, speed_factor)
    wav_spec = _apply_time_mask_waveform(wav_spec, feat_cfg, spec_cfg, seed=sample_seed + 1)
    _save_wav(sample_dir / "1_specaug.wav", wav_spec)

    wav_full = _apply_speed(wav, speed_factor)
    if rir_bank:
        wav_full = _apply_rir(wav_full, rir_bank[rir_idx])
    if noise_bank:
        wav_full = _apply_background_noise(wav_full, noise_bank[noise_idx], snr_db=snr_db)
    wav_full = _apply_time_mask_waveform(wav_full, feat_cfg, spec_cfg, seed=sample_seed + 2)
    _save_wav(sample_dir / "2_full.wav", wav_full)

    meta = {
        "language": lang,
        "text": text,
        "duration_sec": duration,
        "source": str(audio_path),
        "speed_factor": speed_factor,
        "rir_index": rir_idx if rir_bank else None,
        "noise_index": noise_idx if noise_bank else None,
        "snr_db": snr_db,
        "files": {
            "0_clean.wav": "Original audio, no augmentation (epochs 0–9 in v3).",
            "1_specaug.wav": (
                f"Speed x{speed_factor} + time-mask gaps approximating SpecAugment "
                "(epochs 10–19; freq masking is mel-only and not reproduced here)."
            ),
            "2_full.wav": (
                f"Speed x{speed_factor} + RIR + MUSAN at {snr_db:.0f} dB SNR + time masks "
                "(epochs 20+ heavy aug + SpecAugment phase)."
            ),
        },
    }
    (sample_dir / "metadata.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("Wrote previews to {}", sample_dir)


def main(config: PreviewConfig) -> None:
    torch.manual_seed(config.seed)
    feat_cfg = FeatureConfig()
    spec_cfg = SpecAugmentConfig()
    audio_cfg = AudioAugmentConfig()

    noise_bank = load_audio_bank(config.noise_bank, sample_rate=SAMPLE_RATE)
    rir_bank = load_audio_bank(
        config.rir_bank,
        sample_rate=SAMPLE_RATE,
        min_len_sec=0.05,
        normalize_rirs=True,
        max_rir_len_sec=0.5,
        max_bank_items=4096,
    )
    if not noise_bank:
        logger.warning("Noise bank missing; full previews will skip MUSAN.")
    if not rir_bank:
        logger.warning("RIR bank missing; full previews will skip RIR.")

    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = Path(config.val_manifest)

    readme = out_dir / "README.txt"
    readme.write_text(
        "\n".join(
            [
                "Augmentation preview samples",
                "==========================",
                "",
                "Four clips: 2× PL + 2× EN from val set. Each folder has:",
                "",
                "  0_clean.wav     — no augmentation",
                "  1_specaug.wav   — speed perturb + time gaps (SpecAugment phase proxy)",
                "  2_full.wav      — speed + RIR + MUSAN noise + time gaps",
                "",
                "Note: real SpecAugment also masks mel frequency bands (not audible as",
                "waveform edits). Time gaps are a rough listenability proxy.",
                "",
                f"Training probs when active: bg_noise={audio_cfg.bg_noise_prob}, "
                f"rir={audio_cfg.rir_prob} (forced here for demo).",
                "",
            ]
        ),
        encoding="utf-8",
    )

    for sample_idx, (lang, line_index) in enumerate(DEFAULT_SAMPLES):
        row = _load_manifest_row(manifest, line_index)
        if str(row.get("language", "")) != lang:
            logger.warning(
                "Manifest line {} language mismatch (expected {}, got {})",
                line_index,
                lang,
                row.get("language"),
            )
        _process_sample(
            lang=lang,
            row=row,
            out_dir=out_dir,
            feat_cfg=feat_cfg,
            spec_cfg=spec_cfg,
            rir_bank=rir_bank,
            noise_bank=noise_bank,
            seed=config.seed,
            sample_idx=sample_idx,
        )

    logger.info("Done. Open {}", out_dir.resolve())


if __name__ == "__main__":
    main(tyro.cli(PreviewConfig))

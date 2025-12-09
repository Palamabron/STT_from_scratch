from __future__ import annotations

import torch
import torchaudio

from SpeechToText.models.ctc_attention.train import DataConfig

_RESAMPLERS: dict[tuple[int, int], torchaudio.transforms.Resample] = {}


def build_feature_transforms(
    cfg: DataConfig,
) -> tuple[torchaudio.transforms.MelSpectrogram, torchaudio.transforms.AmplitudeToDB]:
    """Return MelSpectrogram and AmplitudeToDB transforms."""
    win_length = int(cfg.sample_rate * cfg.win_length_ms / 1000.0)
    hop_length = int(cfg.sample_rate * cfg.hop_length_ms / 1000.0)

    mel_spec = torchaudio.transforms.MelSpectrogram(
        sample_rate=cfg.sample_rate,
        n_fft=cfg.n_fft,
        win_length=win_length,
        hop_length=hop_length,
        n_mels=cfg.n_mels,
        power=2.0,
        center=True,
        normalized=False,
    )
    amplitude_to_db = torchaudio.transforms.AmplitudeToDB(top_db=80.0)
    return mel_spec, amplitude_to_db


def get_or_create_resampler(
    orig_freq: int,
    new_freq: int,
) -> torchaudio.transforms.Resample:
    """Return cached or new Resample transform."""
    key = (orig_freq, new_freq)
    if key not in _RESAMPLERS:
        _RESAMPLERS[key] = torchaudio.transforms.Resample(
            orig_freq=orig_freq,
            new_freq=new_freq,
        )
    return _RESAMPLERS[key]


def extract_features(
    audio_path: str,
    data_cfg: DataConfig,
    mel_spec: torchaudio.transforms.MelSpectrogram,
    amplitude_to_db: torchaudio.transforms.AmplitudeToDB,
    device: torch.device,
) -> tuple[torch.Tensor, int]:
    """Compute log-mel features [T, F] and length for a file."""
    waveform, sample_rate = torchaudio.load(audio_path)

    if waveform.dim() == 2 and waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if sample_rate != data_cfg.sample_rate:
        resampler = get_or_create_resampler(sample_rate, data_cfg.sample_rate)
        waveform = resampler(waveform)

    waveform = waveform.to(device)

    mel = mel_spec(waveform)
    mel = amplitude_to_db(mel)
    mel = mel.transpose(1, 2).squeeze(0)

    if data_cfg.normalize_features:
        mel = (mel - mel.mean(dim=0, keepdim=True)) / (mel.std(dim=0, keepdim=True) + 1e-5)

    feature_length = mel.size(0)
    return mel, feature_length

from __future__ import annotations

import torch
import torchaudio

from SpeechToText.dataset import DataConfig

_RESAMPLERS: dict[tuple[int, int], torchaudio.transforms.Resample] = {}


def build_feature_transforms(
    cfg: DataConfig,
) -> tuple[torchaudio.transforms.MelSpectrogram, torchaudio.transforms.AmplitudeToDB]:
    """Return MelSpectrogram and AmplitudeToDB transforms."""
    feat = cfg.features
    win_length = int(feat.sample_rate * feat.win_length_ms / 1000.0)
    hop_length = int(feat.sample_rate * feat.hop_length_ms / 1000.0)

    mel_spec = torchaudio.transforms.MelSpectrogram(
        sample_rate=feat.sample_rate,
        n_fft=feat.n_fft,
        win_length=win_length,
        hop_length=hop_length,
        n_mels=feat.n_mels,
        power=2.0,
        center=True,
        normalized=False,
    )
    amplitude_to_db = torchaudio.transforms.AmplitudeToDB(top_db=feat.top_db)
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
    data_config: DataConfig,
    mel_spec: torchaudio.transforms.MelSpectrogram,
    amplitude_to_db: torchaudio.transforms.AmplitudeToDB,
    device: torch.device,
) -> tuple[torch.Tensor, int]:
    """Compute log-mel features [T, F] and length for a file."""
    waveform, sample_rate = torchaudio.load(audio_path)

    if waveform.dim() == 2 and waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    target_sr = data_config.features.sample_rate
    if sample_rate != target_sr:
        resampler = get_or_create_resampler(sample_rate, target_sr)
        waveform = resampler(waveform)

    waveform = waveform.to(device)

    mel = mel_spec(waveform)
    mel = amplitude_to_db(mel)
    mel = mel.transpose(1, 2).squeeze(0)

    feature_length = mel.size(0)
    return mel, feature_length

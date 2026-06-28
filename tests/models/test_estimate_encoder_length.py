from __future__ import annotations

import torch

from SpeechToText.dataset import FeatureConfig, estimate_encoder_output_length
from SpeechToText.features import WaveformFeaturizer, mel_frames_from_audio_lengths
from SpeechToText.models.conformer.subsampling import subsample_lengths


def test_mel_frames_from_audio_lengths_matches_featurizer() -> None:
    feat_cfg = FeatureConfig(sample_rate=16_000, hop_length_ms=10.0)
    hop_length = int(feat_cfg.sample_rate * feat_cfg.hop_length_ms / 1000.0)
    audio_lengths = torch.tensor([8000, 16_000, 32_001], dtype=torch.long)

    featurizer = WaveformFeaturizer(feat_cfg)
    audios = torch.zeros(len(audio_lengths), int(audio_lengths.max()), dtype=torch.float32)
    for index, length in enumerate(audio_lengths.tolist()):
        audios[index, :length] = torch.randn(length)

    _, feat_lens = featurizer(audios, audio_lengths)
    expected = mel_frames_from_audio_lengths(audio_lengths, hop_length=hop_length)
    expected = expected.clamp(max=int(feat_lens.max().item()))

    assert feat_lens.tolist() == expected.tolist()


def test_estimate_encoder_output_length_matches_manual_pipeline() -> None:
    duration_sec = 2.0
    sample_rate = 16_000
    hop_length_ms = 10.0
    subsampling_factor = 4

    hop_length = int(sample_rate * hop_length_ms / 1000.0)
    audio_samples = int(duration_sec * sample_rate)
    feat_len = int(
        mel_frames_from_audio_lengths(
            torch.tensor([audio_samples], dtype=torch.long),
            hop_length=hop_length,
        ).item()
    )
    expected = int(subsample_lengths(feat_len, subsampling_factor))

    assert (
        estimate_encoder_output_length(
            duration_sec,
            sample_rate=sample_rate,
            hop_length_ms=hop_length_ms,
            subsampling_factor=subsampling_factor,
        )
        == expected
    )


def test_estimate_encoder_output_length_applies_speed_margin() -> None:
    without_margin = estimate_encoder_output_length(
        2.0,
        sample_rate=16_000,
        hop_length_ms=10.0,
        subsampling_factor=2,
        min_speed_factor=1.0,
    )
    with_margin = estimate_encoder_output_length(
        2.0,
        sample_rate=16_000,
        hop_length_ms=10.0,
        subsampling_factor=2,
        min_speed_factor=0.95,
    )
    assert with_margin <= without_margin

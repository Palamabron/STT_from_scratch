from __future__ import annotations

import torch

from SpeechToText.augmentation import (
    AudioAugmentConfig,
    GPUAudioAugmentation,
    SpecAugment,
    SpecAugmentConfig,
)
from SpeechToText.dataset import FeatureConfig
from SpeechToText.features import WaveformFeaturizer


def test_gpu_augment_all_clean_pass_is_identity() -> None:
    cfg = AudioAugmentConfig(gain_prob=1.0, bg_noise_prob=1.0)
    gpu_aug = GPUAudioAugmentation(cfg, rir_bank=None, noise_bank=None, augment_start_epoch=0)
    gpu_aug.train()
    gpu_aug.set_current_epoch(20)

    audio = torch.randn(2, 1600).clamp(-1.0, 1.0)
    clean_pass = torch.tensor([True, True])
    out = gpu_aug(audio, clean_pass=clean_pass)

    assert torch.allclose(out, audio)


def test_gpu_augment_preserves_clean_rows() -> None:
    cfg = AudioAugmentConfig(
        gain_prob=1.0,
        gain_db_min=6.0,
        gain_db_max=6.0,
        gain_db_mean=6.0,
        gain_db_std=0.0,
    )
    gpu_aug = GPUAudioAugmentation(cfg, rir_bank=None, noise_bank=None, augment_start_epoch=0)
    gpu_aug.train()
    gpu_aug.set_current_epoch(0)

    audio = torch.randn(2, 1600).clamp(-1.0, 1.0)
    clean_pass = torch.tensor([True, False])
    out = gpu_aug(audio, clean_pass=clean_pass)

    assert torch.allclose(out[0], audio[0])
    assert not torch.allclose(out[1], audio[1])


def test_featurizer_skips_spec_augment_for_clean_pass_rows() -> None:
    feat_cfg = FeatureConfig()
    spec = SpecAugment(SpecAugmentConfig(freq_masks=2, time_masks=4), augment_start_epoch=0)
    featurizer = WaveformFeaturizer(feat_cfg, spec_augment=spec, gpu_augment=None)
    baseline = WaveformFeaturizer(feat_cfg, spec_augment=None, gpu_augment=None)
    featurizer.train()
    baseline.train()
    featurizer.set_current_epoch(0)

    torch.manual_seed(0)
    audio = torch.randn(2, 3200)
    audio_lengths = torch.tensor([3200, 3200])
    clean_pass = torch.tensor([True, False])

    feats_ref, _ = baseline(audio, audio_lengths)
    feats_mixed, _ = featurizer(audio, audio_lengths, clean_pass=clean_pass)

    assert torch.allclose(feats_mixed[0], feats_ref[0])
    assert not torch.allclose(feats_mixed[1], feats_ref[1])


def test_featurizer_ignores_clean_pass_before_augment_start_epoch() -> None:
    feat_cfg = FeatureConfig()
    spec = SpecAugment(SpecAugmentConfig(freq_masks=2, time_masks=4), augment_start_epoch=5)
    featurizer = WaveformFeaturizer(feat_cfg, spec_augment=spec, gpu_augment=None)
    featurizer.train()
    featurizer.set_current_epoch(0)

    audio = torch.randn(2, 3200)
    audio_lengths = torch.tensor([3200, 3200])
    clean_pass = torch.tensor([True, True])

    feats_a, _ = featurizer(audio, audio_lengths, clean_pass=None)
    feats_b, _ = featurizer(audio, audio_lengths, clean_pass=clean_pass)

    assert torch.allclose(feats_a, feats_b)

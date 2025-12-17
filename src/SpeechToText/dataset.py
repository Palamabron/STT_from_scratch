from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.multiprocessing as mp
import torchaudio
import torchaudio.transforms as AT
from loguru import logger
from sentencepiece import SentencePieceProcessor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from .augmentation import AudioAugmentation, AudioAugmentConfig, SpecAugment, SpecAugmentConfig


@dataclass
class DataConfig:
    train_manifest: str
    val_manifest: str
    tokenizer_model: str
    sample_rate: int = 16_000
    n_fft: int = 512
    win_length_ms: float = 25.0
    hop_length_ms: float = 10.0
    n_mels: int = 80
    train_batch_size: int = 32
    val_batch_size: int = 64
    max_duration: float = 15.0
    pin_memory: bool = True
    normalize_features: bool = True
    speed_factors: list[float] = field(default_factory=lambda: [0.9, 1.0, 1.0, 1.0, 1.1])
    num_workers: int = 8


class ManifestDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        manifest_path: str,
        sp: SentencePieceProcessor,
        config: DataConfig,
        split: str = "train",
        spec_augment: SpecAugment | None = None,
        audio_augment: AudioAugmentation | None = None,
    ) -> None:
        self.entries: list[dict[str, Any]] = []
        self.sp = sp
        self.config = config
        self.split = split
        self.spec_augment = spec_augment
        self.audio_augment = audio_augment

        self._epoch = mp.Value("i", 0, lock=False)
        self._worker_epoch_cache = -1

        mpth = Path(manifest_path)
        if not mpth.exists():
            raise FileNotFoundError(f"Manifest not found: {mpth}")

        with mpth.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                dur_raw = obj.get("duration")
                dur = float(dur_raw) if dur_raw is not None else 0.0

                if dur <= config.max_duration:
                    self.entries.append(obj)

        logger.info(f"Loaded {len(self.entries)} entries from {manifest_path} for split={split}")

        win_length = int(config.sample_rate * config.win_length_ms / 1000.0)
        hop_length = int(config.sample_rate * config.hop_length_ms / 1000.0)

        self.mel_spec = AT.MelSpectrogram(
            sample_rate=config.sample_rate,
            n_fft=config.n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=config.n_mels,
            power=2.0,
            center=True,
            normalized=False,
        )
        self.amplitude_to_db = AT.AmplitudeToDB(top_db=80.0)

        self._resamplers: dict[int, AT.Resample] = {}

    def set_current_epoch(self, epoch: int) -> None:
        self._epoch.value = int(epoch)

    def _maybe_sync_epoch(self) -> None:
        e = int(self._epoch.value)
        if e == self._worker_epoch_cache:
            return
        self._worker_epoch_cache = e
        if self.spec_augment is not None:
            self.spec_augment.set_current_epoch(e)
        if self.audio_augment is not None:
            self.audio_augment.set_current_epoch(e)

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        self._maybe_sync_epoch()

        item = self.entries[idx]
        audio_path = item["audio_filepath"]
        text = item["text"]
        lang = item.get("language", "unknown")

        wav, sr = torchaudio.load(audio_path)
        if wav.dim() == 2 and wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)

        if sr != self.config.sample_rate:
            resampler = self._resamplers.get(sr)
            if resampler is None:
                resampler = AT.Resample(orig_freq=sr, new_freq=self.config.sample_rate)
                self._resamplers[sr] = resampler
            wav = resampler(wav)

        wav_1d = wav.squeeze(0)

        if self.audio_augment is not None and self.split == "train":
            wav_1d = self.audio_augment(wav_1d)

        mel = self.mel_spec(wav_1d.unsqueeze(0))
        mel = self.amplitude_to_db(mel)
        mel = mel.transpose(1, 2).squeeze(0)

        if self.config.normalize_features:
            mean = mel.mean(dim=0, keepdim=True)
            std = mel.std(dim=0, keepdim=True).clamp_min(1e-5)
            mel = (mel - mean) / std

        if self.spec_augment is not None and self.split == "train":
            mel = self.spec_augment(mel.unsqueeze(0)).squeeze(0)

        feat_len = int(mel.size(0))

        ids = self.sp.encode(text, out_type=int)
        if not ids:
            logger.warning(f"Empty token sequence for text: {text!r}")
        targets = torch.tensor([i + 1 for i in ids], dtype=torch.long)
        target_len = int(targets.numel())

        return {
            "features": mel,
            "feature_length": feat_len,
            "targets": targets,
            "target_length": target_len,
            "text": text,
            "language": lang,
        }


def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    batch.sort(key=lambda b: b["feature_length"], reverse=True)

    feats_list = [b["features"] for b in batch]
    feat_lengths = torch.tensor([b["feature_length"] for b in batch], dtype=torch.long)
    feats = pad_sequence(feats_list, batch_first=True)

    target_lengths = torch.tensor([b["target_length"] for b in batch], dtype=torch.long)
    targets = (
        batch[0]["targets"] if len(batch) == 1 else torch.cat([b["targets"] for b in batch], dim=0)
    )

    return {
        "features": feats,
        "feature_lengths": feat_lengths,
        "targets": targets,
        "target_lengths": target_lengths,
        "text": [b["text"] for b in batch],
        "language": [b.get("language", "unknown") for b in batch],
    }


def create_dataloaders(
    data_config: DataConfig,
    spec_config: SpecAugmentConfig,
    audio_config: AudioAugmentConfig,
    augment_start_epoch: int = 3,
) -> tuple[DataLoader, DataLoader, SentencePieceProcessor]:
    sp = SentencePieceProcessor()
    sp.load(str(data_config.tokenizer_model))

    spec_aug = SpecAugment(spec_config, augment_start_epoch=augment_start_epoch)
    audio_aug = AudioAugmentation(
        audio_config, sample_rate=data_config.sample_rate, augment_start_epoch=augment_start_epoch
    )

    train_ds = ManifestDataset(
        manifest_path=data_config.train_manifest,
        sp=sp,
        config=data_config,
        split="train",
        spec_augment=spec_aug,
        audio_augment=audio_aug,
    )
    val_ds = ManifestDataset(
        manifest_path=data_config.val_manifest,
        sp=sp,
        config=data_config,
        split="val",
        spec_augment=None,
        audio_augment=None,
    )

    dl_extra: dict[str, Any] = {}
    if data_config.num_workers > 0:
        dl_extra["persistent_workers"] = True
        dl_extra["prefetch_factor"] = 4

    train_loader = DataLoader(
        train_ds,
        batch_size=data_config.train_batch_size,
        shuffle=True,
        num_workers=data_config.num_workers,
        pin_memory=data_config.pin_memory,
        collate_fn=collate_fn,
        **dl_extra,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=data_config.val_batch_size,
        shuffle=False,
        num_workers=data_config.num_workers,
        pin_memory=data_config.pin_memory,
        collate_fn=collate_fn,
        **dl_extra,
    )
    return train_loader, val_loader, sp

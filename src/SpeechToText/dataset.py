from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torchaudio
from loguru import logger
from sentencepiece import SentencePieceProcessor
from torch.utils.data import DataLoader, Dataset

from .augmentation import (
    AudioAugmentation,
    AudioAugmentConfig,
    SpecAugment,
    SpecAugmentConfig,
)


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
    min_duration: float = 0.1
    num_workers: int = 4
    pin_memory: bool = True
    normalize_features: bool = True


class ManifestDataset(Dataset):
    def __init__(
        self,
        manifest_path: str,
        sp: SentencePieceProcessor,
        cfg: DataConfig,
        split: str = "train",
        spec_augment: SpecAugment | None = None,
        audio_augment: AudioAugmentation | None = None,
    ) -> None:
        self.entries: list[dict[str, Any]] = []
        self.sp = sp
        self.cfg = cfg
        self.split = split
        self.spec_augment = spec_augment
        self.audio_augment = audio_augment
        self.current_epoch = 0

        manifest_path = Path(manifest_path)
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")

        with manifest_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                dur = float(obj.get("duration", 0.0))
                if dur < cfg.min_duration or dur > cfg.max_duration:
                    continue
                self.entries.append(obj)

        logger.info(f"Loaded {len(self.entries)} entries from {manifest_path} for split={split}")

        win_length = int(cfg.sample_rate * cfg.win_length_ms / 1000.0)
        hop_length = int(cfg.sample_rate * cfg.hop_length_ms / 1000.0)

        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=cfg.sample_rate,
            n_fft=cfg.n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=cfg.n_mels,
            power=2.0,  # klasyczny power spec
            center=True,
            normalized=False,
        )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB(top_db=80.0)

    def set_current_epoch(self, epoch: int) -> None:
        self.current_epoch = epoch
        if self.spec_augment is not None:
            self.spec_augment.set_current_epoch(epoch)
        if self.audio_augment is not None:
            self.audio_augment.set_current_epoch(epoch)

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self.entries[idx]
        audio_path = item["audio_filepath"]
        text = item["text"]
        lang = item.get("language", "unknown")

        wav, sr = torchaudio.load(audio_path)  # (C, T)
        if wav.dim() == 2 and wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)

        if sr != self.cfg.sample_rate:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sr,
                new_freq=self.cfg.sample_rate,
            )
            wav = resampler(wav)

        wav = wav.squeeze(0)  # (T,)

        if self.audio_augment is not None and self.split == "train":
            wav = self.audio_augment(wav)

        wav = wav.unsqueeze(0)  # (1, T)
        mel = self.mel_spec(wav)  # (n_mels, T')
        mel = self.amplitude_to_db(mel)  # (n_mels, T')
        mel = mel.transpose(1, 2).squeeze(0)  # (T', n_mels)

        if self.cfg.normalize_features:
            mel = (mel - mel.mean(dim=0, keepdim=True)) / (mel.std(dim=0, keepdim=True) + 1e-5)

        if self.spec_augment is not None and self.split == "train":
            mel = self.spec_augment(mel.unsqueeze(0)).squeeze(0)

        feat_len = mel.size(0)

        ids = self.sp.encode(text, out_type=int)
        if len(ids) == 0:
            logger.warning(f"Empty token sequence for text: {text!r}")

        # 0 = blank, więc subwordy od 1 w górę
        ids = [i + 1 for i in ids]
        targets = torch.tensor(ids, dtype=torch.long)
        target_len = targets.size(0)

        return {
            "features": mel,
            "feature_length": feat_len,
            "targets": targets,
            "target_length": target_len,
            "text": text,
            "language": lang,
        }


def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    # sortuj po długości (pomaga przy CTC, ale nie jest konieczne)
    batch = sorted(batch, key=lambda b: b["feature_length"], reverse=True)

    feat_lengths = torch.tensor(
        [b["feature_length"] for b in batch],
        dtype=torch.long,
    )
    max_feat_len = int(feat_lengths.max().item())

    feats = []
    for b in batch:
        f = b["features"]  # (T, F)
        pad_f = torch.zeros(max_feat_len, f.size(1), dtype=f.dtype)
        pad_f[: f.size(0)] = f
        feats.append(pad_f)

    feats = torch.stack(feats, dim=0)  # (B, T_max, F)

    target_lengths = torch.tensor(
        [b["target_length"] for b in batch],
        dtype=torch.long,
    )

    if len(batch) == 1:
        targets = batch[0]["targets"]
    else:
        targets = torch.cat([b["targets"] for b in batch], dim=0)  # (sum_L,)

    texts = [b["text"] for b in batch]
    langs = [b.get("language", "unknown") for b in batch]

    return {
        "features": feats,
        "feature_lengths": feat_lengths,
        "targets": targets,
        "target_lengths": target_lengths,
        "text": texts,
        "language": langs,
    }


def create_dataloaders(
    data_cfg: DataConfig,
    spec_cfg: SpecAugmentConfig,
    audio_cfg: AudioAugmentConfig,
    augment_start_epoch: int = 3,
):
    sp = SentencePieceProcessor()
    sp.load(str(data_cfg.tokenizer_model))

    # Na razie AUGMENTACJE WYŁĄCZONE – włączysz jak już będzie się uczyć.
    spec_aug = None
    audio_aug = None

    train_ds = ManifestDataset(
        manifest_path=data_cfg.train_manifest,
        sp=sp,
        cfg=data_cfg,
        split="train",
        spec_augment=spec_aug,
        audio_augment=audio_aug,
    )

    val_ds = ManifestDataset(
        manifest_path=data_cfg.val_manifest,
        sp=sp,
        cfg=data_cfg,
        split="val",
        spec_augment=None,
        audio_augment=None,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=data_cfg.train_batch_size,
        shuffle=True,
        num_workers=data_cfg.num_workers,
        pin_memory=data_cfg.pin_memory,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=data_cfg.val_batch_size,
        shuffle=False,
        num_workers=data_cfg.num_workers,
        pin_memory=data_cfg.pin_memory,
        collate_fn=collate_fn,
    )

    return train_loader, val_loader, sp

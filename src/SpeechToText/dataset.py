from __future__ import annotations

import json
import os
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, cast

import torch
import torch.multiprocessing as mp
import torchaudio
from loguru import logger
from sentencepiece import SentencePieceProcessor
from torch.utils.data import DataLoader, Dataset, Sampler

from .augmentation import AudioAugmentation, AudioAugmentConfig

CTC_BLANK_ID = 0


@dataclass(slots=True)
class ManifestPaths:
    """Paths to train and validation JSONL manifests."""

    train: str = "data/manifests/final/train_final.jsonl"
    val: str = "data/manifests/final/val_final.jsonl"


@dataclass(slots=True)
class FeatureConfig:
    """Log-mel feature extraction settings."""

    sample_rate: int = 16_000
    n_fft: int = 512
    win_length_ms: float = 25.0
    hop_length_ms: float = 10.0
    n_mels: int = 80
    top_db: float = 80.0


@dataclass(slots=True)
class LoaderConfig:
    """PyTorch DataLoader and batching settings."""

    train_batch_size: int = 32
    val_batch_size: int = 64
    num_workers: int = 8
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 4
    train_max_batch_duration: float | None = 240.0
    train_max_batch_size: int = 64
    bucket_size: int = 4096
    shuffle: bool = True
    seed: int = 42
    multiprocessing_context: str | None = "fork"
    cache_audio: bool = False
    stratify_by_language: bool = False


@dataclass(slots=True)
class FilterConfig:
    """Manifest entry filtering rules."""

    max_duration: float = 16.0
    on_bad_duration: str = "drop"
    subsampling_factor: int = 8
    min_speed_factor: float = 0.95


@dataclass(slots=True)
class DataConfig:
    """Top-level dataset configuration."""

    manifests: ManifestPaths = field(default_factory=ManifestPaths)
    tokenizer_model: str = "models/spm_unigram_4k_trainval.model"
    features: FeatureConfig = field(default_factory=FeatureConfig)
    loader: LoaderConfig = field(default_factory=LoaderConfig)
    filter: FilterConfig = field(default_factory=FilterConfig)


def _parse_duration(value: Any) -> float | None:
    """Parse a positive duration in seconds from a manifest field."""
    if value is None:
        return None
    try:
        duration = float(value)
    except (TypeError, ValueError):
        return None
    if duration <= 0.0:
        return None
    return duration


def _shift_token_ids(token_ids: list[int]) -> list[int]:
    """Reserve ``CTC_BLANK_ID`` and map SentencePiece id ``i`` to ``i + 1``."""
    return [token_id + 1 for token_id in token_ids]


def estimate_encoder_output_length(
    duration_sec: float,
    *,
    sample_rate: int,
    hop_length_ms: float,
    subsampling_factor: int,
    min_speed_factor: float = 1.0,
) -> int:
    """Estimate Conformer encoder output length from manifest audio duration."""
    from SpeechToText.models.conformer.subsampling import subsample_lengths

    effective_duration = max(0.0, float(duration_sec) * float(min_speed_factor))
    audio_samples = int(effective_duration * sample_rate)
    hop_length = max(1, int(sample_rate * hop_length_ms / 1000.0))
    feat_len = (audio_samples // hop_length) + 1
    return int(subsample_lengths(feat_len, subsampling_factor))


@torch.jit.script
def _pcm_to_float32(wav_1d: torch.Tensor) -> torch.Tensor:
    """Convert common PCM dtypes to normalized float32 mono samples."""
    if wav_1d.dtype == torch.float32:
        return wav_1d
    if wav_1d.dtype == torch.float64:
        return wav_1d.to(torch.float32)
    if wav_1d.dtype == torch.int16:
        return wav_1d.to(torch.float32).mul_(1.0 / 32768.0)
    if wav_1d.dtype == torch.int32:
        return wav_1d.to(torch.float32).mul_(1.0 / 2147483648.0)
    if wav_1d.dtype == torch.uint8:
        return wav_1d.to(torch.float32).add_(-128.0).mul_(1.0 / 128.0)
    return wav_1d.to(torch.float32)


class ManifestDataset(Dataset[dict[str, Any]]):
    """JSONL manifest dataset that yields raw waveform batches."""

    def __init__(
        self,
        manifest_path: str,
        sp: SentencePieceProcessor,
        config: DataConfig,
        split: str = "train",
        audio_augment: AudioAugmentation | None = None,
    ) -> None:
        self.sp: Final[SentencePieceProcessor] = sp
        self.config: Final[DataConfig] = config
        self.split: Final[str] = split
        self.audio_augment: Final[AudioAugmentation | None] = audio_augment

        self._epoch = mp.Value("i", 0, lock=False)
        self._worker_epoch_cache: int = -1
        self.cache_enabled = config.loader.cache_audio
        self.audio_cache: dict[int, torch.Tensor] = {}

        manifest_file = Path(manifest_path)
        if not manifest_file.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_file}")

        dropped_bad_dur = 0
        dropped_too_long = 0
        dropped_missing = 0
        dropped_ctc_too_long = 0

        feat_cfg = config.features
        filter_cfg = config.filter

        self.audio_paths: list[str] = []
        self.texts: list[str] = []
        self.langs: list[str] = []
        self.datasets: list[str] = []
        self.targets: list[torch.Tensor] = []
        self.target_lengths: list[int] = []
        self.durations: list[float] = []

        with manifest_file.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)

                duration = _parse_duration(entry.get("duration"))
                if duration is None:
                    if self.config.filter.on_bad_duration == "drop":
                        dropped_bad_dur += 1
                    continue
                if duration > self.config.filter.max_duration:
                    dropped_too_long += 1
                    continue

                audio_path = entry.get("audio_filepath")
                text = (entry.get("text") or "").strip()
                lang = entry.get("language", "unknown")
                dataset_name = entry.get("dataset", "unknown")

                if not isinstance(audio_path, str) or not audio_path:
                    dropped_missing += 1
                    continue
                if not text:
                    dropped_missing += 1
                    continue
                if not isinstance(lang, str) or not lang:
                    lang = "unknown"
                if not isinstance(dataset_name, str) or not dataset_name:
                    dataset_name = "unknown"

                token_ids = self.sp.encode(text, out_type=int)
                if not token_ids:
                    dropped_missing += 1
                    continue

                shifted_ids = _shift_token_ids(token_ids)
                target_len = len(shifted_ids)
                encoder_len = estimate_encoder_output_length(
                    duration,
                    sample_rate=feat_cfg.sample_rate,
                    hop_length_ms=feat_cfg.hop_length_ms,
                    subsampling_factor=filter_cfg.subsampling_factor,
                    min_speed_factor=filter_cfg.min_speed_factor if split == "train" else 1.0,
                )
                if encoder_len < target_len:
                    dropped_ctc_too_long += 1
                    continue

                self.audio_paths.append(audio_path)
                self.texts.append(text)
                self.langs.append(lang)
                self.datasets.append(dataset_name)
                self.targets.append(torch.tensor(shifted_ids, dtype=torch.long))
                self.target_lengths.append(target_len)
                self.durations.append(float(duration))

        logger.info(
            "Loaded {} entries from {} for split={} "
            "(dropped_bad_duration={}, dropped_too_long={}, dropped_missing={}, "
            "dropped_ctc_too_long={})",
            len(self.audio_paths),
            manifest_path,
            split,
            dropped_bad_dur,
            dropped_too_long,
            dropped_missing,
            dropped_ctc_too_long,
        )

    def set_current_epoch(self, epoch: int) -> None:
        """Synchronize augmentation modules with the trainer epoch."""
        self._epoch.value = int(epoch)

    def _maybe_sync_epoch(self) -> None:
        epoch = int(self._epoch.value)
        if epoch == self._worker_epoch_cache:
            return
        self._worker_epoch_cache = epoch
        if self.audio_augment is not None:
            self.audio_augment.set_current_epoch(epoch)

    def _load_waveform(self, idx: int) -> torch.Tensor:
        if self.cache_enabled and idx in self.audio_cache:
            return self.audio_cache[idx]

        wav, sample_rate = torchaudio.load(self.audio_paths[idx])
        if wav.dim() == 2 and wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)

        target_sr = self.config.features.sample_rate
        if sample_rate != target_sr:
            wav = torchaudio.functional.resample(wav, sample_rate, target_sr)

        waveform = _pcm_to_float32(wav.squeeze(0)).contiguous()
        if self.cache_enabled:
            self.audio_cache[idx] = waveform
        return cast(torch.Tensor, waveform)

    def __len__(self) -> int:
        return len(self.audio_paths)

    @torch.inference_mode()
    def __getitem__(self, idx: int) -> dict[str, Any]:
        self._maybe_sync_epoch()

        targets = self.targets[idx]
        target_len = self.target_lengths[idx]
        waveform = self._load_waveform(idx)

        if self.audio_augment is not None and self.split == "train":
            waveform = self.audio_augment(waveform)

        sample: dict[str, Any] = {
            "audio": waveform,
            "audio_length": int(waveform.numel()),
            "targets": targets,
            "target_length": int(target_len),
        }

        if self.split != "train":
            sample["text"] = self.texts[idx]
            sample["language"] = self.langs[idx]
            sample["dataset"] = self.datasets[idx]
            sample["duration"] = float(self.durations[idx])

        return sample


def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Pad variable-length waveforms and concatenate token targets."""
    batch_size = len(batch)

    lengths = torch.empty((batch_size,), dtype=torch.long)
    for index, item in enumerate(batch):
        lengths[index] = int(item["audio_length"])
    max_len = int(lengths.max().item())

    audios = torch.zeros((batch_size, max_len), dtype=torch.float32)
    for index, item in enumerate(batch):
        waveform = item["audio"]
        waveform_len = int(waveform.numel())
        audios[index, :waveform_len].copy_(waveform)

    target_lengths = torch.empty((batch_size,), dtype=torch.long)
    for index, item in enumerate(batch):
        target_lengths[index] = int(item["target_length"])
    targets = torch.cat([item["targets"] for item in batch], dim=0)

    collated: dict[str, Any] = {
        "audio": audios,
        "audio_length": lengths,
        "targets": targets,
        "target_length": target_lengths,
    }

    if "text" in batch[0]:
        collated["text"] = [item["text"] for item in batch]
        collated["language"] = [item["language"] for item in batch]
        collated["dataset"] = [item["dataset"] for item in batch]
        collated["duration"] = torch.tensor([item["duration"] for item in batch], dtype=torch.float)

    return collated


class DurationBatchSampler(Sampler[list[int]]):
    """Bucketed sampler that limits total audio duration per batch."""

    def __init__(
        self,
        durations: list[float],
        max_batch_duration: float,
        max_batch_size: int,
        bucket_size: int,
        shuffle: bool = True,
        seed: int = 42,
        min_batch_size: int = 1,
        languages: list[str] | None = None,
        stratify_by_language: bool = False,
    ) -> None:
        self.durations: Final[list[float]] = durations
        self.languages: Final[list[str] | None] = languages
        self.stratify_by_language: Final[bool] = bool(stratify_by_language)
        self.max_batch_duration: Final[float] = float(max_batch_duration)
        self.max_batch_size: Final[int] = int(max_batch_size)
        self.bucket_size: Final[int] = int(bucket_size)
        self.shuffle: Final[bool] = bool(shuffle)
        self.seed: Final[int] = int(seed)
        self.min_batch_size: Final[int] = int(min_batch_size)

        if self.max_batch_duration <= 0.0:
            raise ValueError("max_batch_duration must be positive")
        if self.max_batch_size <= 0:
            raise ValueError("max_batch_size must be positive")
        if self.bucket_size <= 0:
            raise ValueError("bucket_size must be positive")
        if self.min_batch_size <= 0:
            raise ValueError("min_batch_size must be positive")

        self._num_samples: Final[int] = len(durations)
        self._epoch: int = 0

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def _yield_batches(self, indices: list[int]) -> Iterator[list[int]]:
        for start in range(0, len(indices), self.bucket_size):
            bucket = indices[start : start + self.bucket_size]
            bucket.sort(key=lambda idx: self.durations[idx])

            batch: list[int] = []
            batch_duration = 0.0

            for idx in bucket:
                duration = float(self.durations[idx])
                if not batch:
                    batch = [idx]
                    batch_duration = duration
                    continue

                exceeds_duration = batch_duration + duration > self.max_batch_duration
                exceeds_size = len(batch) >= self.max_batch_size
                if exceeds_duration or exceeds_size:
                    if len(batch) >= self.min_batch_size:
                        yield batch
                    batch = [idx]
                    batch_duration = duration
                else:
                    batch.append(idx)
                    batch_duration += duration

            if batch and len(batch) >= self.min_batch_size:
                yield batch

    def __iter__(self) -> Iterator[list[int]]:
        generator = torch.Generator()
        generator.manual_seed(self.seed + self._epoch if self.shuffle else self.seed)

        if self.stratify_by_language and self.languages is not None:
            by_lang: dict[str, list[int]] = {}
            for index in range(self._num_samples):
                language = self.languages[index]
                by_lang.setdefault(language, []).append(index)

            batches: list[list[int]] = []
            for lang_indices in by_lang.values():
                lang_order = torch.randperm(len(lang_indices), generator=generator).tolist()
                shuffled = [lang_indices[i] for i in lang_order]
                batches.extend(list(self._yield_batches(shuffled)))

            if self.shuffle:
                batch_order = torch.randperm(len(batches), generator=generator).tolist()
                for batch_index in batch_order:
                    yield batches[batch_index]
            else:
                yield from batches
            return

        indices = torch.randperm(self._num_samples, generator=generator).tolist()
        yield from self._yield_batches(indices)

    def _count_batches(self, indices: list[int]) -> int:
        return sum(1 for _ in self._yield_batches(indices))

    def __len__(self) -> int:
        if self.stratify_by_language and self.languages is not None:
            by_lang: dict[str, list[int]] = {}
            for index in range(self._num_samples):
                language = self.languages[index]
                by_lang.setdefault(language, []).append(index)
            return max(
                1, sum(self._count_batches(lang_indices) for lang_indices in by_lang.values())
            )

        indices = list(range(self._num_samples))
        return max(1, self._count_batches(indices))


def _worker_init_fn(worker_id: int) -> None:
    """Limit per-worker thread usage for stable multiprocessing."""
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def create_dataloaders(
    data_config: DataConfig,
    audio_config: AudioAugmentConfig,
    audio_augment_start_epoch: int = 3,
) -> tuple[DataLoader, DataLoader, SentencePieceProcessor]:
    """Build train/validation dataloaders and load the SentencePiece model."""
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    sp = SentencePieceProcessor()
    sp.load(str(data_config.tokenizer_model))

    audio_aug = AudioAugmentation(
        audio_config,
        sample_rate=data_config.features.sample_rate,
        augment_start_epoch=audio_augment_start_epoch,
    )

    train_ds = ManifestDataset(
        manifest_path=data_config.manifests.train,
        sp=sp,
        config=data_config,
        split="train",
        audio_augment=audio_aug,
    )
    val_ds = ManifestDataset(
        manifest_path=data_config.manifests.val,
        sp=sp,
        config=data_config,
        split="val",
        audio_augment=None,
    )

    loader_kwargs: dict[str, Any] = {}
    if data_config.loader.num_workers > 0:
        loader_kwargs["persistent_workers"] = bool(data_config.loader.persistent_workers)
        loader_kwargs["prefetch_factor"] = int(data_config.loader.prefetch_factor)
        loader_kwargs["worker_init_fn"] = _worker_init_fn

    mp_ctx = data_config.loader.multiprocessing_context
    if mp_ctx:
        loader_kwargs["multiprocessing_context"] = mp_ctx

    train_max_duration = data_config.loader.train_max_batch_duration
    if train_max_duration is not None and train_max_duration > 0:
        sampler = DurationBatchSampler(
            train_ds.durations,
            max_batch_duration=float(train_max_duration),
            max_batch_size=int(data_config.loader.train_max_batch_size),
            bucket_size=int(data_config.loader.bucket_size),
            shuffle=bool(data_config.loader.shuffle),
            seed=int(data_config.loader.seed),
            min_batch_size=1,
            languages=train_ds.langs,
            stratify_by_language=bool(data_config.loader.stratify_by_language),
        )
        train_loader = DataLoader(
            train_ds,
            batch_sampler=sampler,
            num_workers=data_config.loader.num_workers,
            pin_memory=data_config.loader.pin_memory,
            collate_fn=collate_fn,
            **loader_kwargs,
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=data_config.loader.train_batch_size,
            shuffle=True,
            num_workers=data_config.loader.num_workers,
            pin_memory=data_config.loader.pin_memory,
            collate_fn=collate_fn,
            drop_last=True,
            **loader_kwargs,
        )

    val_loader = DataLoader(
        val_ds,
        batch_size=data_config.loader.val_batch_size,
        shuffle=False,
        num_workers=data_config.loader.num_workers,
        pin_memory=data_config.loader.pin_memory,
        collate_fn=collate_fn,
        **loader_kwargs,
    )

    return train_loader, val_loader, sp

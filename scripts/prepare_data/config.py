from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Paths:
    root_dir: Path = Path(".")
    data_dir: Path = Path("data")
    audio_dir: Path = Path("data/audio")
    manifests_dir: Path = Path("data/manifests")
    individual_manifests_dir: Path = Path("data/manifests/individual")
    final_train_manifest: Path = Path("data/manifests/final/train_final.jsonl")
    final_val_manifest: Path = Path("data/manifests/final/val_final.jsonl")
    final_test_manifest: Path = Path("data/manifests/final/test_final.jsonl")
    final_manifests_dir: Path = Path("data/manifests/final")


@dataclass
class Run:
    target_sr: int = 16000
    skip_existing: bool = True
    do_train: bool = True
    do_val: bool = True
    shuffle_seed: int = 42
    lowercase_text: bool = True
    normalize_peak: bool = True
    sample_per_dataset: int | None = None
    max_failures: int = 10000
    num_workers: int | None = None
    fetch_shards: int | None = None


@dataclass
class HuggingFace:
    token_env: str = "HF_TOKEN"


@dataclass
class DatasetSpec:
    name: str
    hf_id: str
    split: str
    lang: str
    samples: int
    text_col: str
    audio_col: str = "audio"
    config_name: str | None = None
    use_streaming: bool = True
    start_offset: int = 0


@dataclass
class Datasets:
    train: list[DatasetSpec] = field(default_factory=list)
    val: list[DatasetSpec] = field(default_factory=list)


@dataclass
class AppConfig:
    paths: Paths = field(default_factory=Paths)
    run: Run = field(default_factory=Run)
    hf: HuggingFace = field(default_factory=HuggingFace)
    datasets: Datasets = field(default_factory=Datasets)
    max_train_hours: float | None = None

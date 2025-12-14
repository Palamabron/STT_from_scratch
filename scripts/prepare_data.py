from __future__ import annotations

import argparse
import io
import json
import os
from pathlib import Path
from typing import Any, Optional

import datasets
import soundfile as sf
import torch
import torchaudio
import tyro
from loguru import logger
from tqdm import tqdm

from data_config import AppConfig, DatasetSpec


def load_yaml_config(config_path: Path) -> AppConfig:
    import yaml

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    paths = raw["paths"]
    run = raw["run"]
    hf = raw["hf"]
    datasets_section = raw["datasets"]

    train_specs = [DatasetSpec(**item) for item in datasets_section.get("train", [])]
    val_specs = [DatasetSpec(**item) for item in datasets_section.get("val", [])]

    return AppConfig(
        paths=type(AppConfig().paths)(**{k: Path(v) for k, v in paths.items()}),
        run=type(AppConfig().run)(**run),
        hf=type(AppConfig().hf)(**hf),
        datasets=type(AppConfig().datasets)(train=train_specs, val=val_specs),
    )


def resolve_under_root(root_dir: Path, configured_path: Path) -> Path:
    if configured_path.is_absolute():
        return configured_path.resolve()
    return (root_dir / configured_path).resolve()


def extract_audio(sample: dict[str, Any], audio_col: str) -> tuple[torch.Tensor, int]:
    audio_value = sample[audio_col]

    if isinstance(audio_value, dict):
        if "array" in audio_value and "sampling_rate" in audio_value:
            audio_tensor = torch.as_tensor(audio_value["array"], dtype=torch.float32)
            return audio_tensor, int(audio_value["sampling_rate"])

        if "bytes" in audio_value:
            audio_array, sampling_rate = sf.read(io.BytesIO(audio_value["bytes"]), dtype="float32")
            audio_tensor = torch.as_tensor(audio_array, dtype=torch.float32)
            return audio_tensor, int(sampling_rate)

    raise TypeError(f"Unsupported audio format for column={audio_col}")


def to_mono_channel_first(audio_tensor: torch.Tensor) -> torch.Tensor:
    if audio_tensor.ndim == 1:
        return audio_tensor.unsqueeze(0)

    if audio_tensor.ndim == 2:
        if audio_tensor.shape[0] > audio_tensor.shape[1]:
            audio_tensor = audio_tensor.t()
        if audio_tensor.shape[0] > 1:
            audio_tensor = audio_tensor.mean(dim=0, keepdim=True)
        return audio_tensor

    raise ValueError(f"Unsupported audio ndim={audio_tensor.ndim}")


def normalize_peak(audio_tensor: torch.Tensor) -> torch.Tensor:
    peak_value = torch.max(torch.abs(audio_tensor))
    if peak_value > 0:
        return audio_tensor / (peak_value + 1e-6)
    return audio_tensor


def load_hf_dataset(dataset_spec: DatasetSpec, hf_token: Optional[str], shuffle_seed: int):
    dataset_obj = datasets.load_dataset(
        path=dataset_spec.hf_id,
        name=dataset_spec.config_name,
        split=dataset_spec.split,
        streaming=dataset_spec.use_streaming,
        trust_remote_code=True,
        token=hf_token,
    )
    if not dataset_spec.use_streaming:
        dataset_obj = dataset_obj.shuffle(seed=shuffle_seed)
    return dataset_obj


def process_dataset(
    dataset_spec: DatasetSpec,
    audio_root_dir: Path,
    manifests_root_dir: Path,
    run_config,
    hf_token: Optional[str],
) -> Optional[Path]:
    manifest_path = manifests_root_dir / f"{dataset_spec.name}.jsonl"
    if run_config.skip_existing and manifest_path.exists():
        logger.info(f"Skipping manifest: {manifest_path}")
        return manifest_path

    requested_samples = dataset_spec.samples
    if run_config.sample_per_dataset is not None:
        requested_samples = min(requested_samples, run_config.sample_per_dataset)

    audio_output_dir = audio_root_dir / dataset_spec.name
    audio_output_dir.mkdir(parents=True, exist_ok=True)
    manifests_root_dir.mkdir(parents=True, exist_ok=True)

    dataset_obj = load_hf_dataset(dataset_spec, hf_token, run_config.shuffle_seed)
    dataset_iterator = iter(dataset_obj)

    resampler_cache: dict[int, torchaudio.transforms.Resample] = {}
    written_samples = 0

    progress_bar = tqdm(total=requested_samples, desc=dataset_spec.name, dynamic_ncols=True)

    with open(manifest_path, "w", encoding="utf-8") as manifest_file:
        while written_samples < requested_samples:
            try:
                sample = next(dataset_iterator)
            except StopIteration:
                break

            raw_text = sample.get(dataset_spec.text_col)
            if raw_text is None:
                continue

            text_value = str(raw_text).strip()
            if len(text_value) < 2:
                continue

            if run_config.lowercase_text:
                text_value = text_value.lower()

            audio_tensor, original_sr = extract_audio(sample, dataset_spec.audio_col)
            audio_tensor = to_mono_channel_first(audio_tensor)

            if original_sr != run_config.target_sr:
                resampler = resampler_cache.get(original_sr)
                if resampler is None:
                    resampler = torchaudio.transforms.Resample(original_sr, run_config.target_sr)
                    resampler_cache[original_sr] = resampler
                audio_tensor = resampler(audio_tensor)

            if run_config.normalize_peak:
                audio_tensor = normalize_peak(audio_tensor)

            audio_filename = f"{dataset_spec.name}_{written_samples:06d}.wav"
            audio_path = audio_output_dir / audio_filename
            sf.write(str(audio_path), audio_tensor.squeeze(0).cpu().numpy(), run_config.target_sr)

            duration_seconds = float(audio_tensor.shape[-1] / run_config.target_sr)
            manifest_item = {
                "audio_filepath": str(audio_path.resolve()),
                "text": text_value,
                "duration": duration_seconds,
                "language": dataset_spec.lang,
                "dataset": dataset_spec.name,
            }
            manifest_file.write(json.dumps(manifest_item, ensure_ascii=False) + "\n")

            written_samples += 1
            progress_bar.update(1)

    progress_bar.close()

    if written_samples == 0:
        manifest_path.unlink(missing_ok=True)
        return None

    return manifest_path


def merge_manifests(input_manifests: list[Path], output_manifest: Path) -> None:
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with open(output_manifest, "w", encoding="utf-8") as output_file:
        for input_manifest in input_manifests:
            if input_manifest.exists():
                output_file.write(input_manifest.read_text(encoding="utf-8"))


def run_pipeline(app_config: AppConfig) -> None:
    root_dir = app_config.paths.root_dir.resolve()

    audio_root_dir = resolve_under_root(root_dir, app_config.paths.audio_dir)
    individual_manifests_dir = resolve_under_root(root_dir, app_config.paths.individual_manifests_dir)
    final_train_manifest = resolve_under_root(root_dir, app_config.paths.final_train_manifest)
    final_val_manifest = resolve_under_root(root_dir, app_config.paths.final_val_manifest)

    hf_token = os.getenv(app_config.hf.token_env)

    train_manifest_paths: list[Path] = []
    val_manifest_paths: list[Path] = []

    if app_config.run.do_train:
        for dataset_spec in app_config.datasets.train:
            manifest_path = process_dataset(
                dataset_spec=dataset_spec,
                audio_root_dir=audio_root_dir,
                manifests_root_dir=individual_manifests_dir,
                run_config=app_config.run,
                hf_token=hf_token,
            )
            if manifest_path is not None:
                train_manifest_paths.append(manifest_path)

    if app_config.run.do_val:
        for dataset_spec in app_config.datasets.val:
            manifest_path = process_dataset(
                dataset_spec=dataset_spec,
                audio_root_dir=audio_root_dir,
                manifests_root_dir=individual_manifests_dir,
                run_config=app_config.run,
                hf_token=hf_token,
            )
            if manifest_path is not None:
                val_manifest_paths.append(manifest_path)

    if train_manifest_paths:
        merge_manifests(train_manifest_paths, final_train_manifest)
        logger.info(f"Wrote: {final_train_manifest}")

    if val_manifest_paths:
        merge_manifests(val_manifest_paths, final_val_manifest)
        logger.info(f"Wrote: {final_val_manifest}")


def entrypoint() -> None:
    argument_parser = argparse.ArgumentParser(add_help=False)
    argument_parser.add_argument("--config", type=Path, required=True)
    parsed_args, remaining_args = argument_parser.parse_known_args()

    base_config = load_yaml_config(parsed_args.config)
    app_config = tyro.cli(AppConfig, default=base_config, args=remaining_args)  # default=... pattern [web:21][web:1]
    run_pipeline(app_config)


if __name__ == "__main__":
    entrypoint()

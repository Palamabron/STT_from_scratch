from __future__ import annotations

import argparse
import concurrent.futures as cf
import io
import json
import os
import re
import shutil
import threading
from pathlib import Path
from typing import Any

import datasets
import soundfile as sf
import torch
import torchaudio
import tyro
import yaml
from datasets import DownloadConfig, get_dataset_split_names
from dotenv import load_dotenv
from loguru import logger
from tqdm import tqdm

from data_config import AppConfig, DatasetSpec

try:
    from datasets.utils.file_utils import xopen as hf_xopen
except Exception:
    hf_xopen = None


def load_yaml_config(config_path: Path) -> AppConfig:
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    paths = raw["paths"]
    run = raw["run"]
    hf = raw.get("hf", {"token_env": "HF_TOKEN"})
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


def _safe_audio_suffix(audio_value: dict[str, Any]) -> str:
    raw_path = audio_value.get("path")
    if isinstance(raw_path, str) and raw_path:
        suffix = Path(raw_path).suffix
        if suffix:
            return suffix
    return ".audio"


def _try_get_duration_seconds(audio_path: Path) -> float | None:
    try:
        info = torchaudio.info(str(audio_path))
        if info.sample_rate and info.num_frames:
            return float(info.num_frames / info.sample_rate)
        return None
    except Exception:
        return None


def _safe_tag(value: str) -> str:
    value = value.strip()
    value = re.sub(r"[^0-9a-zA-Z._-]+", "_", value)
    return value[:120] if len(value) > 120 else value


def _materialize_raw_audio(
    *,
    audio_value: dict[str, Any],
    output_dir: Path,
    file_prefix: str,
    file_index: int,
) -> tuple[Path, float | None]:
    suffix = _safe_audio_suffix(audio_value)
    output_path = output_dir / f"{file_prefix}_{file_index:06d}{suffix}"

    audio_bytes = audio_value.get("bytes")
    audio_path = audio_value.get("path")

    if isinstance(audio_bytes, (bytes, bytearray)) and len(audio_bytes) > 0:
        output_path.write_bytes(audio_bytes)
        return output_path, _try_get_duration_seconds(output_path)

    if isinstance(audio_path, str) and audio_path:
        source_path = Path(audio_path)
        if source_path.exists():
            shutil.copy2(source_path, output_path)
            return output_path, _try_get_duration_seconds(output_path)

        if hf_xopen is not None:
            with hf_xopen(audio_path, "rb") as f:
                output_path.write_bytes(f.read())
            return output_path, _try_get_duration_seconds(output_path)

    raise TypeError("Audio value missing usable bytes/path")


def load_hf_dataset(dataset_spec: DatasetSpec, hf_token: str | None, shuffle_seed: int):
    available_splits = get_dataset_split_names(
        path=dataset_spec.hf_id,
        config_name=dataset_spec.config_name,
        token=hf_token,
        trust_remote_code=True,
    )
    if dataset_spec.split not in available_splits:
        raise ValueError(
            f"Bad split: {dataset_spec.split}. Available splits: {available_splits}. "
            f"Dataset: {dataset_spec.hf_id} config={dataset_spec.config_name}"
        )

    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "600")
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")

    download_config = DownloadConfig(
        resume_download=True,
        max_retries=100,
    )

    dataset_obj = datasets.load_dataset(
        path=dataset_spec.hf_id,
        name=dataset_spec.config_name,
        split=dataset_spec.split,
        streaming=dataset_spec.use_streaming,
        trust_remote_code=True,
        token=hf_token,
        download_config=download_config,
    )

    try:
        dataset_obj = dataset_obj.cast_column(dataset_spec.audio_col, datasets.Audio(decode=False))
    except Exception:
        pass

    if not dataset_spec.use_streaming:
        dataset_obj = dataset_obj.shuffle(seed=shuffle_seed)

    return dataset_obj


def count_jsonl_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with open(path, "rb") as f:
        return sum(1 for _ in f)


def state_path_for_source(manifest_path: Path, dataset_spec: DatasetSpec) -> Path:
    source_id = _safe_tag(f"{dataset_spec.hf_id}__{dataset_spec.config_name}__{dataset_spec.split}")
    return manifest_path.with_suffix(f".{source_id}.state.json")


def load_resume_state(path: Path) -> dict[str, int]:
    if not path.exists():
        return {"cursor": 0}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        cursor = int(data.get("cursor", 0))
        return {"cursor": max(cursor, 0)}
    except Exception:
        return {"cursor": 0}


def save_resume_state_atomic(path: Path, cursor: int) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps({"cursor": int(cursor)}), encoding="utf-8")
    os.replace(tmp, path)


def next_free_file_index(output_dir: Path, file_prefix: str, start_index: int) -> int:
    file_index = start_index
    while True:
        if not any(output_dir.glob(f"{file_prefix}_{file_index:06d}.*")):
            return file_index
        file_index += 1


_thread_local = threading.local()


def _get_resampler(original_sr: int, target_sr: int) -> torchaudio.transforms.Resample:
    cache = getattr(_thread_local, "resampler_cache", None)
    if cache is None:
        cache = {}
        _thread_local.resampler_cache = cache
    key = (original_sr, target_sr)
    resampler = cache.get(key)
    if resampler is None:
        resampler = torchaudio.transforms.Resample(original_sr, target_sr)
        cache[key] = resampler
    return resampler


def _process_one_sample(
    *,
    sample: dict[str, Any],
    dataset_spec: DatasetSpec,
    run_config,
    audio_output_dir: Path,
    file_prefix: str,
    file_index: int,
    text_value: str,
) -> dict[str, Any] | None:
    audio_filepath: Path | None = None
    duration_seconds: float | None = None

    audio_value = sample.get(dataset_spec.audio_col)

    if isinstance(audio_value, dict) and ("bytes" in audio_value or "path" in audio_value):
        audio_filepath, duration_seconds = _materialize_raw_audio(
            audio_value=audio_value,
            output_dir=audio_output_dir,
            file_prefix=file_prefix,
            file_index=file_index,
        )
    else:
        audio_tensor, original_sr = extract_audio(sample, dataset_spec.audio_col)
        audio_tensor = to_mono_channel_first(audio_tensor)

        if original_sr != run_config.target_sr:
            resampler = _get_resampler(original_sr, run_config.target_sr)
            audio_tensor = resampler(audio_tensor)

        if run_config.normalize_peak:
            audio_tensor = normalize_peak(audio_tensor)

        audio_filepath = audio_output_dir / f"{file_prefix}_{file_index:06d}.wav"
        sf.write(
            str(audio_filepath),
            audio_tensor.squeeze(0).cpu().numpy(),
            run_config.target_sr,
        )
        duration_seconds = float(audio_tensor.shape[-1] / run_config.target_sr)

    if audio_filepath is None:
        return None

    if duration_seconds is None:
        duration_seconds = 0.0

    source_id = _safe_tag(f"{dataset_spec.config_name}__{dataset_spec.split}")

    return {
        "audio_filepath": str(audio_filepath.resolve()),
        "text": text_value,
        "duration": float(duration_seconds),
        "language": dataset_spec.lang,
        "dataset": dataset_spec.name,
        "source": source_id,
    }


def process_dataset(
    dataset_spec: DatasetSpec,
    audio_root_dir: Path,
    manifests_root_dir: Path,
    run_config,
    hf_token: str | None,
    requested_samples_override: int | None = None,
) -> Path | None:
    manifest_path = manifests_root_dir / f"{dataset_spec.name}.jsonl"
    state_path = state_path_for_source(manifest_path, dataset_spec)

    requested_samples = int(
        requested_samples_override
        if requested_samples_override is not None
        else dataset_spec.samples
    )
    if run_config.sample_per_dataset is not None:
        requested_samples = min(requested_samples, run_config.sample_per_dataset)

    if requested_samples <= 0:
        logger.info(f"No samples requested for {dataset_spec.name}, skipping.")
        return None

    audio_output_dir = audio_root_dir / dataset_spec.name
    audio_output_dir.mkdir(parents=True, exist_ok=True)
    manifests_root_dir.mkdir(parents=True, exist_ok=True)

    if not run_config.skip_existing:
        manifest_path.unlink(missing_ok=True)
        state_path.unlink(missing_ok=True)

    have = count_jsonl_lines(manifest_path)
    if have >= requested_samples:
        logger.info(f"{dataset_spec.name}: OK ({have}/{requested_samples})")
        return manifest_path

    missing = requested_samples - have
    logger.info(
        f"{dataset_spec.name}: resume have={have} target={requested_samples} missing={missing} "
        f"(source={dataset_spec.hf_id}/{dataset_spec.config_name}/{dataset_spec.split})"
    )

    resume_state = load_resume_state(state_path)
    cursor = int(resume_state["cursor"])

    dataset_obj = load_hf_dataset(dataset_spec, hf_token, run_config.shuffle_seed)

    effective_skip = dataset_spec.start_offset + cursor
    if effective_skip > 0:
        if dataset_spec.use_streaming:
            dataset_obj = dataset_obj.skip(effective_skip)
        else:
            total_len = len(dataset_obj)
            if effective_skip >= total_len:
                save_resume_state_atomic(state_path, total_len)
                logger.info(
                    f"{dataset_spec.name}: source exhausted (effective_skip={effective_skip} >= len={total_len}), skipping source."
                )
                return manifest_path
            dataset_obj = dataset_obj.select(range(effective_skip, total_len))

    dataset_iterator = iter(dataset_obj)

    failures = 0
    written_samples = have

    progress_bar = tqdm(
        total=requested_samples,
        initial=have,
        desc=dataset_spec.name,
        dynamic_ncols=True,
    )

    file_mode = "a" if manifest_path.exists() else "w"

    checkpoint_every = 200
    last_checkpoint_cursor = cursor

    num_workers_cfg = getattr(run_config, "num_workers", None)
    num_workers_env = os.getenv("PREPARE_DATA_NUM_WORKERS")
    num_workers = int(num_workers_cfg) if num_workers_cfg is not None else int(num_workers_env or 4)
    num_workers = max(1, num_workers)
    max_in_flight = max(8, num_workers * 8)

    source_tag = _safe_tag(f"{dataset_spec.config_name}__{dataset_spec.split}")
    file_prefix = _safe_tag(f"{dataset_spec.name}__{source_tag}")
    file_index_cursor = 0

    with (
        open(manifest_path, file_mode, encoding="utf-8") as manifest_file,
        cf.ThreadPoolExecutor(max_workers=num_workers) as ex,
    ):
        futures: set[cf.Future] = set()

        def drain(block: bool) -> None:
            nonlocal written_samples, failures
            if not futures:
                return
            done, not_done = cf.wait(
                futures, timeout=None if block else 0.0, return_when=cf.FIRST_COMPLETED
            )
            if not done:
                return
            futures.clear()
            futures.update(not_done)

            for fut in done:
                try:
                    item = fut.result()
                except Exception as exc:
                    failures += 1
                    if failures >= run_config.max_failures:
                        save_resume_state_atomic(state_path, cursor)
                        raise RuntimeError(
                            f"{dataset_spec.name}: reached max_failures={run_config.max_failures}"
                        ) from exc
                    continue

                if item is None:
                    continue

                if written_samples >= requested_samples:
                    continue

                manifest_file.write(json.dumps(item, ensure_ascii=False) + "\n")
                written_samples += 1
                progress_bar.update(1)

        while written_samples < requested_samples:
            made_progress = False

            while (
                written_samples + len(futures) < requested_samples and len(futures) < max_in_flight
            ):
                try:
                    sample = next(dataset_iterator)
                except StopIteration:
                    break

                made_progress = True
                cursor += 1

                if cursor - last_checkpoint_cursor >= checkpoint_every:
                    save_resume_state_atomic(state_path, cursor)
                    last_checkpoint_cursor = cursor

                raw_text = sample.get(dataset_spec.text_col)
                if raw_text is None:
                    continue

                text_value = str(raw_text).strip()
                if len(text_value) < 2:
                    continue

                if run_config.lowercase_text:
                    text_value = text_value.lower()

                file_index_cursor = next_free_file_index(
                    audio_output_dir, file_prefix, file_index_cursor
                )
                file_index = file_index_cursor
                file_index_cursor += 1

                futures.add(
                    ex.submit(
                        _process_one_sample,
                        sample=sample,
                        dataset_spec=dataset_spec,
                        run_config=run_config,
                        audio_output_dir=audio_output_dir,
                        file_prefix=file_prefix,
                        file_index=file_index,
                        text_value=text_value,
                    )
                )

            if futures:
                drain(block=True)
                continue

            if not made_progress:
                break

        while futures:
            drain(block=True)

    progress_bar.close()
    save_resume_state_atomic(state_path, cursor)

    if written_samples == 0:
        manifest_path.unlink(missing_ok=True)
        state_path.unlink(missing_ok=True)
        return None

    if written_samples < requested_samples:
        logger.warning(
            f"{dataset_spec.name}: not enough samples produced. have={written_samples} "
            f"target={requested_samples} missing={requested_samples - written_samples}"
        )

    return manifest_path


def merge_manifests(input_manifests: list[Path], output_manifest: Path) -> None:
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with open(output_manifest, "w", encoding="utf-8") as output_file:
        for input_manifest in input_manifests:
            if input_manifest.exists():
                output_file.write(input_manifest.read_text(encoding="utf-8"))


def _group_by_name(specs: list[DatasetSpec]) -> dict[str, list[DatasetSpec]]:
    groups: dict[str, list[DatasetSpec]] = {}
    for s in specs:
        groups.setdefault(s.name, []).append(s)
    return groups


def _target_for_bucket(specs: list[DatasetSpec]) -> int:
    target = int(specs[0].samples)
    for s in specs[1:]:
        if int(s.samples) != target:
            logger.warning(
                f"Bucket '{specs[0].name}': inconsistent samples values: "
                f"{target} vs {int(s.samples)}. Using {target}."
            )
    return target


def run_pipeline(app_config: AppConfig) -> None:
    root_dir = app_config.paths.root_dir.resolve()

    audio_root_dir = resolve_under_root(root_dir, app_config.paths.audio_dir)
    individual_manifests_dir = resolve_under_root(
        root_dir, app_config.paths.individual_manifests_dir
    )
    final_train_manifest = resolve_under_root(root_dir, app_config.paths.final_train_manifest)
    final_val_manifest = resolve_under_root(root_dir, app_config.paths.final_val_manifest)

    hf_token = os.getenv(app_config.hf.token_env)

    train_manifest_paths: list[Path] = []
    val_manifest_paths: list[Path] = []

    if app_config.run.do_train:
        for bucket_name, specs in _group_by_name(app_config.datasets.train).items():
            target = _target_for_bucket(specs)
            logger.info(f"Train bucket: {bucket_name} target={target} sources={len(specs)}")

            manifest_path_for_bucket = individual_manifests_dir / f"{bucket_name}.jsonl"
            for spec in specs:
                have_now = count_jsonl_lines(manifest_path_for_bucket)
                if have_now >= target:
                    break

                mp = process_dataset(
                    dataset_spec=spec,
                    audio_root_dir=audio_root_dir,
                    manifests_root_dir=individual_manifests_dir,
                    run_config=app_config.run,
                    hf_token=hf_token,
                    requested_samples_override=target,
                )
                if mp is not None and mp not in train_manifest_paths:
                    train_manifest_paths.append(mp)

    if app_config.run.do_val:
        for bucket_name, specs in _group_by_name(app_config.datasets.val).items():
            target = _target_for_bucket(specs)
            logger.info(f"Val bucket: {bucket_name} target={target} sources={len(specs)}")

            manifest_path_for_bucket = individual_manifests_dir / f"{bucket_name}.jsonl"
            for spec in specs:
                have_now = count_jsonl_lines(manifest_path_for_bucket)
                if have_now >= target:
                    break

                mp = process_dataset(
                    dataset_spec=spec,
                    audio_root_dir=audio_root_dir,
                    manifests_root_dir=individual_manifests_dir,
                    run_config=app_config.run,
                    hf_token=hf_token,
                    requested_samples_override=target,
                )
                if mp is not None and mp not in val_manifest_paths:
                    val_manifest_paths.append(mp)

    if train_manifest_paths:
        merge_manifests(train_manifest_paths, final_train_manifest)
        logger.info(f"Wrote: {final_train_manifest}")

    if val_manifest_paths:
        merge_manifests(val_manifest_paths, final_val_manifest)
        logger.info(f"Wrote: {final_val_manifest}")


def _resolve_config_path(config_arg: str | None) -> Path:
    default_path = Path("configs/data.yaml")
    candidates: list[Path] = []

    if config_arg is None or str(config_arg).strip() == "":
        candidates.append(default_path)
    else:
        p = Path(config_arg)
        candidates.append(p)
        if not p.is_absolute():
            candidates.append(Path("configs") / p.name)
            candidates.append(Path("configs") / p)
            stem = p.stem
            candidates.append(Path("configs") / f"{stem}.yaml")
            candidates.append(Path("configs") / f"{stem}.yml")

    for c in candidates:
        if c.exists():
            return c

    available = []
    cfg_dir = Path("configs")
    if cfg_dir.exists():
        available = sorted([p.name for p in cfg_dir.glob("*.y*ml")])

    raise FileNotFoundError(
        f"Config not found. Tried: {[str(c) for c in candidates]}. "
        f"Available in ./configs: {available}"
    )


def entrypoint() -> None:
    load_dotenv()
    argument_parser = argparse.ArgumentParser(add_help=False)
    argument_parser.add_argument("--config", type=str, required=False, default=None)
    parsed_args, remaining_args = argument_parser.parse_known_args()

    config_path = _resolve_config_path(parsed_args.config)
    base_config = load_yaml_config(config_path)
    app_config = tyro.cli(AppConfig, default=base_config, args=remaining_args)
    run_pipeline(app_config)


if __name__ == "__main__":
    entrypoint()

from __future__ import annotations

import hashlib
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
from datasets import DownloadConfig, get_dataset_split_names
from datasets.utils.file_utils import xopen as hf_xopen
from loguru import logger

from .config import DatasetSpec


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


def duration_seconds(audio_path: Path) -> float | None:
    """Read audio duration from file headers (torchaudio, then soundfile fallback)."""
    try:
        info = torchaudio.info(str(audio_path))
        sr = getattr(info, "sample_rate", 0) or 0
        nf = getattr(info, "num_frames", 0) or 0
        if sr > 0 and nf > 0:
            return float(nf / sr)
    except Exception:
        pass

    try:
        with sf.SoundFile(str(audio_path)) as handle:
            if handle.samplerate <= 0:
                return None
            return float(handle.frames / handle.samplerate)
    except Exception:
        return None


def _try_get_duration_seconds(audio_path: Path) -> float | None:
    return duration_seconds(audio_path)


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

    if isinstance(audio_bytes, bytes | bytearray) and len(audio_bytes) > 0:
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


def is_hf_rate_limit_error(exc: BaseException) -> bool:
    """Return True when *exc* looks like a HuggingFace Hub 429 rate-limit error."""
    try:
        from huggingface_hub.errors import HfHubHTTPError

        if isinstance(exc, HfHubHTTPError):
            response = getattr(exc, "response", None)
            if response is not None and getattr(response, "status_code", None) == 429:
                return True
    except ImportError:
        pass

    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        message = str(current).lower()
        if "429" in message or "too many requests" in message or "rate limit" in message:
            return True
        current = current.__cause__ or current.__context__
    return False


def next_sample_with_hf_retry(
    iterator,
    *,
    label: str,
    shard_idx: int | None = None,
    max_attempts: int = 24,
    initial_backoff_sec: float = 45.0,
    max_backoff_sec: float = 330.0,
):
    """Fetch the next HF streaming sample, backing off on Hub 429 rate limits."""
    import time

    attempt = 0
    while True:
        try:
            return next(iterator)
        except StopIteration:
            raise
        except Exception as exc:
            if not is_hf_rate_limit_error(exc) or attempt >= max_attempts:
                raise
            wait = min(initial_backoff_sec * (2**attempt), max_backoff_sec)
            shard_note = f" shard={shard_idx}" if shard_idx is not None else ""
            logger.warning(
                f"{label}{shard_note}: HF rate limit, sleeping {wait:.0f}s "
                f"(retry {attempt + 1}/{max_attempts})"
            )
            time.sleep(wait)
            attempt += 1


def shard_streaming_dataset(dataset_obj: Any, num_shards: int, shard_idx: int) -> Any:
    """Split a streaming HF dataset across parallel fetch workers by data-source shard."""
    from datasets.iterable_dataset import IterableDataset

    if num_shards <= 1:
        return dataset_obj

    sharded_ex = dataset_obj._ex_iterable.shard_data_sources(shard_idx, num_shards)
    return IterableDataset(
        sharded_ex,
        info=dataset_obj.info,
        split=dataset_obj.split,
        formatting=dataset_obj._formatting,
        shuffling=dataset_obj._shuffling,
        distributed=dataset_obj._distributed,
        token_per_repo_id=dataset_obj._token_per_repo_id,
    )


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
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "120")

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


def state_path_for_source(manifest_path: Path, dataset_spec: DatasetSpec) -> Path:
    source_id = _safe_tag(f"{dataset_spec.hf_id}__{dataset_spec.config_name}__{dataset_spec.split}")
    return manifest_path.with_suffix(f".{source_id}.state.json")


def load_resume_state(path: Path) -> dict[str, int | bool]:
    if not path.exists():
        return {"cursor": 0, "exhausted": False}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        cursor = int(data.get("cursor", 0))
        return {"cursor": max(cursor, 0), "exhausted": bool(data.get("exhausted", False))}
    except Exception:
        return {"cursor": 0, "exhausted": False}


def save_resume_state_atomic(path: Path, cursor: int, *, exhausted: bool | None = None) -> None:
    payload: dict[str, int | bool] = {"cursor": int(cursor)}
    if exhausted is not None:
        payload["exhausted"] = exhausted
    elif path.exists():
        try:
            prev = json.loads(path.read_text(encoding="utf-8"))
            if prev.get("exhausted"):
                payload["exhausted"] = True
        except Exception:
            pass
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload), encoding="utf-8")
    os.replace(tmp, path)


def next_free_file_index(output_dir: Path, file_prefix: str, start_index: int) -> int:
    file_index = start_index
    while True:
        if not any(output_dir.glob(f"{file_prefix}_{file_index:06d}.*")):
            return file_index
        file_index += 1


def _make_sample_uid(
    *,
    sample: dict[str, Any],
    dataset_spec: DatasetSpec,
    global_pos: int,
) -> str:
    audio_value = sample.get(dataset_spec.audio_col)

    if isinstance(audio_value, dict):
        p = audio_value.get("path")
        if isinstance(p, str) and p:
            key = f"path:{p}"
        else:
            b = audio_value.get("bytes")
            if isinstance(b, bytes | bytearray) and len(b) > 0:
                key = "sha1:" + hashlib.sha1(b).hexdigest()
            else:
                key = f"pos:{global_pos}"
    else:
        sid = sample.get("id", None) or sample.get("_id", None)
        if sid is not None:
            key = f"id:{sid}"
        else:
            key = f"pos:{global_pos}"

    cfg = dataset_spec.config_name or ""
    return _safe_tag(f"{dataset_spec.hf_id}__{cfg}__{dataset_spec.split}__{key}")


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
    uid: str,
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

    if duration_seconds is None or duration_seconds <= 0.0:
        if audio_filepath.exists():
            audio_filepath.unlink(missing_ok=True)
        return None

    source_id = _safe_tag(f"{dataset_spec.config_name}__{dataset_spec.split}")

    return {
        "uid": uid,
        "audio_filepath": str(audio_filepath.resolve()),
        "text": text_value,
        "duration": float(duration_seconds),
        "language": dataset_spec.lang,
        "dataset": dataset_spec.name,
        "source": source_id,
    }

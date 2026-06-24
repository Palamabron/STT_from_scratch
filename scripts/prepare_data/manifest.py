from __future__ import annotations

import concurrent.futures as cf
import json
import os
import threading
from pathlib import Path
from typing import Any

from loguru import logger
from tqdm import tqdm

from .buckets import UnderdeliveryPolicy, handle_underdelivery, load_existing_keys
from .config import DatasetSpec, Run
from .hf_loader import (
    _make_sample_uid,
    _process_one_sample,
    _safe_tag,
    load_hf_dataset,
    load_resume_state,
    next_free_file_index,
    next_sample_with_hf_retry,
    save_resume_state_atomic,
    shard_streaming_dataset,
    state_path_for_source,
)


def _resolve_num_workers(run_config: Run) -> int:
    num_workers = run_config.num_workers
    if num_workers is None:
        num_workers = int(os.getenv("PREPARE_DATA_NUM_WORKERS") or 4)
    return max(1, int(num_workers))


def _resolve_fetch_shards(run_config: Run, *, dataset_n_shards: int | None = None) -> int:
    fetch_shards = run_config.fetch_shards
    if fetch_shards is None:
        fetch_shards = int(os.getenv("PREPARE_DATA_FETCH_SHARDS") or 1)
    fetch_shards = max(1, int(fetch_shards))
    if dataset_n_shards is not None and dataset_n_shards > 0:
        fetch_shards = min(fetch_shards, int(dataset_n_shards))
    return fetch_shards


def _shard_state_path(state_path: Path, shard_idx: int) -> Path:
    return state_path.with_name(f"{state_path.stem}.shard{shard_idx}{state_path.suffix}")


class _SampleProcessor:
    """Shared fetch/process loop state for one HF source."""

    def __init__(
        self,
        *,
        dataset_spec: DatasetSpec,
        run_config: Run,
        audio_output_dir: Path,
        manifest_file,
        state_path: Path,
        requested_samples: int,
        have: int,
        progress_bar: tqdm,
        ex: cf.ThreadPoolExecutor,
        seen: set[str],
    ) -> None:
        self.dataset_spec = dataset_spec
        self.run_config = run_config
        self.audio_output_dir = audio_output_dir
        self.manifest_file = manifest_file
        self.state_path = state_path
        self.requested_samples = requested_samples
        self.progress_bar = progress_bar
        self.ex = ex
        self.seen = seen
        self.lock = threading.Lock()
        self.pending_uid: set[str] = set()
        self.futures: set[cf.Future] = set()
        self.future_uid: dict[cf.Future, str] = {}
        self.failures = 0
        self.written_samples = have
        self.file_index_cursor = 0
        self.max_in_flight = max(8, _resolve_num_workers(run_config) * 8)
        source_tag = _safe_tag(f"{dataset_spec.config_name}__{dataset_spec.split}")
        self.file_prefix = _safe_tag(f"{dataset_spec.name}__{source_tag}")
        self.stop_event = threading.Event()

    def should_stop(self) -> bool:
        return self.stop_event.is_set() or self.written_samples >= self.requested_samples

    def drain(self, block: bool) -> None:
        with self.lock:
            if not self.futures:
                return
            done, not_done = cf.wait(
                self.futures,
                timeout=None if block else 0.0,
                return_when=cf.FIRST_COMPLETED,
            )
            if not done:
                return
            self.futures.clear()
            self.futures.update(not_done)

            for fut in done:
                uid = self.future_uid.pop(fut, None)
                if uid is not None:
                    self.pending_uid.discard(uid)

                try:
                    item = fut.result()
                except Exception as exc:
                    self.failures += 1
                    if self.failures >= self.run_config.max_failures:
                        raise RuntimeError(
                            f"{self.dataset_spec.name}: reached max_failures="
                            f"{self.run_config.max_failures}"
                        ) from exc
                    continue

                if item is None:
                    continue

                key = item.get("uid") or item.get("audio_filepath")
                if not isinstance(key, str) or not key:
                    continue
                if key in self.seen:
                    continue

                self.seen.add(key)
                now = len(self.seen)
                if now > self.written_samples and now <= self.requested_samples:
                    self.manifest_file.write(json.dumps(item, ensure_ascii=False) + "\n")
                    self.written_samples = now
                    self.progress_bar.n = self.written_samples
                    self.progress_bar.refresh()
                    if self.written_samples >= self.requested_samples:
                        self.stop_event.set()

    def submit_sample(self, sample: dict[str, Any], *, global_pos: int) -> bool:
        if self.should_stop():
            return False

        raw_text = sample.get(self.dataset_spec.text_col)
        if raw_text is None:
            return True

        text_value = str(raw_text).strip()
        if len(text_value) < 2:
            return True

        if self.run_config.lowercase_text:
            text_value = text_value.lower()

        uid = _make_sample_uid(
            sample=sample,
            dataset_spec=self.dataset_spec,
            global_pos=global_pos,
        )

        with self.lock:
            if self.should_stop():
                return False
            if uid in self.seen or uid in self.pending_uid:
                return True
            if self.written_samples + len(self.futures) >= self.requested_samples:
                self.stop_event.set()
                return False
            self.pending_uid.add(uid)
            file_index = next_free_file_index(
                self.audio_output_dir,
                self.file_prefix,
                self.file_index_cursor,
            )
            self.file_index_cursor = file_index + 1

            fut = self.ex.submit(
                _process_one_sample,
                uid=uid,
                sample=sample,
                dataset_spec=self.dataset_spec,
                run_config=self.run_config,
                audio_output_dir=self.audio_output_dir,
                file_prefix=self.file_prefix,
                file_index=file_index,
                text_value=text_value,
            )
            self.futures.add(fut)
            self.future_uid[fut] = uid
        return True

    def wait_for_capacity(self) -> None:
        while not self.should_stop():
            with self.lock:
                if len(self.futures) < self.max_in_flight:
                    return
            self.drain(block=True)

    def finalize(self) -> None:
        while True:
            with self.lock:
                if not self.futures:
                    break
            self.drain(block=True)


def _resolve_resume_cursor(
    *,
    cursor: int,
    have: int,
    dataset_len: int | None,
    use_streaming: bool,
    skip_existing: bool,
    source_name: str,
    state_path: Path,
) -> int:
    """Normalize resume cursor; scanned HF positions must not skip non-streaming datasets."""
    stale = False
    if dataset_len is not None and cursor > dataset_len:
        stale = True
    elif have > 0 and cursor > 5 * have:
        stale = True

    if stale:
        logger.warning(
            f"{source_name}: resetting stale cursor={cursor} (have={have}, dataset_len={dataset_len})"
        )
        save_resume_state_atomic(state_path, 0, exhausted=False)
        return 0

    if not use_streaming and skip_existing and have > 0:
        if cursor != 0:
            logger.info(
                f"{source_name}: non-streaming resume with skip_existing; "
                f"ignoring scanned cursor={cursor}, iterating from start"
            )
        return 0

    return cursor


def _run_fetch_loop(
    *,
    processor: _SampleProcessor,
    dataset_iterator,
    state_path: Path,
    start_cursor: int,
    start_offset: int,
    shard_idx: int = 0,
    fetch_shards: int = 1,
) -> int:
    cursor = start_cursor
    last_checkpoint_cursor = cursor
    checkpoint_every = 200

    try:
        while not processor.should_stop():
            processor.wait_for_capacity()
            if processor.should_stop():
                break

            try:
                sample = next_sample_with_hf_retry(
                    dataset_iterator,
                    label=processor.dataset_spec.name,
                    shard_idx=shard_idx if fetch_shards > 1 else None,
                )
            except StopIteration:
                break

            cursor += 1
            if cursor - last_checkpoint_cursor >= checkpoint_every:
                save_resume_state_atomic(state_path, cursor)
                last_checkpoint_cursor = cursor

            global_pos = start_offset + shard_idx + cursor * fetch_shards
            if not processor.submit_sample(sample, global_pos=global_pos):
                break

            processor.drain(block=False)
    finally:
        processor.finalize()
        save_resume_state_atomic(state_path, cursor)

    return cursor


def _process_source_streaming_sharded(
    *,
    dataset_spec: DatasetSpec,
    audio_root_dir: Path,
    manifest_path: Path,
    state_path: Path,
    run_config: Run,
    hf_token: str | None,
    requested_samples: int,
    have: int,
    fetch_shards: int,
    base_dataset,
    underdelivery_policy: UnderdeliveryPolicy,
) -> Path | None:
    audio_output_dir = audio_root_dir / dataset_spec.name
    file_mode = "a" if manifest_path.exists() else "w"
    seen = load_existing_keys(manifest_path)

    progress_bar = tqdm(
        total=requested_samples,
        initial=have,
        desc=f"{dataset_spec.name} (x{fetch_shards} fetch)",
        dynamic_ncols=True,
    )

    num_workers = _resolve_num_workers(run_config)
    logger.info(
        f"{dataset_spec.name}: parallel streaming fetch shards={fetch_shards} "
        f"process_workers={num_workers}"
    )

    with (
        open(manifest_path, file_mode, encoding="utf-8") as manifest_file,
        cf.ThreadPoolExecutor(max_workers=num_workers) as process_ex,
        cf.ThreadPoolExecutor(max_workers=fetch_shards) as fetch_ex,
    ):
        processor = _SampleProcessor(
            dataset_spec=dataset_spec,
            run_config=run_config,
            audio_output_dir=audio_output_dir,
            manifest_file=manifest_file,
            state_path=state_path,
            requested_samples=requested_samples,
            have=have,
            progress_bar=progress_bar,
            ex=process_ex,
            seen=seen,
        )

        fetch_futures: list[cf.Future] = []
        for shard_idx in range(fetch_shards):
            shard_state_path = _shard_state_path(state_path, shard_idx)
            shard_cursor = int(load_resume_state(shard_state_path)["cursor"])

            dataset_obj = shard_streaming_dataset(base_dataset, fetch_shards, shard_idx)
            if shard_cursor > 0:
                dataset_obj = dataset_obj.skip(shard_cursor)

            fetch_futures.append(
                fetch_ex.submit(
                    _run_fetch_loop,
                    processor=processor,
                    dataset_iterator=iter(dataset_obj),
                    state_path=shard_state_path,
                    start_cursor=shard_cursor,
                    start_offset=dataset_spec.start_offset,
                    shard_idx=shard_idx,
                    fetch_shards=fetch_shards,
                )
            )

        for fut in cf.as_completed(fetch_futures):
            fut.result()

    progress_bar.close()

    written_samples = processor.written_samples
    if written_samples == 0:
        manifest_path.unlink(missing_ok=True)
        for shard_idx in range(fetch_shards):
            _shard_state_path(state_path, shard_idx).unlink(missing_ok=True)
        state_path.unlink(missing_ok=True)
        return None

    if written_samples < requested_samples:
        msg = (
            f"{dataset_spec.name}: not enough samples produced. have={written_samples} "
            f"target={requested_samples} missing={requested_samples - written_samples} "
            f"(source={dataset_spec.hf_id}/{dataset_spec.config_name}/{dataset_spec.split})"
        )
        handle_underdelivery(underdelivery_policy, msg)
        if not dataset_spec.use_streaming:
            save_resume_state_atomic(state_path, cursor, exhausted=True)
            logger.info(
                f"{dataset_spec.name}: marked source exhausted after non-streaming full scan"
            )

    return manifest_path


def process_source(
    dataset_spec: DatasetSpec,
    audio_root_dir: Path,
    manifests_root_dir: Path,
    run_config: Run,
    hf_token: str | None,
    requested_samples_override: int | None = None,
    *,
    underdelivery_policy: UnderdeliveryPolicy,
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

    seen = load_existing_keys(manifest_path)
    have = len(seen)
    if have >= requested_samples:
        logger.info(f"{dataset_spec.name}: OK ({have}/{requested_samples})")
        return manifest_path

    missing = requested_samples - have
    logger.info(
        f"{dataset_spec.name}: resume have={have} target={requested_samples} missing={missing} "
        f"(source={dataset_spec.hf_id}/{dataset_spec.config_name}/{dataset_spec.split})"
    )

    resume_state = load_resume_state(state_path)
    if resume_state.get("exhausted"):
        if have < requested_samples:
            logger.warning(
                f"{dataset_spec.name}: clearing stale exhausted flag "
                f"(have={have} < target={requested_samples})"
            )
            save_resume_state_atomic(state_path, int(resume_state["cursor"]), exhausted=False)
        else:
            logger.info(
                f"{dataset_spec.name}: source exhausted (cached), skipping "
                f"(source={dataset_spec.hf_id}/{dataset_spec.config_name}/{dataset_spec.split})"
            )
            return manifest_path

    cursor = int(resume_state["cursor"])

    fetch_shards = _resolve_fetch_shards(run_config)
    if dataset_spec.use_streaming and fetch_shards > 1:
        probe = load_hf_dataset(dataset_spec, hf_token, run_config.shuffle_seed)
        n_shards = int(getattr(probe, "n_shards", 0) or getattr(probe._ex_iterable, "n_shards", 0))
        fetch_shards = _resolve_fetch_shards(run_config, dataset_n_shards=n_shards or None)
        if fetch_shards > 1:
            return _process_source_streaming_sharded(
                dataset_spec=dataset_spec,
                audio_root_dir=audio_root_dir,
                manifest_path=manifest_path,
                state_path=state_path,
                run_config=run_config,
                hf_token=hf_token,
                requested_samples=requested_samples,
                have=have,
                fetch_shards=fetch_shards,
                base_dataset=probe,
                underdelivery_policy=underdelivery_policy,
            )

    dataset_obj = load_hf_dataset(dataset_spec, hf_token, run_config.shuffle_seed)

    dataset_len = None if dataset_spec.use_streaming else len(dataset_obj)
    cursor = _resolve_resume_cursor(
        cursor=cursor,
        have=have,
        dataset_len=dataset_len,
        use_streaming=dataset_spec.use_streaming,
        skip_existing=run_config.skip_existing,
        source_name=dataset_spec.name,
        state_path=state_path,
    )

    effective_skip = dataset_spec.start_offset + cursor
    if effective_skip > 0:
        if dataset_spec.use_streaming:
            dataset_obj = dataset_obj.skip(effective_skip)
        else:
            total_len = len(dataset_obj)
            if effective_skip >= total_len:
                save_resume_state_atomic(state_path, total_len, exhausted=True)
                logger.info(
                    f"{dataset_spec.name}: source exhausted (effective_skip={effective_skip} >= len={total_len}), skipping source."
                )
                return manifest_path
            dataset_obj = dataset_obj.select(range(effective_skip, total_len))

    written_samples = have

    progress_bar = tqdm(
        total=requested_samples,
        initial=written_samples,
        desc=dataset_spec.name,
        dynamic_ncols=True,
    )

    file_mode = "a" if manifest_path.exists() else "w"
    num_workers = _resolve_num_workers(run_config)

    with (
        open(manifest_path, file_mode, encoding="utf-8") as manifest_file,
        cf.ThreadPoolExecutor(max_workers=num_workers) as ex,
    ):
        processor = _SampleProcessor(
            dataset_spec=dataset_spec,
            run_config=run_config,
            audio_output_dir=audio_output_dir,
            manifest_file=manifest_file,
            state_path=state_path,
            requested_samples=requested_samples,
            have=have,
            progress_bar=progress_bar,
            ex=ex,
            seen=seen,
        )
        cursor = _run_fetch_loop(
            processor=processor,
            dataset_iterator=iter(dataset_obj),
            state_path=state_path,
            start_cursor=cursor,
            start_offset=dataset_spec.start_offset,
        )

        written_samples = processor.written_samples

    progress_bar.close()

    if written_samples == 0:
        manifest_path.unlink(missing_ok=True)
        state_path.unlink(missing_ok=True)
        return None

    if written_samples < requested_samples:
        msg = (
            f"{dataset_spec.name}: not enough samples produced. have={written_samples} "
            f"target={requested_samples} missing={requested_samples - written_samples} "
            f"(source={dataset_spec.hf_id}/{dataset_spec.config_name}/{dataset_spec.split})"
        )
        handle_underdelivery(underdelivery_policy, msg)
        if not dataset_spec.use_streaming:
            save_resume_state_atomic(state_path, cursor, exhausted=True)
            logger.info(
                f"{dataset_spec.name}: marked source exhausted after non-streaming full scan"
            )

    return manifest_path


def process_train_source(
    *,
    dataset_spec: DatasetSpec,
    audio_root_dir: Path,
    manifests_root_dir: Path,
    run_config: Run,
    hf_token: str | None,
    requested_samples_override: int | None = None,
) -> Path | None:
    return process_source(
        dataset_spec,
        audio_root_dir,
        manifests_root_dir,
        run_config,
        hf_token,
        requested_samples_override,
        underdelivery_policy=UnderdeliveryPolicy.WARN,
    )


def process_val_source(
    *,
    dataset_spec: DatasetSpec,
    audio_root_dir: Path,
    manifests_root_dir: Path,
    run_config: Run,
    hf_token: str | None,
    requested_samples_override: int | None = None,
) -> Path | None:
    return process_source(
        dataset_spec,
        audio_root_dir,
        manifests_root_dir,
        run_config,
        hf_token,
        requested_samples_override,
        underdelivery_policy=UnderdeliveryPolicy.RAISE,
    )

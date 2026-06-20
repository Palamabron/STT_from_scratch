from __future__ import annotations

import concurrent.futures as cf
import json
import os
from pathlib import Path

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
    save_resume_state_atomic,
    state_path_for_source,
)


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
        initial=written_samples,
        desc=dataset_spec.name,
        dynamic_ncols=True,
    )

    file_mode = "a" if manifest_path.exists() else "w"

    checkpoint_every = 200
    last_checkpoint_cursor = cursor

    num_workers = run_config.num_workers
    if num_workers is None:
        num_workers = int(os.getenv("PREPARE_DATA_NUM_WORKERS") or 4)
    num_workers = max(1, int(num_workers))
    max_in_flight = max(8, num_workers * 8)

    source_tag = _safe_tag(f"{dataset_spec.config_name}__{dataset_spec.split}")
    file_prefix = _safe_tag(f"{dataset_spec.name}__{source_tag}")
    file_index_cursor = 0

    with (
        open(manifest_path, file_mode, encoding="utf-8") as manifest_file,
        cf.ThreadPoolExecutor(max_workers=num_workers) as ex,
    ):
        futures: set[cf.Future] = set()
        future_uid: dict[cf.Future, str] = {}
        pending_uid: set[str] = set()

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
                uid = future_uid.pop(fut, None)
                if uid is not None:
                    pending_uid.discard(uid)

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

                key = item.get("uid") or item.get("audio_filepath")
                if not isinstance(key, str) or not key:
                    continue
                if key in seen:
                    continue

                seen.add(key)
                now = len(seen)
                if now > written_samples and now <= requested_samples:
                    manifest_file.write(json.dumps(item, ensure_ascii=False) + "\n")
                    written_samples = now
                    progress_bar.n = written_samples
                    progress_bar.refresh()

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

                global_pos = dataset_spec.start_offset + cursor
                uid = _make_sample_uid(
                    sample=sample, dataset_spec=dataset_spec, global_pos=global_pos
                )
                if uid in seen or uid in pending_uid:
                    continue
                pending_uid.add(uid)

                file_index_cursor = next_free_file_index(
                    audio_output_dir, file_prefix, file_index_cursor
                )
                file_index = file_index_cursor
                file_index_cursor += 1

                fut = ex.submit(
                    _process_one_sample,
                    uid=uid,
                    sample=sample,
                    dataset_spec=dataset_spec,
                    run_config=run_config,
                    audio_output_dir=audio_output_dir,
                    file_prefix=file_prefix,
                    file_index=file_index,
                    text_value=text_value,
                )
                futures.add(fut)
                future_uid[fut] = uid

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
        msg = (
            f"{dataset_spec.name}: not enough samples produced. have={written_samples} "
            f"target={requested_samples} missing={requested_samples - written_samples} "
            f"(source={dataset_spec.hf_id}/{dataset_spec.config_name}/{dataset_spec.split})"
        )
        handle_underdelivery(underdelivery_policy, msg)

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

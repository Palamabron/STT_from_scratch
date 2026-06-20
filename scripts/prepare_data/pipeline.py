from __future__ import annotations

import json
import os
from pathlib import Path

from loguru import logger

from .buckets import (
    UnderdeliveryPolicy,
    assert_bucket_ready,
    count_unique_in_manifest,
    group_by_name,
    handle_underdelivery,
    target_for_bucket,
)
from .config import AppConfig, DatasetSpec
from .hf_loader import resolve_under_root
from .manifest import process_train_source, process_val_source
from .overlap import assert_no_overlap, assert_val_has_no_train_bucket_names


def build_train_final_from_buckets(
    *,
    bucket_specs: list[DatasetSpec],
    individual_manifests_dir: Path,
    output_manifest: Path,
) -> None:
    _build_final_from_buckets(
        bucket_specs=bucket_specs,
        individual_manifests_dir=individual_manifests_dir,
        output_manifest=output_manifest,
        underdelivery_policy=UnderdeliveryPolicy.WARN,
    )


def build_val_final_from_buckets(
    *,
    bucket_specs: list[DatasetSpec],
    individual_manifests_dir: Path,
    output_manifest: Path,
) -> None:
    _build_final_from_buckets(
        bucket_specs=bucket_specs,
        individual_manifests_dir=individual_manifests_dir,
        output_manifest=output_manifest,
        underdelivery_policy=UnderdeliveryPolicy.RAISE,
    )


def _build_final_from_buckets(
    *,
    bucket_specs: list[DatasetSpec],
    individual_manifests_dir: Path,
    output_manifest: Path,
    underdelivery_policy: UnderdeliveryPolicy,
) -> None:
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    groups = group_by_name(bucket_specs)
    seen_global: set[str] = set()

    with output_manifest.open("w", encoding="utf-8") as out:
        for bucket_name, specs in groups.items():
            target = target_for_bucket(specs)
            bucket_manifest = individual_manifests_dir / f"{bucket_name}.jsonl"
            if not bucket_manifest.exists():
                continue

            kept = 0
            with bucket_manifest.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if kept >= target:
                        break
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    key = obj.get("uid") or obj.get("audio_filepath")
                    if not isinstance(key, str) or not key:
                        continue
                    if key in seen_global:
                        continue
                    seen_global.add(key)
                    out.write(json.dumps(obj, ensure_ascii=False) + "\n")
                    kept += 1

            if kept < target:
                handle_underdelivery(
                    underdelivery_policy,
                    f"Bucket '{bucket_name}': final manifest kept={kept} target={target}",
                )


def validate_final_manifests(app_config: AppConfig) -> None:
    """Ensure train/val finals exist and do not overlap."""
    root_dir = app_config.paths.root_dir.resolve()
    train_out = resolve_under_root(root_dir, app_config.paths.final_train_manifest)
    val_out = resolve_under_root(root_dir, app_config.paths.final_val_manifest)
    if not train_out.exists() or not val_out.exists():
        return

    train_bucket_names = set(group_by_name(app_config.datasets.train))
    assert_no_overlap(train_out, val_out)
    assert_val_has_no_train_bucket_names(val_out, train_bucket_names)
    logger.info("Validated train/val final manifests: no overlap detected")


def run_pipeline(app_config: AppConfig) -> None:
    root_dir = app_config.paths.root_dir.resolve()

    audio_root_dir = resolve_under_root(root_dir, app_config.paths.audio_dir)
    individual_manifests_dir = resolve_under_root(
        root_dir, app_config.paths.individual_manifests_dir
    )
    final_train_manifest = resolve_under_root(root_dir, app_config.paths.final_train_manifest)
    final_val_manifest = resolve_under_root(root_dir, app_config.paths.final_val_manifest)

    hf_token = os.getenv(app_config.hf.token_env)

    if app_config.run.do_train:
        for bucket_name, specs in group_by_name(app_config.datasets.train).items():
            target = target_for_bucket(specs)
            manifest_path_for_bucket = individual_manifests_dir / f"{bucket_name}.jsonl"
            have_now = count_unique_in_manifest(manifest_path_for_bucket)
            if have_now >= target:
                logger.info(
                    f"Train bucket: {bucket_name} OK ({have_now}/{target}) sources={len(specs)}"
                )
                continue

            logger.info(
                f"Train bucket: {bucket_name} target={target} have={have_now} sources={len(specs)}"
            )

            for spec in specs:
                have_now = count_unique_in_manifest(manifest_path_for_bucket)
                if have_now >= target:
                    break
                process_train_source(
                    dataset_spec=spec,
                    audio_root_dir=audio_root_dir,
                    manifests_root_dir=individual_manifests_dir,
                    run_config=app_config.run,
                    hf_token=hf_token,
                    requested_samples_override=target,
                )

        build_train_final_from_buckets(
            bucket_specs=app_config.datasets.train,
            individual_manifests_dir=individual_manifests_dir,
            output_manifest=final_train_manifest,
        )
        logger.info(f"Wrote: {final_train_manifest}")

    if app_config.run.do_val:
        for bucket_name, specs in group_by_name(app_config.datasets.val).items():
            target = target_for_bucket(specs)
            manifest_path_for_bucket = individual_manifests_dir / f"{bucket_name}.jsonl"
            have_now = count_unique_in_manifest(manifest_path_for_bucket)
            if have_now >= target:
                logger.info(
                    f"Val bucket: {bucket_name} OK ({have_now}/{target}) sources={len(specs)}"
                )
                continue

            logger.info(
                f"Val bucket: {bucket_name} target={target} have={have_now} sources={len(specs)}"
            )

            for spec in specs:
                have_now = count_unique_in_manifest(manifest_path_for_bucket)
                if have_now >= target:
                    break
                process_val_source(
                    dataset_spec=spec,
                    audio_root_dir=audio_root_dir,
                    manifests_root_dir=individual_manifests_dir,
                    run_config=app_config.run,
                    hf_token=hf_token,
                    requested_samples_override=target,
                )

            assert_bucket_ready(
                bucket_name,
                manifest_path_for_bucket,
                target,
                UnderdeliveryPolicy.RAISE,
            )

        build_val_final_from_buckets(
            bucket_specs=app_config.datasets.val,
            individual_manifests_dir=individual_manifests_dir,
            output_manifest=final_val_manifest,
        )
        logger.info(f"Wrote: {final_val_manifest}")

    validate_final_manifests(app_config)

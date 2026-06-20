from __future__ import annotations

import json
from enum import Enum
from pathlib import Path

from loguru import logger

from .config import DatasetSpec


class UnderdeliveryPolicy(Enum):
    WARN = "warn"
    RAISE = "raise"


def handle_underdelivery(policy: UnderdeliveryPolicy, message: str) -> None:
    if policy == UnderdeliveryPolicy.RAISE:
        raise RuntimeError(message)
    logger.warning(message)


def group_by_name(specs: list[DatasetSpec]) -> dict[str, list[DatasetSpec]]:
    groups: dict[str, list[DatasetSpec]] = {}
    for spec in specs:
        groups.setdefault(spec.name, []).append(spec)
    return groups


def target_for_bucket(specs: list[DatasetSpec]) -> int:
    target = int(specs[0].samples)
    for spec in specs[1:]:
        if int(spec.samples) != target:
            logger.warning(
                f"Bucket '{specs[0].name}': inconsistent samples values: "
                f"{target} vs {int(spec.samples)}. Using {target}."
            )
    return target


def load_existing_keys(manifest_path: Path) -> set[str]:
    seen: set[str] = set()
    if not manifest_path.exists():
        return seen

    with manifest_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            key = obj.get("uid") or obj.get("audio_filepath")
            if isinstance(key, str) and key:
                seen.add(key)
    return seen


def count_unique_in_manifest(manifest_path: Path) -> int:
    return len(load_existing_keys(manifest_path))


def assert_bucket_ready(
    bucket_name: str,
    manifest_path: Path,
    target: int,
    policy: UnderdeliveryPolicy,
) -> None:
    if not manifest_path.exists():
        handle_underdelivery(
            policy,
            f"Bucket '{bucket_name}': manifest missing at {manifest_path}",
        )
        return

    have = count_unique_in_manifest(manifest_path)
    if have < target:
        handle_underdelivery(policy, f"Bucket '{bucket_name}': have={have} target={target}")

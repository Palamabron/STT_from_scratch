import json
from pathlib import Path

import pytest

from prepare_data.buckets import (
    UnderdeliveryPolicy,
    assert_bucket_ready,
    group_by_name,
    handle_underdelivery,
    target_for_bucket,
)
from prepare_data.config import DatasetSpec


def _spec(name: str, samples: int) -> DatasetSpec:
    return DatasetSpec(
        name=name,
        hf_id="example/dataset",
        config_name="default",
        split="train",
        lang="en",
        samples=samples,
        text_col="text",
    )


def test_group_by_name_merges_shared_bucket() -> None:
    specs = [_spec("bucket_a", 10), _spec("bucket_a", 10), _spec("bucket_b", 5)]
    grouped = group_by_name(specs)
    assert set(grouped) == {"bucket_a", "bucket_b"}
    assert len(grouped["bucket_a"]) == 2


def test_target_for_bucket_uses_first_value() -> None:
    assert target_for_bucket([_spec("x", 7), _spec("x", 9)]) == 7


def test_handle_underdelivery_warn_does_not_raise() -> None:
    handle_underdelivery(UnderdeliveryPolicy.WARN, "missing samples")


def test_handle_underdelivery_raise() -> None:
    with pytest.raises(RuntimeError, match="missing samples"):
        handle_underdelivery(UnderdeliveryPolicy.RAISE, "missing samples")


def test_assert_bucket_ready_raises_for_missing_manifest(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="manifest missing"):
        assert_bucket_ready(
            "val_bucket",
            tmp_path / "missing.jsonl",
            target=10,
            policy=UnderdeliveryPolicy.RAISE,
        )


def test_assert_bucket_ready_counts_unique_entries(tmp_path: Path) -> None:
    manifest = tmp_path / "bucket.jsonl"
    manifest.write_text(
        "\n".join(json.dumps({"uid": f"u{i}", "audio_filepath": f"/tmp/{i}.wav"}) for i in range(3))
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="have=3 target=5"):
        assert_bucket_ready("bucket", manifest, target=5, policy=UnderdeliveryPolicy.RAISE)

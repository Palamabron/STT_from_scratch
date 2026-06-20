from __future__ import annotations

import json
from pathlib import Path

import pytest

from prepare_data.overlap import assert_no_overlap


def test_assert_no_overlap_raises_on_shared_uid(tmp_path: Path) -> None:
    train = tmp_path / "train.jsonl"
    val = tmp_path / "val.jsonl"
    row = {"uid": "shared", "audio_filepath": "/tmp/a.wav", "text": "hello"}
    train.write_text(json.dumps(row) + "\n", encoding="utf-8")
    val.write_text(json.dumps(row) + "\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="overlap detected"):
        assert_no_overlap(train, val)


def test_assert_no_overlap_passes_when_disjoint(tmp_path: Path) -> None:
    train = tmp_path / "train.jsonl"
    val = tmp_path / "val.jsonl"
    train.write_text(
        json.dumps({"uid": "a", "audio_filepath": "/tmp/a.wav"}) + "\n", encoding="utf-8"
    )
    val.write_text(
        json.dumps({"uid": "b", "audio_filepath": "/tmp/b.wav"}) + "\n", encoding="utf-8"
    )
    assert_no_overlap(train, val)

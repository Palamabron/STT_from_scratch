from __future__ import annotations

from typing import NotRequired, TypedDict

from .tensors import Tensor


class TrainBatch(TypedDict):
    features: Tensor
    feature_lengths: Tensor
    targets: Tensor
    target_lengths: Tensor
    language: NotRequired[list[str]]


class ValBatch(TrainBatch):
    text: list[str]

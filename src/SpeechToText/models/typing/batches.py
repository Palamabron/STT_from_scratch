from __future__ import annotations

from typing import NotRequired, TypedDict

from .tensors import Tensor


class TrainBatch(TypedDict):
    audio: Tensor
    audio_length: Tensor
    targets: Tensor
    target_length: Tensor
    clean_pass: NotRequired[Tensor]
    language: NotRequired[list[str]]
    dataset: NotRequired[list[str]]


class ValBatch(TrainBatch):
    text: list[str]
    duration: NotRequired[Tensor]

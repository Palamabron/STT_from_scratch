from __future__ import annotations

from typing import Any, TypeVar, cast

import torch
from loguru import logger

BatchT = TypeVar("BatchT")


def filter_batch_by_encoder_length(
    batch: BatchT,
    out_lengths: torch.Tensor,
    target_lengths: torch.Tensor,
) -> tuple[BatchT, torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """Drop utterances where encoder output length is shorter than the target.

    Returns a filtered batch and aligned length tensors, or ``None`` when no
    valid utterances remain.
    """
    valid = out_lengths >= target_lengths
    if not bool(valid.any()):
        return None
    if bool(valid.all()):
        return batch, out_lengths, target_lengths, valid

    indices = torch.nonzero(valid, as_tuple=False).view(-1).tolist()
    index_set = set(indices)
    batch_map = cast(dict[str, Any], batch)
    filtered = cast(BatchT, {})

    audio = batch_map["audio"]
    audio_lengths = batch_map["audio_length"]
    filtered_map = cast(dict[str, Any], filtered)
    filtered_map["audio"] = audio.index_select(0, torch.tensor(indices, device=audio.device))
    filtered_map["audio_length"] = audio_lengths.index_select(
        0, torch.tensor(indices, device=audio_lengths.device)
    )

    target_lengths_f = target_lengths.index_select(
        0, torch.tensor(indices, device=target_lengths.device)
    )
    out_lengths_f = out_lengths.index_select(0, torch.tensor(indices, device=out_lengths.device))

    targets_list: list[torch.Tensor] = []
    offset = 0
    for index in range(int(target_lengths.shape[0])):
        length = int(target_lengths[index].item())
        if index in index_set:
            targets_list.append(batch_map["targets"][offset : offset + length])
        offset += length
    filtered_map["targets"] = torch.cat(targets_list, dim=0)
    filtered_map["target_length"] = target_lengths_f

    if "language" in batch_map:
        filtered_map["language"] = [batch_map["language"][i] for i in indices]
    if "dataset" in batch_map:
        filtered_map["dataset"] = [batch_map["dataset"][i] for i in indices]
    if "text" in batch_map:
        filtered_map["text"] = [batch_map["text"][i] for i in indices]
    if "clean_pass" in batch_map:
        clean_pass = batch_map["clean_pass"]
        filtered_map["clean_pass"] = clean_pass.index_select(
            0, torch.tensor(indices, device=clean_pass.device)
        )

    return filtered, out_lengths_f, target_lengths_f, valid


def warn_empty_training_batch(batch_idx: int, batch_size: int) -> None:
    """Log when every utterance in a training batch was dropped by length filtering."""
    logger.warning(
        "Skipping training batch {} (size={}): all utterances have encoder length < target length",
        batch_idx,
        batch_size,
    )

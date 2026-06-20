from __future__ import annotations

import heapq
from dataclasses import dataclass

import torch
from jiwer import cer as jiwer_cer


def accumulate_blank_stats(
    log_probs: torch.Tensor,
    out_lengths: torch.Tensor,
    blank_id: int,
) -> tuple[int, int]:
    """Return ``(blank_count, total_frames)`` from greedy argmax predictions."""
    preds = torch.argmax(log_probs.detach(), dim=-1)
    lengths = out_lengths.detach().cpu()
    blank_count = 0
    total_frames = 0
    for seq, length in zip(preds.cpu(), lengths, strict=True):
        valid = seq[: int(length.item())]
        total_frames += int(valid.numel())
        blank_count += int((valid == blank_id).sum().item())
    return blank_count, total_frames


@dataclass(slots=True)
class ValUtteranceRecord:
    dataset: str
    language: str
    reference: str
    hypothesis: str
    cer: float
    audio: torch.Tensor


class WorstValExamplesCollector:
    """Track blank-collapse telemetry and retain the worst validation utterances."""

    def __init__(self, max_examples: int = 50) -> None:
        self.max_examples = max_examples
        self._heap: list[tuple[float, int, ValUtteranceRecord]] = []
        self._counter = 0
        self._blank_count = 0
        self._blank_total = 0

    def reset(self) -> None:
        self._heap.clear()
        self._counter = 0
        self._blank_count = 0
        self._blank_total = 0

    def accumulate_blank_stats(
        self,
        log_probs: torch.Tensor,
        out_lengths: torch.Tensor,
        blank_id: int,
    ) -> None:
        blank_count, total_frames = accumulate_blank_stats(log_probs, out_lengths, blank_id)
        self._blank_count += blank_count
        self._blank_total += total_frames

    def blank_fraction(self) -> float:
        return self._blank_count / max(self._blank_total, 1)

    def add(
        self,
        *,
        dataset: str,
        language: str,
        reference: str,
        hypothesis: str,
        audio: torch.Tensor,
    ) -> None:
        cer = float(jiwer_cer(reference, hypothesis))
        record = ValUtteranceRecord(
            dataset=dataset,
            language=language,
            reference=reference,
            hypothesis=hypothesis,
            cer=cer,
            audio=audio.detach().cpu(),
        )
        self._counter += 1
        item = (cer, self._counter, record)
        if len(self._heap) < self.max_examples:
            heapq.heappush(self._heap, item)
        elif cer > self._heap[0][0]:
            heapq.heapreplace(self._heap, item)

    def worst_first(self) -> list[ValUtteranceRecord]:
        return [record for _, _, record in sorted(self._heap, key=lambda x: (-x[0], x[1]))]


def log_wandb_worst_val_examples(
    logger: object | None,
    examples: list[ValUtteranceRecord],
    *,
    sample_rate: int,
    epoch: int,
    key: str = "val/worst_examples",
) -> None:
    if logger is None or not examples:
        return

    try:
        import wandb
        from lightning.pytorch.loggers import WandbLogger
    except ImportError:
        return

    if not isinstance(logger, WandbLogger):
        return

    table = wandb.Table(
        columns=["epoch", "dataset", "language", "reference", "hypothesis", "cer", "audio"],
    )
    for record in examples:
        table.add_data(
            epoch,
            record.dataset,
            record.language,
            record.reference,
            record.hypothesis,
            record.cer,
            wandb.Audio(record.audio.numpy(), sample_rate=sample_rate),
        )
    logger.experiment.log({key: table})

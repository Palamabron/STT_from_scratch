from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import torch
import tyro
from loguru import logger

_EPOCH_RE = re.compile(r"epoch=(\d+)")


def _epoch_from_checkpoint(path: Path) -> int | None:
    match = _EPOCH_RE.search(path.name)
    if match:
        return int(match.group(1))
    match = re.match(r"^(\d+)-", path.name)
    if match:
        return int(match.group(1))
    return None


def _list_epoch_checkpoints(checkpoint_dir: Path) -> list[tuple[int, Path]]:
    candidates: list[tuple[int, Path]] = []
    for path in sorted(checkpoint_dir.glob("*.ckpt")):
        if path.name == "last.ckpt":
            continue
        epoch = _epoch_from_checkpoint(path)
        if epoch is not None:
            candidates.append((epoch, path))
    candidates.sort(key=lambda item: item[0])
    return candidates


def _average_state_dicts(state_dicts: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    if not state_dicts:
        raise ValueError("No state dicts to average")
    keys = state_dicts[0].keys()
    averaged: dict[str, torch.Tensor] = {}
    for key in keys:
        tensors = [state[key] for state in state_dicts if key in state]
        if not tensors:
            continue
        first = tensors[0]
        if not first.is_floating_point():
            averaged[key] = first.clone()
            continue
        stacked = torch.stack([tensor.float() for tensor in tensors], dim=0)
        averaged[key] = stacked.mean(dim=0).to(dtype=first.dtype)
    return averaged


@dataclass(slots=True)
class AverageCheckpointsConfig:
    """Average the last N consecutive epoch checkpoints (SWA-style).

    Prefer averaging checkpoints from the final training epochs while the
    learning rate is flat or decaying slowly (Lightning ``use_swa`` or a manual
    SWA LR schedule). Do **not** cherry-pick distant "best WER" epochs — that
    often hurts because weights lie on different regions of the loss manifold.
    """

    checkpoint_dir: str
    output: str
    last_n: int = 5
    exclude_last: bool = True


def main(cfg: AverageCheckpointsConfig) -> None:
    checkpoint_dir = Path(cfg.checkpoint_dir)
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

    ranked = _list_epoch_checkpoints(checkpoint_dir)
    if not ranked:
        raise FileNotFoundError(f"No epoch checkpoints found in {checkpoint_dir}")

    if cfg.exclude_last and ranked[-1][1].name != "last.ckpt":
        pass

    selected = ranked[-cfg.last_n :] if cfg.last_n > 0 else ranked
    if len(selected) < 2:
        raise ValueError(
            f"Need at least 2 checkpoints to average; found {len(selected)} in {checkpoint_dir}",
        )

    logger.info(
        "Averaging last {} consecutive epoch checkpoints (SWA-style, not best-by-metric):",
        len(selected),
    )
    for epoch, path in selected:
        logger.info("  epoch {:03d}: {}", epoch, path.name)

    state_dicts: list[dict[str, torch.Tensor]] = []
    template: dict | None = None
    for _, path in selected:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        if template is None:
            template = ckpt
        state_dicts.append(ckpt["state_dict"])

    assert template is not None
    averaged = _average_state_dicts(state_dicts)
    template["state_dict"] = averaged
    template["epoch"] = selected[-1][0]

    output_path = Path(cfg.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(template, output_path)
    logger.info("Saved averaged checkpoint to {}", output_path)


if __name__ == "__main__":
    main(tyro.cli(AverageCheckpointsConfig))

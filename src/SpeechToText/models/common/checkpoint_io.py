from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import torch


def load_lightning_checkpoint(path: str | Path) -> dict[str, Any]:
    """Load a PyTorch Lightning checkpoint dict from disk."""
    return cast(dict[str, Any], torch.load(str(path), map_location="cpu", weights_only=False))

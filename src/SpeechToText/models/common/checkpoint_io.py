from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any, cast

import torch


def load_lightning_checkpoint(path: str | Path) -> dict[str, Any]:
    """Load a Lightning checkpoint that may reference ``__main__.TrainConfig``."""
    from SpeechToText.models.ctc.train import TrainConfig as CtcTrainConfig
    from SpeechToText.models.ctc_attention.train import TrainConfig as CtcAttnTrainConfig
    from SpeechToText.models.tdt.train import TrainConfig as TdtTrainConfig

    main_module = sys.modules.setdefault("__main__", types.ModuleType("__main__"))

    last_error: Exception | None = None
    for config_cls in (CtcTrainConfig, CtcAttnTrainConfig, TdtTrainConfig):
        main_module.TrainConfig = config_cls  # type: ignore[attr-defined]
        try:
            return cast(
                dict[str, Any],
                torch.load(str(path), map_location="cpu", weights_only=False),
            )
        except Exception as exc:
            last_error = exc

    if last_error is not None:
        raise last_error
    raise RuntimeError(f"Failed to load checkpoint: {path}")

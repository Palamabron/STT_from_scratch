from __future__ import annotations

from typing import NamedTuple

import torch


class TDTLosses(NamedTuple):
    total: torch.Tensor
    rnnt: torch.Tensor
    lsm: torch.Tensor

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch
import torch.nn as nn


@dataclass
class JointNetConfig:
    """Configuration for the RNN-T / TDT joint network."""

    vocab_size: int = 1025
    enc_d: int = 256
    pred_d: int = 256
    joint_d: int = 512
    use_tdt: bool = False
    num_duration_classes: int = 5


class JointNet(nn.Module):
    """Combine encoder and predictor states into vocabulary (and optional duration) logits."""

    def __init__(self, cfg: JointNetConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.enc_proj = nn.Linear(cfg.enc_d, cfg.joint_d)
        self.pred_proj = nn.Linear(cfg.pred_d, cfg.joint_d)
        self.out = nn.Linear(cfg.joint_d, cfg.vocab_size)
        self.duration_out: nn.Linear | None
        if cfg.use_tdt:
            self.duration_out = nn.Linear(cfg.joint_d, cfg.num_duration_classes)
        else:
            self.duration_out = None

    def _joint_hidden(self, enc: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
        enc_h = self.enc_proj(enc).unsqueeze(2)
        pred_h = self.pred_proj(pred).unsqueeze(1)
        return torch.tanh(enc_h + pred_h)

    def forward(
        self, enc: torch.Tensor, pred: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        hidden = self._joint_hidden(enc, pred)
        token_logits = cast(torch.Tensor, self.out(hidden))
        if self.duration_out is not None:
            return token_logits, cast(torch.Tensor, self.duration_out(hidden))
        return token_logits

    def forward_chunked(
        self,
        enc: torch.Tensor,
        pred: torch.Tensor,
        *,
        fused_batch_size: int | None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if fused_batch_size is None or fused_batch_size <= 0 or enc.size(0) <= fused_batch_size:
            return self.forward(enc, pred)

        token_chunks: list[torch.Tensor] = []
        duration_chunks: list[torch.Tensor] = []
        use_tdt = self.duration_out is not None

        for start in range(0, int(enc.size(0)), fused_batch_size):
            end = start + fused_batch_size
            out = self.forward(enc[start:end], pred[start:end])
            if use_tdt:
                assert isinstance(out, tuple)
                token_chunks.append(out[0])
                duration_chunks.append(out[1])
            else:
                assert isinstance(out, torch.Tensor)
                token_chunks.append(out)

        if use_tdt:
            return torch.cat(token_chunks, dim=0), torch.cat(duration_chunks, dim=0)
        return torch.cat(token_chunks, dim=0)

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from SpeechToText.models.conformer import FastConformerEncoder, FastConformerEncoderConfig
from SpeechToText.models.typing import CTCOutput


@dataclass
class FastConformerCTCConfig:
    encoder: FastConformerEncoderConfig = field(default_factory=FastConformerEncoderConfig)
    aux_interval: int = 4


class FastConformerCTC(nn.Module):
    def __init__(self, cfg: FastConformerCTCConfig, vocab_size: int, blank_id: int = 0) -> None:
        super().__init__()
        self.cfg = cfg
        self.blank_id = int(blank_id)

        self.encoder = FastConformerEncoder(cfg.encoder)
        self.proj = nn.Linear(cfg.encoder.d_model, vocab_size)

        self.aux_layers: list[int] = []
        if cfg.aux_interval > 0:
            for i in range(cfg.encoder.n_layers - 1):
                if (i + 1) % cfg.aux_interval == 0:
                    self.aux_layers.append(i)

        self.aux_projs = nn.ModuleList(
            [nn.Linear(cfg.encoder.d_model, vocab_size) for _ in self.aux_layers]
        )

    def forward(self, feats: torch.Tensor, feat_lengths: torch.Tensor) -> CTCOutput:
        if self.aux_layers:
            enc, out_lengths, layer_outs = self.encoder(
                feats, feat_lengths, return_layer_outputs=True
            )
        else:
            enc, out_lengths = self.encoder(feats, feat_lengths, return_layer_outputs=False)
            layer_outs = []

        logits = self.proj(enc)
        log_probs = F.log_softmax(logits, dim=-1)

        aux_log_probs: torch.Tensor
        if self.aux_layers:
            aux_encs = [layer_outs[i] for i in self.aux_layers]

            aux_logits = torch.stack(
                [proj(h) for proj, h in zip(self.aux_projs, aux_encs, strict=True)],
                dim=0,
            )
            aux_log_probs = F.log_softmax(aux_logits, dim=-1)
        else:
            aux_log_probs = torch.empty(0, device=log_probs.device, dtype=log_probs.dtype)

        return CTCOutput(log_probs=log_probs, out_lengths=out_lengths, aux_log_probs=aux_log_probs)

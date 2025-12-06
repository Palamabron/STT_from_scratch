from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torchaudio.models import Conformer


@dataclass
class FastConformerCTCConfig:
    sample_rate: int = 16_000
    n_layers: int = 8
    d_model: int = 256
    num_heads: int = 8
    ff_expansion_factor: int = 4
    conv_kernel_size: int = 9
    features: int = 80
    dropout: float = 0.15
    stride: int = 4
    conv_channels: int = 64


class FastConformerCTC(nn.Module):
    def __init__(
        self,
        cfg: FastConformerCTCConfig,
        vocab_size: int,
        blank_id: int = 0,
    ):
        super().__init__()
        self.cfg = cfg
        self.blank_id = blank_id

        self.conv_downsample = nn.Sequential(
            nn.Conv1d(
                in_channels=cfg.features,
                out_channels=cfg.conv_channels,
                kernel_size=11,
                stride=cfg.stride,
                padding=5,
            ),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
        )

        self.input_proj = nn.Linear(cfg.conv_channels, cfg.d_model)

        self.encoder = Conformer(
            input_dim=cfg.d_model,
            num_heads=cfg.num_heads,
            ffn_dim=cfg.d_model * cfg.ff_expansion_factor,
            num_layers=cfg.n_layers,
            depthwise_conv_kernel_size=cfg.conv_kernel_size,
            dropout=cfg.dropout,
        )

        self.proj = nn.Linear(cfg.d_model, vocab_size)
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(
        self,
        feats: torch.Tensor,
        feat_lengths: torch.Tensor,
    ):
        """
        feats: (B, T, F)
        feat_lengths: (B,)
        """
        x = feats.transpose(1, 2)  # (B, F, T)
        x = self.conv_downsample(x)
        x = x.transpose(1, 2)  # (B, T', C)

        stride = self.cfg.stride
        out_lengths = (feat_lengths + stride - 1) // stride

        x = self.input_proj(x)  # (B, T', d_model)
        enc_out, _ = self.encoder(x, out_lengths)  # (B, T', d_model)

        logits = self.proj(enc_out)  # (B, T', vocab)
        log_probs = self.log_softmax(logits)

        return log_probs, out_lengths

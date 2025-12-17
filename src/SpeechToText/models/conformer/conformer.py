from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from .drop_path import DropPath
from .subsampling import ConvSubsampling4, ConvSubsamplingConfig


@dataclass
class FastConformerEncoderConfig:
    in_feats: int = 80
    d_model: int = 256
    n_layers: int = 16
    n_heads: int = 4
    ffn_mult: int = 4
    dropout: float = 0.1
    conv_kernel: int = 31
    drop_path_prob: float = 0.1


def _lengths_to_attn_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    ids = torch.arange(max_len, device=lengths.device).unsqueeze(0)
    return ids >= lengths.unsqueeze(1)


class ConformerConvModule(nn.Module):
    def __init__(self, d_model: int, kernel: int, dropout: float) -> None:
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        self.pw1 = nn.Conv1d(d_model, 2 * d_model, kernel_size=1)
        self.dw = nn.Conv1d(
            d_model, d_model, kernel_size=kernel, padding=kernel // 2, groups=d_model
        )
        self.bn = nn.BatchNorm1d(d_model)
        self.pw2 = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        x = self.ln(x)
        x = x.transpose(1, 2)  # [B, D, T]
        x = self.pw1(x)
        a, b = cast(tuple[torch.Tensor, torch.Tensor], x.chunk(2, dim=1))
        x = a * torch.sigmoid(b)  # GLU
        x = self.dw(x)
        x = self.bn(x)
        x = F.silu(x)
        x = self.pw2(x)
        x = self.dropout(x)
        return cast(torch.Tensor, x.transpose(1, 2))  # [B, T, D]


class ConformerFFN(nn.Module):
    def __init__(self, d_model: int, mult: int, dropout: float) -> None:
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, mult * d_model),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(mult * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.ff(self.ln(x)))


class ConformerBlock(nn.Module):
    def __init__(
        self,
        *,
        d_model: int,
        n_heads: int,
        ffn_mult: int,
        dropout: float,
        conv_kernel: int,
        drop_path: float,
    ) -> None:
        super().__init__()
        self.ff1 = ConformerFFN(d_model, ffn_mult, dropout)
        self.ff2 = ConformerFFN(d_model, ffn_mult, dropout)

        self.mha_ln = nn.LayerNorm(d_model)
        self.mha = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.conv = ConformerConvModule(d_model, conv_kernel, dropout)
        self.final_ln = nn.LayerNorm(d_model)

        self.dp = DropPath(drop_path)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None) -> torch.Tensor:
        x = x + 0.5 * self.dp(self.ff1(x))

        y = self.mha_ln(x)
        y, _ = self.mha(y, y, y, key_padding_mask=key_padding_mask, need_weights=False)
        x = x + self.dp(y)

        x = x + self.dp(self.conv(x))

        x = x + 0.5 * self.dp(self.ff2(x))
        return cast(torch.Tensor, self.final_ln(x))


class FastConformerEncoder(nn.Module):
    def __init__(self, cfg: FastConformerEncoderConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.sub = ConvSubsampling4(
            ConvSubsamplingConfig(in_feats=cfg.in_feats, d_model=cfg.d_model, dropout=cfg.dropout)
        )
        self.in_ln = nn.LayerNorm(cfg.d_model)
        self.in_drop = nn.Dropout(cfg.dropout)

        # linear schedule drop_path across depth
        if cfg.n_layers <= 1:
            dp = [cfg.drop_path_prob]
        else:
            dp = [cfg.drop_path_prob * (i / (cfg.n_layers - 1)) for i in range(cfg.n_layers)]

        self.blocks = nn.ModuleList(
            [
                ConformerBlock(
                    d_model=cfg.d_model,
                    n_heads=cfg.n_heads,
                    ffn_mult=cfg.ffn_mult,
                    dropout=cfg.dropout,
                    conv_kernel=cfg.conv_kernel,
                    drop_path=dp[i],
                )
                for i in range(cfg.n_layers)
            ]
        )

    def forward(
        self, feats: torch.Tensor, feat_lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x, out_lengths = self.sub(feats, feat_lengths)
        x = self.in_drop(self.in_ln(x))

        kpm = _lengths_to_attn_mask(out_lengths, max_len=x.size(1))
        for blk in self.blocks:
            x = blk(x, key_padding_mask=kpm)
        return x, out_lengths

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from .drop_path import DropPath
from .subsampling import (
    ConvSubsampling2,
    ConvSubsampling4,
    ConvSubsampling8,
    ConvSubsamplingConfig,
)


@dataclass
class FastConformerEncoderConfig:
    """Hyper-parameters for the Fast-Conformer encoder stack."""

    in_feats: int = 80
    d_model: int = 256
    n_layers: int = 12
    n_heads: int = 4
    ffn_mult: int = 4
    dropout: float = 0.1
    conv_kernel: int = 31
    drop_path_prob: float = 0.1
    subsampling_factor: int = 8


def _lengths_to_attn_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    """Return a padding mask where ``True`` marks padded positions."""
    ids = torch.arange(max_len, device=lengths.device).unsqueeze(0)
    return ids >= lengths.unsqueeze(1)


class ConformerConvModule(nn.Module):
    """Depthwise separable convolution block used inside a Conformer layer."""

    def __init__(self, d_model: int, kernel: int, dropout: float) -> None:
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        self.pw1 = nn.Conv1d(d_model, 2 * d_model, kernel_size=1, bias=False)
        self.dw = nn.Conv1d(
            d_model, d_model, kernel_size=kernel, padding=kernel // 2, groups=d_model, bias=False
        )
        self.bn = nn.BatchNorm1d(d_model)
        self.pw2 = nn.Conv1d(d_model, d_model, kernel_size=1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ln(x)
        x = x.transpose(1, 2)
        x = self.pw1(x)
        a, b = cast(tuple[torch.Tensor, torch.Tensor], x.chunk(2, dim=1))
        x = a * torch.sigmoid(b)
        x = self.dw(x)
        x = self.bn(x)
        x = F.silu(x)
        x = self.pw2(x)
        x = self.dropout(x)
        return cast(torch.Tensor, x.transpose(1, 2))


class ConformerFFN(nn.Module):
    """Feed-forward sub-layer with pre-normalization."""

    def __init__(self, d_model: int, mult: int, dropout: float) -> None:
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, mult * d_model, bias=False),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(mult * d_model, d_model, bias=False),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.ff(self.ln(x)))


class ConformerSelfAttention(nn.Module):
    """Multi-head self-attention with scaled dot-product attention."""

    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None) -> torch.Tensor:
        batch, time, channels = x.shape
        qkv = (
            self.qkv(x).reshape(batch, time, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn_mask = None
        if key_padding_mask is not None:
            attn_mask = ~key_padding_mask.view(batch, 1, 1, time)

        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
        )
        y = y.transpose(1, 2).contiguous().view(batch, time, channels)
        return cast(torch.Tensor, self.out_proj(y))


class ConformerBlock(nn.Module):
    """Single Fast-Conformer block: FFN → attention → conv → FFN."""

    def __init__(
        self,
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
        self.mha = ConformerSelfAttention(d_model, n_heads, dropout)
        self.conv = ConformerConvModule(d_model, conv_kernel, dropout)
        self.final_ln = nn.LayerNorm(d_model)
        self.dp = DropPath(drop_path)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None) -> torch.Tensor:
        x = x + 0.5 * self.dp(self.ff1(x))
        y = self.mha(self.mha_ln(x), key_padding_mask=key_padding_mask)
        x = x + self.dp(y)
        x = x + self.dp(self.conv(x))
        x = x + 0.5 * self.dp(self.ff2(x))
        return cast(torch.Tensor, self.final_ln(x))


class FastConformerEncoder(nn.Module):
    """Fast-Conformer encoder with configurable convolutional subsampling."""

    def __init__(self, cfg: FastConformerEncoderConfig) -> None:
        super().__init__()
        self.cfg = cfg

        sub_cfg = ConvSubsamplingConfig(
            in_feats=cfg.in_feats, d_model=cfg.d_model, dropout=cfg.dropout
        )
        subsampling_modules = {
            2: ConvSubsampling2,
            4: ConvSubsampling4,
            8: ConvSubsampling8,
        }
        subsampling_cls = subsampling_modules.get(cfg.subsampling_factor)
        if subsampling_cls is None:
            raise ValueError(f"Unsupported subsampling_factor: {cfg.subsampling_factor}")
        self.sub = subsampling_cls(sub_cfg)

        self.in_ln = nn.LayerNorm(cfg.d_model)
        self.in_drop = nn.Dropout(cfg.dropout)

        drop_path_rates = (
            [cfg.drop_path_prob * (i / (cfg.n_layers - 1)) for i in range(cfg.n_layers)]
            if cfg.n_layers > 1
            else [0.0]
        )

        self.blocks = nn.ModuleList(
            [
                ConformerBlock(
                    d_model=cfg.d_model,
                    n_heads=cfg.n_heads,
                    ffn_mult=cfg.ffn_mult,
                    dropout=cfg.dropout,
                    conv_kernel=cfg.conv_kernel,
                    drop_path=drop_path_rates[i],
                )
                for i in range(cfg.n_layers)
            ]
        )

    def forward(
        self, feats: torch.Tensor, feat_lengths: torch.Tensor, return_layer_outputs: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        """Encode log-mel features and return encoded states with output lengths."""
        x, out_lengths = self.sub(feats, feat_lengths)
        x = self.in_drop(self.in_ln(x))
        key_padding_mask = _lengths_to_attn_mask(out_lengths, max_len=x.size(1))

        layer_outs = []
        for block in self.blocks:
            x = block(x, key_padding_mask=key_padding_mask)
            if return_layer_outputs:
                layer_outs.append(x)

        if return_layer_outputs:
            return x, out_lengths, layer_outs
        return x, out_lengths

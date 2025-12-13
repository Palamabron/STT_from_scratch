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
    aux_interval: int = 4


class FastConformerCTC(nn.Module):
    def __init__(
        self,
        cfg: FastConformerCTCConfig,
        vocab_size: int,
        blank_id: int = 0,
    ) -> None:
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

        aux_interval = max(1, cfg.aux_interval)
        aux_layer_indices: list[int] = []
        for i in range(cfg.n_layers - 1):
            if (i + 1) % aux_interval == 0:
                aux_layer_indices.append(i)
        self.aux_layer_indices = aux_layer_indices
        self.aux_projs = nn.ModuleList(
            [nn.Linear(cfg.d_model, vocab_size) for _ in aux_layer_indices]
        )

        self.log_softmax = nn.LogSoftmax(dim=-1)

    @staticmethod
    def _lengths_to_padding_mask(lengths: torch.Tensor) -> torch.Tensor:
        batch_size = lengths.shape[0]
        max_length = int(lengths.max().item())
        return torch.arange(max_length, device=lengths.device).expand(
            batch_size, max_length
        ) >= lengths.unsqueeze(1)

    def forward(
        self,
        feats: torch.Tensor,
        feat_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            feats: float tensor of shape (B, T, F)
            feat_lengths: int tensor of shape (B,)
        Returns:
            log_probs: (B, T', V)
            out_lengths: (B,)
            aux_log_probs: (N_aux, B, T', V) or empty tensor if no aux heads
        """
        x = feats.transpose(1, 2)
        x = self.conv_downsample(x)
        x = x.transpose(1, 2)

        stride = self.cfg.stride
        out_lengths = (feat_lengths + stride - 1) // stride

        x = self.input_proj(x)

        encoder_padding_mask = self._lengths_to_padding_mask(out_lengths)
        x_t = x.transpose(0, 1)

        aux_logits_list: list[torch.Tensor] = []
        layer_to_head = {layer_i: head_i for head_i, layer_i in enumerate(self.aux_layer_indices)}

        for layer_idx, layer in enumerate(self.encoder.conformer_layers):
            x_t = layer(x_t, encoder_padding_mask)
            head_idx = layer_to_head.get(layer_idx)
            if head_idx is not None:
                h = x_t.transpose(0, 1)  # (B, T', D)
                aux_logits_list.append(self.aux_projs[head_idx](h))  # (B, T', V)

        enc_out = x_t.transpose(0, 1)

        logits = self.proj(enc_out)
        log_probs = self.log_softmax(logits)

        if aux_logits_list:
            aux_logits = torch.stack(aux_logits_list, dim=0)  # (N_aux, B, T', V)
            aux_log_probs = self.log_softmax(aux_logits)
        else:
            V = logits.size(-1)
            aux_log_probs = torch.empty(
                0,
                logits.size(0),
                logits.size(1),
                V,
                device=logits.device,
                dtype=logits.dtype,
            )

        return log_probs, out_lengths, aux_log_probs

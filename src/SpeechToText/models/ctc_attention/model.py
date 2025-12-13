from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchaudio.models import Conformer

from ..ctc.model import FastConformerCTCConfig


@dataclass
class AttentionDecoderConfig:
    num_layers: int = 4
    num_heads: int = 4
    ff_expansion_factor: int = 4
    dropout: float = 0.15
    max_len: int = 256


class FastConformerCTCAttention(nn.Module):
    def __init__(
        self,
        enc_cfg: FastConformerCTCConfig,
        ctc_vocab_size: int,
        sp_vocab_size: int,
        dec_cfg: AttentionDecoderConfig | None = None,
        blank_id: int = 0,
    ) -> None:
        super().__init__()
        if dec_cfg is None:
            dec_cfg = AttentionDecoderConfig(
                num_layers=4,
                num_heads=enc_cfg.num_heads,
                ff_expansion_factor=4,
                dropout=enc_cfg.dropout,
                max_len=256,
            )

        self.enc_cfg = enc_cfg
        self.dec_cfg = dec_cfg
        self.blank_id = blank_id

        self.conv_downsample = nn.Sequential(
            nn.Conv1d(
                in_channels=enc_cfg.features,
                out_channels=enc_cfg.conv_channels,
                kernel_size=11,
                stride=enc_cfg.stride,
                padding=5,
            ),
            nn.ReLU(),
            nn.Dropout(enc_cfg.dropout),
        )
        self.input_proj = nn.Linear(enc_cfg.conv_channels, enc_cfg.d_model)

        self.encoder = Conformer(
            input_dim=enc_cfg.d_model,
            num_heads=enc_cfg.num_heads,
            ffn_dim=enc_cfg.d_model * enc_cfg.ff_expansion_factor,
            num_layers=enc_cfg.n_layers,
            depthwise_conv_kernel_size=enc_cfg.conv_kernel_size,
            dropout=enc_cfg.dropout,
        )

        self.ctc_proj = nn.Linear(enc_cfg.d_model, ctc_vocab_size)

        aux_interval = max(1, enc_cfg.aux_interval)
        aux_layer_indices: list[int] = []
        for i in range(enc_cfg.n_layers - 1):
            if (i + 1) % aux_interval == 0:
                aux_layer_indices.append(i)
        self.aux_layer_indices = aux_layer_indices
        self.aux_projs = nn.ModuleList(
            [nn.Linear(enc_cfg.d_model, ctc_vocab_size) for _ in aux_layer_indices]
        )

        self.sp_vocab_size = sp_vocab_size
        self.pad_id = sp_vocab_size
        self.bos_id = sp_vocab_size + 1
        self.eos_id = sp_vocab_size + 2
        self.decoder_vocab_size = sp_vocab_size + 3

        d_model = enc_cfg.d_model
        self.tok_embed = nn.Embedding(self.decoder_vocab_size, d_model)
        self.pos_embed = nn.Embedding(dec_cfg.max_len, d_model)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=dec_cfg.num_heads,
            dim_feedforward=d_model * dec_cfg.ff_expansion_factor,
            dropout=dec_cfg.dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=dec_cfg.num_layers,
        )

        self.decoder_proj = nn.Linear(d_model, self.decoder_vocab_size)

    @staticmethod
    def _lengths_to_padding_mask(lengths: torch.Tensor) -> torch.Tensor:
        batch_size = lengths.shape[0]
        max_length = int(lengths.max().item())
        return torch.arange(max_length, device=lengths.device).expand(
            batch_size, max_length
        ) >= lengths.unsqueeze(1)

    @staticmethod
    def _generate_square_subsequent_mask(sz: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones(sz, sz, device=device, dtype=torch.bool), diagonal=1)

    def _encode(
        self,
        feats: torch.Tensor,
        feat_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = feats.transpose(1, 2)
        x = self.conv_downsample(x)
        x = x.transpose(1, 2)

        stride = self.enc_cfg.stride
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
                aux_logits_list.append(self.aux_projs[head_idx](h))  # (B, T', V_ctc)

        enc_out = x_t.transpose(0, 1)

        ctc_logits = self.ctc_proj(enc_out)
        ctc_log_probs = F.log_softmax(ctc_logits, dim=-1)

        if aux_logits_list:
            aux_logits = torch.stack(aux_logits_list, dim=0)  # (N_aux, B, T', V_ctc)
            aux_log_probs = F.log_softmax(aux_logits, dim=-1)
        else:
            V = ctc_logits.size(-1)
            aux_log_probs = torch.empty(
                0,
                ctc_logits.size(0),
                ctc_logits.size(1),
                V,
                device=ctc_logits.device,
                dtype=ctc_logits.dtype,
            )

        return enc_out, out_lengths, ctc_log_probs, aux_log_probs

    def _decode(
        self,
        enc_out: torch.Tensor,
        out_lengths: torch.Tensor,
        decoder_input: torch.Tensor,
    ) -> torch.Tensor:
        B, L = decoder_input.shape
        device = decoder_input.device

        positions = torch.arange(L, device=device).unsqueeze(0).expand(B, L)
        tgt = self.tok_embed(decoder_input) + self.pos_embed(positions)

        tgt_key_padding_mask = decoder_input.eq(self.pad_id)
        tgt_mask = self._generate_square_subsequent_mask(L, device=device)

        max_T = enc_out.size(1)
        enc_pad_mask = torch.arange(max_T, device=device).unsqueeze(0) >= out_lengths.unsqueeze(1)

        decoded = self.decoder(
            tgt=tgt,
            memory=enc_out,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=enc_pad_mask,
        )
        logits = self.decoder_proj(decoded)
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs

    def forward(
        self,
        feats: torch.Tensor,
        feat_lengths: torch.Tensor,
        decoder_input: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """
        Returns CTC and optional attention decoder log-probabilities.
        """
        enc_out, out_lengths, ctc_log_probs, aux_log_probs = self._encode(feats, feat_lengths)

        if decoder_input is not None:
            dec_log_probs = self._decode(enc_out, out_lengths, decoder_input)
        else:
            dec_log_probs = None

        return ctc_log_probs, out_lengths, aux_log_probs, dec_log_probs

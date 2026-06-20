from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from SpeechToText.models.common.aux_layers import resolve_aux_layer_indices
from SpeechToText.models.conformer import FastConformerEncoder, FastConformerEncoderConfig
from SpeechToText.models.typing import CTCAttnOutput


@dataclass
class AttentionDecoderConfig:
    """Transformer decoder hyper-parameters."""

    num_layers: int = 4
    num_heads: int = 4
    ffn_mult: int = 4
    dropout: float = 0.1
    max_len: int = 256


@dataclass
class FastConformerCTCAttentionConfig:
    """Model configuration for the CTC + attention hybrid."""

    encoder: FastConformerEncoderConfig = field(default_factory=FastConformerEncoderConfig)
    aux_interval: int = 0
    aux_layer: int | None = None
    decoder: AttentionDecoderConfig = field(default_factory=AttentionDecoderConfig)


class FastConformerCTCAttention(nn.Module):
    """Fast-Conformer encoder with CTC, auxiliary CTC, and attention heads."""

    def __init__(
        self,
        cfg: FastConformerCTCAttentionConfig,
        *,
        vocab_size: int,
        blank_id: int,
        pad_id: int,
        bos_id: int,
        eos_id: int,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.vocab_size = int(vocab_size)
        self.blank_id = int(blank_id)
        self.pad_id = int(pad_id)
        self.bos_id = int(bos_id)
        self.eos_id = int(eos_id)

        self.encoder = FastConformerEncoder(cfg.encoder)
        self.ctc_proj = nn.Linear(cfg.encoder.d_model, self.vocab_size)

        self.aux_layers: list[int] = resolve_aux_layer_indices(
            n_layers=cfg.encoder.n_layers,
            aux_interval=cfg.aux_interval,
            aux_layer=cfg.aux_layer,
        )

        self.aux_projs = nn.ModuleList(
            [nn.Linear(cfg.encoder.d_model, self.vocab_size) for _ in self.aux_layers]
        )

        decoder_cfg = cfg.decoder
        self.tok_embed = nn.Embedding(self.vocab_size, cfg.encoder.d_model)
        self.pos_embed = nn.Embedding(decoder_cfg.max_len, cfg.encoder.d_model)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=cfg.encoder.d_model,
            nhead=decoder_cfg.num_heads,
            dim_feedforward=cfg.encoder.d_model * decoder_cfg.ffn_mult,
            dropout=decoder_cfg.dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=decoder_cfg.num_layers)
        self.dec_proj = nn.Linear(cfg.encoder.d_model, self.vocab_size)

    @staticmethod
    def _square_subsequent_mask(size: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones(size, size, device=device, dtype=torch.bool), diagonal=1)

    @staticmethod
    def _lengths_to_kpm(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
        ids = torch.arange(max_len, device=lengths.device).unsqueeze(0)
        return ids >= lengths.unsqueeze(1)

    def encode(
        self, feats: torch.Tensor, feat_lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the encoder and collect auxiliary CTC logits when configured."""
        x, out_lengths = self.encoder.sub(feats, feat_lengths)
        x = self.encoder.in_drop(self.encoder.in_ln(x))
        key_padding_mask = self._lengths_to_kpm(out_lengths, x.size(1))

        aux_logits_list: list[torch.Tensor] = []
        layer_to_head = {
            layer_index: head_index for head_index, layer_index in enumerate(self.aux_layers)
        }

        for layer_index, block in enumerate(self.encoder.blocks):
            x = block(x, key_padding_mask=key_padding_mask)
            head_index = layer_to_head.get(layer_index)
            if head_index is not None:
                aux_logits_list.append(self.aux_projs[head_index](x))

        if aux_logits_list:
            aux_logits = torch.stack(aux_logits_list, dim=0)
            aux_log_probs = F.log_softmax(aux_logits, dim=-1)
        else:
            logits = self.ctc_proj(x)
            aux_log_probs = torch.empty(
                (0, logits.size(0), logits.size(1), logits.size(2)),
                device=logits.device,
                dtype=logits.dtype,
            )

        return x, out_lengths, aux_log_probs

    def decode(
        self,
        enc: torch.Tensor,
        out_lengths: torch.Tensor,
        decoder_input: torch.Tensor,
    ) -> torch.Tensor:
        """Run the autoregressive attention decoder."""
        batch_size, decoder_len = decoder_input.shape
        if decoder_len > self.cfg.decoder.max_len:
            raise ValueError(
                f"decoder_input length {decoder_len} > decoder.max_len={self.cfg.decoder.max_len}"
            )

        device = decoder_input.device
        positions = (
            torch.arange(decoder_len, device=device).unsqueeze(0).expand(batch_size, decoder_len)
        )

        target = self.tok_embed(decoder_input) + self.pos_embed(positions)
        target_key_padding_mask = decoder_input.eq(self.pad_id)
        target_mask = self._square_subsequent_mask(decoder_len, device=device)
        memory_key_padding_mask = self._lengths_to_kpm(out_lengths, enc.size(1))

        decoded = self.decoder(
            tgt=target,
            memory=enc,
            tgt_mask=target_mask,
            tgt_key_padding_mask=target_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )

        logits = self.dec_proj(decoded)
        return F.log_softmax(logits, dim=-1)

    def forward(
        self,
        feats: torch.Tensor,
        feat_lengths: torch.Tensor,
        decoder_input: torch.Tensor | None = None,
    ) -> CTCAttnOutput:
        enc, out_lengths, aux_log_probs = self.encode(feats, feat_lengths)

        ctc_logits = self.ctc_proj(enc)
        ctc_log_probs = F.log_softmax(ctc_logits, dim=-1)

        dec_log_probs = (
            self.decode(enc, out_lengths, decoder_input) if decoder_input is not None else None
        )

        return CTCAttnOutput(
            ctc_log_probs=ctc_log_probs,
            out_lengths=out_lengths,
            aux_log_probs=aux_log_probs,
            dec_log_probs=dec_log_probs,
        )

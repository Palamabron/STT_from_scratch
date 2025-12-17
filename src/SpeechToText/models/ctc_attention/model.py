from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from SpeechToText.models.conformer import FastConformerEncoder, FastConformerEncoderConfig
from SpeechToText.models.typing import CTCAttnOutput


@dataclass
class AttentionDecoderConfig:
    num_layers: int = 4
    num_heads: int = 4
    ffn_mult: int = 4
    dropout: float = 0.1
    max_len: int = 256


@dataclass
class FastConformerCTCAttentionConfig:
    encoder: FastConformerEncoderConfig = field(default_factory=FastConformerEncoderConfig)
    aux_interval: int = 4
    decoder: AttentionDecoderConfig = field(default_factory=AttentionDecoderConfig)


class FastConformerCTCAttention(nn.Module):
    """
    Encoder: FastConformerEncoder
    Heads:
      - CTC head (blank + SP pieces): vocab_size_ctc = sp_vocab_size + 1
      - Optional aux CTC heads (every aux_interval blocks)
      - Autoregressive TransformerDecoder over SentencePiece + {pad,bos,eos}:
          vocab_size_dec = sp_vocab_size + 3
    """

    def __init__(
        self,
        cfg: FastConformerCTCAttentionConfig,
        *,
        ctc_vocab_size: int,
        sp_vocab_size: int,
        blank_id: int = 0,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.blank_id = int(blank_id)

        self.sp_vocab_size = int(sp_vocab_size)
        self.pad_id = self.sp_vocab_size
        self.bos_id = self.sp_vocab_size + 1
        self.eos_id = self.sp_vocab_size + 2
        self.dec_vocab_size = self.sp_vocab_size + 3

        self.encoder = FastConformerEncoder(cfg.encoder)

        self.ctc_proj = nn.Linear(cfg.encoder.d_model, ctc_vocab_size)

        # Aux heads: capture encoder states after selected blocks.
        self.aux_layers: list[int] = []
        if cfg.aux_interval > 0:
            for i in range(cfg.encoder.n_layers - 1):
                if (i + 1) % cfg.aux_interval == 0:
                    self.aux_layers.append(i)
        self.aux_projs = nn.ModuleList(
            [nn.Linear(cfg.encoder.d_model, ctc_vocab_size) for _ in self.aux_layers]
        )

        # Decoder
        d_cfg = cfg.decoder
        self.tok_embed = nn.Embedding(self.dec_vocab_size, cfg.encoder.d_model)
        self.pos_embed = nn.Embedding(d_cfg.max_len, cfg.encoder.d_model)

        layer = nn.TransformerDecoderLayer(
            d_model=cfg.encoder.d_model,
            nhead=d_cfg.num_heads,
            dim_feedforward=cfg.encoder.d_model * d_cfg.ffn_mult,
            dropout=d_cfg.dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=d_cfg.num_layers)
        self.dec_proj = nn.Linear(cfg.encoder.d_model, self.dec_vocab_size)

    @staticmethod
    def _square_subsequent_mask(sz: int, device: torch.device) -> torch.Tensor:
        # True where positions are masked (upper triangular without diagonal)
        return torch.triu(torch.ones(sz, sz, device=device, dtype=torch.bool), diagonal=1)

    @staticmethod
    def _lengths_to_kpm(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
        ids = torch.arange(max_len, device=lengths.device).unsqueeze(0)
        return ids >= lengths.unsqueeze(1)

    def encode(
        self, feats: torch.Tensor, feat_lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
          enc: [B,T,D]
          out_lengths: [B]
          aux_log_probs: [Naux,B,T,V] or empty
        """
        # We need aux states, so we replicate the encoder forward with a hook into blocks.
        x, out_lengths = self.encoder.sub(feats, feat_lengths)
        x = self.encoder.in_drop(self.encoder.in_ln(x))
        kpm = self._lengths_to_kpm(out_lengths, x.size(1))

        aux_logits_list: list[torch.Tensor] = []
        layer_to_head = {layer_i: head_i for head_i, layer_i in enumerate(self.aux_layers)}

        for layer_idx, blk in enumerate(self.encoder.blocks):
            x = blk(x, key_padding_mask=kpm)
            head_idx = layer_to_head.get(layer_idx)
            if head_idx is not None:
                aux_logits_list.append(self.aux_projs[head_idx](x))  # [B,T,V]

        if aux_logits_list:
            aux_logits = torch.stack(aux_logits_list, dim=0)  # [Naux,B,T,V]
            aux_log_probs = F.log_softmax(aux_logits, dim=-1)
        else:
            # shape kept consistent with your previous code: [0,B,T,V]
            logits = self.ctc_proj(x)
            aux_log_probs = torch.empty(
                (0, logits.size(0), logits.size(1), logits.size(2)),
                device=logits.device,
                dtype=logits.dtype,
            )

        return x, out_lengths, aux_log_probs

    def decode(
        self,
        enc: torch.Tensor,  # [B,T,D]
        out_lengths: torch.Tensor,  # [B]
        decoder_input: torch.Tensor,  # [B,U]
    ) -> torch.Tensor:
        b, u = decoder_input.shape
        if u > self.cfg.decoder.max_len:
            raise ValueError(
                f"decoder_input length {u} > decoder.max_len={self.cfg.decoder.max_len}"
            )

        device = decoder_input.device
        pos = torch.arange(u, device=device).unsqueeze(0).expand(b, u)

        tgt = self.tok_embed(decoder_input) + self.pos_embed(pos)
        tgt_key_padding_mask = decoder_input.eq(self.pad_id)
        tgt_mask = self._square_subsequent_mask(u, device=device)

        mem_key_padding_mask = self._lengths_to_kpm(out_lengths, enc.size(1))

        dec = self.decoder(
            tgt=tgt,
            memory=enc,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=mem_key_padding_mask,
        )

        logits = self.dec_proj(dec)
        return F.log_softmax(logits, dim=-1)  # [B,U,V_dec]

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

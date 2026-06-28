from __future__ import annotations

import torch
import torch.nn as nn

from SpeechToText.models.common.aux_layers import resolve_aux_layer_indices
from SpeechToText.models.conformer import FastConformerEncoder
from SpeechToText.models.shared.config import SharedASRConfig
from SpeechToText.models.tdt.decoder import TDTDecoder
from SpeechToText.models.tdt.joint import JointNet


class SharedFastConformerASR(nn.Module):
    """Multi-Head ASR Model with a Shared FastConformer Encoder."""

    def __init__(
        self,
        cfg: SharedASRConfig,
        *,
        vocab_size: int,
        blank_id: int,
        pad_id: int,
        bos_id: int,
        eos_id: int,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.vocab_size = vocab_size
        self.blank_id = blank_id
        self.pad_id = pad_id
        self.bos_id = bos_id
        self.eos_id = eos_id

        self.encoder = FastConformerEncoder(cfg.encoder)

        self.aux_layers: list[int] = []
        if "ctc" in cfg.active_heads:
            self.ctc_proj = nn.Linear(cfg.encoder.d_model, self.vocab_size)
            self.aux_layers = resolve_aux_layer_indices(
                n_layers=cfg.encoder.n_layers,
                aux_interval=cfg.aux_interval,
                aux_layer=cfg.aux_layer,
            )
            self.aux_projs = nn.ModuleList(
                [nn.Linear(cfg.encoder.d_model, self.vocab_size) for _ in self.aux_layers]
            )

        if "attn" in cfg.active_heads:
            attn_cfg = cfg.attn_decoder
            self.tok_embed = nn.Embedding(self.vocab_size, cfg.encoder.d_model)
            self.pos_embed = nn.Embedding(attn_cfg.max_len, cfg.encoder.d_model)

            decoder_layer = nn.TransformerDecoderLayer(
                d_model=cfg.encoder.d_model,
                nhead=attn_cfg.num_heads,
                dim_feedforward=cfg.encoder.d_model * attn_cfg.ffn_mult,
                dropout=attn_cfg.dropout,
                batch_first=True,
            )
            self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=attn_cfg.num_layers)
            self.dec_proj = nn.Linear(cfg.encoder.d_model, self.vocab_size)

        if "tdt" in cfg.active_heads:
            cfg.tdt_decoder.vocab_size = self.vocab_size
            cfg.tdt_joint.vocab_size = self.vocab_size
            cfg.tdt_decoder.d_model = cfg.encoder.d_model
            cfg.tdt_joint.enc_d = cfg.encoder.d_model
            cfg.tdt_joint.pred_d = cfg.encoder.d_model

            self.tdt_decoder = TDTDecoder(cfg.tdt_decoder)
            self.tdt_joint = JointNet(cfg.tdt_joint)

    def encode(
        self, feats: torch.Tensor, feat_lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        """Encodes log-mel features and returns encoder output with auxiliary layers."""
        if self.aux_layers:
            enc, out_lengths, layer_outs = self.encoder(
                feats, feat_lengths, return_layer_outputs=True
            )
        else:
            enc, out_lengths = self.encoder(feats, feat_lengths, return_layer_outputs=False)
            layer_outs = []

        return enc, out_lengths.clamp(max=enc.size(1)), layer_outs

    def forward_ctc(
        self, enc: torch.Tensor, out_lengths: torch.Tensor, layer_outs: list[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Computes CTC log probabilities for primary and auxiliary heads."""
        logits = self.ctc_proj(enc)
        aux_logits = [proj(layer_outs[i]) for i, proj in enumerate(self.aux_projs)]
        return logits, torch.stack(aux_logits, dim=0) if aux_logits else torch.tensor([])

    def forward_attn(
        self, enc: torch.Tensor, out_lengths: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """Computes Attention decoder output log probabilities."""
        tgt_len = targets.size(1)
        tgt = self.tok_embed(targets) + self.pos_embed.weight[:tgt_len]
        
        mask = torch.triu(torch.ones(tgt_len, tgt_len, device=enc.device), diagonal=1).bool()
        memory_mask = (~(torch.arange(enc.size(1), device=enc.device) < out_lengths.unsqueeze(1))).T

        dec_out = self.decoder(tgt, enc, tgt_mask=mask, memory_key_padding_mask=memory_mask)
        return self.dec_proj(dec_out)

    def forward_tdt(
        self, enc: torch.Tensor, out_lengths: torch.Tensor, targets: torch.Tensor, target_lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        """Computes transducer branch joint network outputs."""
        dec_out = self.tdt_decoder(targets, target_lengths)
        return self.tdt_joint(enc.unsqueeze(2), dec_out.unsqueeze(1), out_lengths)

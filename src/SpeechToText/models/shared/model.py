from __future__ import annotations

from dataclasses import replace
from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from SpeechToText.models.common.aux_layers import resolve_aux_layer_indices
from SpeechToText.models.conformer import FastConformerEncoder
from SpeechToText.models.shared.config import SharedASRConfig
from SpeechToText.models.tdt.decoder import TDTDecoder
from SpeechToText.models.tdt.joint import JointNet
from SpeechToText.models.typing import SharedASROutput


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
            tdt_decoder_cfg = replace(
                cfg.tdt_decoder,
                vocab_size=self.vocab_size,
                d_model=cfg.encoder.d_model,
            )
            tdt_joint_cfg = replace(
                cfg.tdt_joint,
                vocab_size=self.vocab_size,
                enc_d=cfg.encoder.d_model,
                pred_d=cfg.encoder.d_model,
            )

            self.tdt_decoder = TDTDecoder(tdt_decoder_cfg)
            self.tdt_joint = JointNet(tdt_joint_cfg)

    @staticmethod
    def _build_tdt_decoder_input(
        targets: torch.Tensor, target_lengths: torch.Tensor, blank_id: int
    ) -> torch.Tensor:
        batch_size, max_target_len = targets.shape
        dec_in = torch.full(
            (batch_size, max_target_len + 1),
            blank_id,
            dtype=targets.dtype,
            device=targets.device,
        )
        for batch_index in range(batch_size):
            target_len = int(target_lengths[batch_index].item())
            if target_len > 0:
                dec_in[batch_index, 1 : target_len + 1] = targets[batch_index, :target_len]
        return dec_in

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
        ctc_logits = self.ctc_proj(enc)
        ctc_log_probs = F.log_softmax(ctc_logits, dim=-1)

        aux_logits = [
            proj(layer_outs[layer_index])
            for layer_index, proj in zip(self.aux_layers, self.aux_projs, strict=True)
        ]
        if aux_logits:
            aux_stacked = torch.stack(aux_logits, dim=0)
            aux_log_probs = F.log_softmax(aux_stacked, dim=-1)
        else:
            aux_log_probs = torch.empty(
                (0, ctc_log_probs.size(0), ctc_log_probs.size(1), ctc_log_probs.size(2)),
                device=ctc_log_probs.device,
                dtype=ctc_log_probs.dtype,
            )

        return ctc_log_probs, aux_log_probs

    def forward_attn(
        self, enc: torch.Tensor, out_lengths: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """Computes Attention decoder output log probabilities."""
        batch_size, tgt_len = targets.shape
        device = enc.device
        positions = torch.arange(tgt_len, device=device).unsqueeze(0).expand(batch_size, tgt_len)

        tgt = self.tok_embed(targets) + self.pos_embed(positions)
        tgt_mask = torch.triu(
            torch.ones(tgt_len, tgt_len, device=device, dtype=torch.bool), diagonal=1
        )
        tgt_key_padding_mask = targets.eq(self.pad_id)
        memory_mask = torch.arange(enc.size(1), device=device).unsqueeze(
            0
        ) >= out_lengths.unsqueeze(1)

        dec_out = self.decoder(
            tgt,
            enc,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_mask,
        )
        logits = self.dec_proj(dec_out)
        return cast(torch.Tensor, F.log_softmax(logits, dim=-1))

    def forward_tdt(
        self,
        enc: torch.Tensor,
        out_lengths: torch.Tensor,
        targets: torch.Tensor,
        target_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        """Computes transducer branch joint network outputs."""
        dec_in = self._build_tdt_decoder_input(targets, target_lengths, self.blank_id)
        dec_out = self.tdt_decoder(dec_in)
        joint_out = self.tdt_joint(enc, dec_out)
        if isinstance(joint_out, tuple):
            token_logits, duration_logits = joint_out
            return token_logits, duration_logits, out_lengths
        return joint_out, None, out_lengths

    def forward(
        self,
        feats: torch.Tensor,
        feat_lengths: torch.Tensor,
        *,
        targets: torch.Tensor | None = None,
        target_lengths: torch.Tensor | None = None,
        decoder_input: torch.Tensor | None = None,
    ) -> SharedASROutput:
        """Run the shared encoder and any active decoding heads."""
        enc, out_lengths, layer_outs = self.encode(feats, feat_lengths)

        ctc_log_probs: torch.Tensor | None = None
        aux_log_probs: torch.Tensor | None = None
        if "ctc" in self.cfg.active_heads:
            ctc_log_probs, aux_log_probs = self.forward_ctc(enc, out_lengths, layer_outs)

        dec_log_probs: torch.Tensor | None = None
        if "attn" in self.cfg.active_heads:
            attn_targets = decoder_input if decoder_input is not None else targets
            if attn_targets is not None:
                dec_log_probs = self.forward_attn(enc, out_lengths, attn_targets)

        token_logits: torch.Tensor | None = None
        duration_logits: torch.Tensor | None = None
        if "tdt" in self.cfg.active_heads and targets is not None and target_lengths is not None:
            token_logits, duration_logits, _ = self.forward_tdt(
                enc, out_lengths, targets, target_lengths
            )

        return SharedASROutput(
            out_lengths=out_lengths,
            ctc_log_probs=ctc_log_probs,
            aux_log_probs=aux_log_probs,
            dec_log_probs=dec_log_probs,
            token_logits=token_logits,
            duration_logits=duration_logits,
        )

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from SpeechToText.models.conformer import FastConformerEncoder, FastConformerEncoderConfig
from SpeechToText.models.typing import TDTOutput

from .decoder import TDTDecoder, TDTDecoderConfig
from .joint import JointNet, JointNetConfig


@dataclass
class FastConformerTDTConfig:
    encoder: FastConformerEncoderConfig = field(default_factory=FastConformerEncoderConfig)
    decoder: TDTDecoderConfig = field(default_factory=TDTDecoderConfig)
    joint: JointNetConfig = field(default_factory=JointNetConfig)
    blank_id: int = 0
    joint_fused_batch_size: int | None = 4


class FastConformerTDT(nn.Module):
    def __init__(self, cfg: FastConformerTDTConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.blank_id = int(cfg.blank_id)

        if cfg.decoder.vocab_size != cfg.joint.vocab_size:
            raise ValueError("decoder.vocab_size must equal joint.vocab_size")

        self.encoder = FastConformerEncoder(cfg.encoder)
        self.decoder = TDTDecoder(cfg.decoder)
        self.joint = JointNet(cfg.joint)

    @staticmethod
    def build_decoder_input_from_concat(
        targets_concat: torch.Tensor, target_lengths: torch.Tensor, blank_id: int
    ) -> torch.Tensor:
        device = targets_concat.device
        b = int(target_lengths.size(0))
        u_max = int(target_lengths.max().item()) if b > 0 else 0

        dec_in = torch.full((b, u_max + 1), blank_id, dtype=torch.long, device=device)

        off = 0
        for i in range(b):
            u = int(target_lengths[i].item())
            if u > 0:
                dec_in[i, 1 : u + 1] = targets_concat[off : off + u].to(torch.long)
            off += u
        return dec_in

    @staticmethod
    def pad_targets_from_concat(
        targets_concat: torch.Tensor, target_lengths: torch.Tensor, pad_id: int
    ) -> torch.Tensor:
        device = targets_concat.device
        b = int(target_lengths.size(0))
        u_max = int(target_lengths.max().item()) if b > 0 else 0
        out = torch.full((b, u_max), pad_id, dtype=torch.long, device=device)
        off = 0
        for i in range(b):
            u = int(target_lengths[i].item())
            if u > 0:
                out[i, :u] = targets_concat[off : off + u].to(torch.long)
            off += u
        return out

    def forward(
        self,
        feats: torch.Tensor,
        feat_lengths: torch.Tensor,
        targets_concat: torch.Tensor,
        target_lengths: torch.Tensor,
    ) -> TDTOutput:
        enc, out_lengths = self.encoder(feats, feat_lengths)
        out_lengths = out_lengths.clamp(max=int(enc.size(1)))
        dec_in = self.build_decoder_input_from_concat(targets_concat, target_lengths, self.blank_id)
        dec = self.decoder(dec_in)

        joint_out = self.joint.forward_chunked(
            enc,
            dec,
            fused_batch_size=self.cfg.joint_fused_batch_size,
        )

        token_logits: torch.Tensor
        duration_log_probs: torch.Tensor | None = None
        if isinstance(joint_out, tuple):
            token_logits, duration_logits = joint_out
        else:
            token_logits = joint_out
            duration_logits = None

        if self.training:
            # Loss path uses raw logits; skip expensive joint log-softmax during training.
            log_probs = token_logits
        else:
            log_probs = F.log_softmax(token_logits, dim=-1)
            if duration_logits is not None:
                duration_log_probs = F.log_softmax(duration_logits, dim=-1)

        return TDTOutput(
            log_probs=log_probs,
            out_lengths=out_lengths,
            target_lengths=target_lengths,
            token_logits=token_logits,
            duration_logits=duration_logits,
            duration_log_probs=duration_log_probs,
        )

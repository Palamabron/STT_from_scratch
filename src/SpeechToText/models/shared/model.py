from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

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

        # 1. SHARED ENCODER
        self.encoder = FastConformerEncoder(cfg.encoder)

        # 2. CTC HEAD (Optional)
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
        else:
            self.aux_layers = []

        # 3. ATTENTION DECODER HEAD (Optional)
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

        # 4. TDT / TRANSUDER HEAD (Optional)
        if "tdt" in cfg.active_heads:
            # Force matching vocab sizes and dimensions in sub-configs
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
        """Encode log-mel features and return representation with auxiliary outputs."""
        if self.aux_layers:
            enc, out_lengths, layer_outs = self.encoder(
                feats, feat_lengths, return_layer_outputs=True
            )
        else:
            enc, out_lengths = self.encoder(feats, feat_lengths, return_layer_outputs=False)
            layer_outs = []

        out_lengths = out_lengths.clamp(max=int(enc.size(1)))
        return enc, out_lengths, layer_outs

    def forward_ctc(
        self, enc: torch.Tensor, out_lengths: torch.Tensor, layer_outs: list[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute primary and auxiliary CTC log probs."""
        if "ctc" not in self.cfg.active_heads:
            raise ValueError("CTC head is not active in model config.")

        logits = self.ctc_proj(enc)
        log_probs = F.log_softmax(logits, dim=-1)

        if self.aux_layers and layer_outs:
            aux_encs = [layer_outs[i] for i in self.aux_layers]
            aux_logits = torch.stack(
                [proj(h) for proj, h in zip(self.aux_projs, aux_encs, strict=True)],
                dim=0,
            )
            aux_log_probs = F.log_softmax(aux_logits, dim=-1)
        else:
            aux_log_probs = torch.empty(0, device=log_probs.device, dtype=log_probs.dtype)

        return log_probs, aux_log_probs

    def _square_subsequent_mask(self, size: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones(size, size, device=device, dtype=torch.bool), diagonal=1)

    def _lengths_to_kpm(self, lengths: torch.Tensor, max_len: int) -> torch.Tensor:
        ids = torch.arange(max_len, device=lengths.device).unsqueeze(0)
        return ids >= lengths.unsqueeze(1)

    def forward_attention(
        self,
        enc: torch.Tensor,
        out_lengths: torch.Tensor,
        targets: torch.Tensor,
        target_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Compute Attention decoder log-probs."""
        if "attn" not in self.cfg.active_heads:
            raise ValueError("Attention head is not active in model config.")

        device = enc.device
        batch_size, seq_len = targets.shape

        # Embed targets
        target_embed = self.tok_embed(targets)
        pos = torch.arange(seq_len, device=device).unsqueeze(0)
        pos_embed = self.pos_embed(pos)
        tgt = target_embed + pos_embed

        # Masks
        tgt_mask = self._square_subsequent_mask(seq_len, device)
        tgt_key_padding_mask = self._lengths_to_kpm(target_lengths, seq_len)
        memory_key_padding_mask = self._lengths_to_kpm(out_lengths, enc.size(1))

        # Decode
        dec_out = self.decoder(
            tgt=tgt,
            memory=enc,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )

        logits = self.dec_proj(dec_out)
        return F.log_softmax(logits, dim=-1)

    def _build_tdt_decoder_input_from_concat(
        self, targets_concat: torch.Tensor, target_lengths: torch.Tensor
    ) -> torch.Tensor:
        device = targets_concat.device
        b = int(target_lengths.size(0))
        u_max = int(target_lengths.max().item()) if b > 0 else 0

        dec_in = torch.full((b, u_max + 1), self.blank_id, dtype=torch.long, device=device)
        off = 0
        for i in range(b):
            u = int(target_lengths[i].item())
            if u > 0:
                dec_in[i, 1 : u + 1] = targets_concat[off : off + u].to(torch.long)
            off += u
        return dec_in

    def forward_tdt(
        self,
        enc: torch.Tensor,
        out_lengths: torch.Tensor,
        targets_concat: torch.Tensor,
        target_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Compute TDT token (and optional duration) logits."""
        if "tdt" not in self.cfg.active_heads:
            raise ValueError("TDT head is not active in model config.")

        dec_in = self._build_tdt_decoder_input_from_concat(targets_concat, target_lengths)
        dec_out = self.tdt_decoder(dec_in)

        joint_out = self.tdt_joint.forward_chunked(
            enc,
            dec_out,
            fused_batch_size=4,
        )

        if isinstance(joint_out, tuple):
            token_logits, duration_logits = joint_out
        else:
            token_logits = joint_out
            duration_logits = None

        return token_logits, duration_logits

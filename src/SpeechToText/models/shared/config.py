from __future__ import annotations

from dataclasses import dataclass, field

from SpeechToText.models.conformer import FastConformerEncoderConfig
from SpeechToText.models.ctc_attention.model import AttentionDecoderConfig
from SpeechToText.models.tdt.decoder import TDTDecoderConfig
from SpeechToText.models.tdt.joint import JointNetConfig


@dataclass
class SharedASRConfig:
    """Configuration for Shared Encoder Multi-Head ASR."""

    encoder: FastConformerEncoderConfig = field(default_factory=FastConformerEncoderConfig)

    # Active heads to instantiate and train (e.g., ["ctc", "attn", "tdt"])
    active_heads: list[str] = field(default_factory=lambda: ["ctc", "attn"])

    # Loss weights for multi-loss backprop
    ctc_weight: float = 0.3
    aux_ctc_weight: float = 0.3
    attn_weight: float = 0.7
    tdt_weight: float = 1.0

    # Shared auxiliary layer config (for CTC heads)
    aux_interval: int = 0
    aux_layer: int | None = 8

    # Decoder configs
    attn_decoder: AttentionDecoderConfig = field(default_factory=AttentionDecoderConfig)
    tdt_decoder: TDTDecoderConfig = field(default_factory=TDTDecoderConfig)
    tdt_joint: JointNetConfig = field(default_factory=JointNetConfig)

    # Freezing schedules (epochs)
    freeze_encoder_epochs: int = 0
    decoder_warmup_epochs: int = 0
    ctc_calibration_epochs: int = 0

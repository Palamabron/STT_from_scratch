from __future__ import annotations

from pydantic import BaseModel, Field

from SpeechToText.models.conformer import FastConformerEncoderConfig
from SpeechToText.models.ctc_attention.model import AttentionDecoderConfig
from SpeechToText.models.tdt.decoder import TDTDecoderConfig
from SpeechToText.models.tdt.joint import JointNetConfig


class SharedASRConfig(BaseModel):
    """Configuration for Shared Encoder Multi-Head ASR."""

    encoder: FastConformerEncoderConfig = Field(default_factory=FastConformerEncoderConfig)

    active_heads: list[str] = Field(
        default_factory=lambda: ["ctc", "attn"],
        description="List of active model heads to instantiate (e.g., ['ctc', 'attn', 'tdt']).",
    )

    ctc_weight: float = Field(0.3, description="Loss weight for primary CTC branch.")
    aux_ctc_weight: float = Field(0.3, description="Loss weight for auxiliary CTC branch.")
    attn_weight: float = Field(0.7, description="Loss weight for attention decoder branch.")
    tdt_weight: float = Field(1.0, description="Loss weight for transducer branch.")

    aux_interval: int = Field(0, description="Interval for auxiliary CTC layers.")
    aux_layer: int | None = Field(8, description="Target layer index for auxiliary supervision.")

    attn_decoder: AttentionDecoderConfig = Field(default_factory=AttentionDecoderConfig)
    tdt_decoder: TDTDecoderConfig = Field(default_factory=TDTDecoderConfig)
    tdt_joint: JointNetConfig = Field(default_factory=JointNetConfig)

    freeze_encoder_epochs: int = Field(0, description="Epochs to freeze encoder during warm-up.")
    decoder_warmup_epochs: int = Field(0, description="Epochs to train decoder heads in isolation.")
    ctc_calibration_epochs: int = Field(0, description="Epochs for auxiliary CTC calibration.")

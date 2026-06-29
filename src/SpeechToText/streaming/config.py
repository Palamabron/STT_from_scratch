from __future__ import annotations

from pydantic import BaseModel, Field


class StreamingConfig(BaseModel):
    """Configuration for real-time streaming ASR."""

    sample_rate: int = Field(16_000, description="Audio sample rate in Hz.")
    chunk_ms: int = Field(320, description="Processing chunk duration in milliseconds.")
    hop_ms: int = Field(160, description="Step duration between processing chunks in milliseconds.")
    context_ms: int = Field(1000, description="Left context duration for state preservation.")
    subsampling_factor: int = Field(8, description="Encoder subsampling factor.")
    hop_length_ms: float = Field(10.0, description="Feature extraction hop size in milliseconds.")
    max_symbols_per_t: int = Field(10, description="Maximum transducer expansion steps per frame.")

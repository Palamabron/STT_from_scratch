from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class StreamingConfig:
    """Configuration for real-time streaming ASR."""

    sample_rate: int = 16_000
    chunk_ms: int = 320  # Size of the processed chunk (320ms)
    hop_ms: int = 160  # Step size between chunks (160ms)
    context_ms: int = 1000  # Left context size for conformer state preservation (1000ms)
    subsampling_factor: int = 8  # Subsampling factor of the FastConformer encoder
    hop_length_ms: float = 10.0  # Feature extractor hop size in ms
    max_symbols_per_t: int = 10  # Max transducer expansion steps per time step

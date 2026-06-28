from __future__ import annotations

import numpy as np


class LatencyTracker:
    """Tracks latency metrics during real-time streaming simulation."""

    def __init__(self) -> None:
        self.audio_processed_sec: float = 0.0
        self.total_processing_time_sec: float = 0.0
        self.latencies_ms: list[float] = []

    def reset(self) -> None:
        """Reset the tracker state."""
        self.audio_processed_sec = 0.0
        self.total_processing_time_sec = 0.0
        self.latencies_ms = []

    def record_step(self, chunk_duration_sec: float, processing_time_sec: float) -> None:
        """Record a single processing step.

        Args:
            chunk_duration_sec: Duration of the audio chunk in seconds.
            processing_time_sec: Time spent processing the chunk in seconds.
        """
        self.audio_processed_sec += chunk_duration_sec
        self.total_processing_time_sec += processing_time_sec
        self.latencies_ms.append(processing_time_sec * 1000.0)

    def rtf(self) -> float:
        """Calculate Real-Time Factor (RTF)."""
        if self.audio_processed_sec <= 0.0:
            return 0.0
        return self.total_processing_time_sec / self.audio_processed_sec

    def mean_latency_ms(self) -> float:
        """Calculate the average latency per step in milliseconds."""
        if not self.latencies_ms:
            return 0.0
        return float(np.mean(self.latencies_ms))

    def p50_latency_ms(self) -> float:
        """Calculate the 50th percentile (median) latency in milliseconds."""
        if not self.latencies_ms:
            return 0.0
        return float(np.percentile(self.latencies_ms, 50))

    def p95_latency_ms(self) -> float:
        """Calculate the 95th percentile (tail) latency in milliseconds."""
        if not self.latencies_ms:
            return 0.0
        return float(np.percentile(self.latencies_ms, 95))

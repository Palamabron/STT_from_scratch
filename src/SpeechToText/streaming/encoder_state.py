from __future__ import annotations

import torch
import torch.nn as nn
from loguru import logger

from SpeechToText.streaming.config import StreamingConfig


class StreamingEncoderState:
    """Manages the audio buffer and sliding-window feature extraction + encoding.

    Each hop re-encodes the left-context + chunk window. The Conformer encoder is not
    incremental yet, so streaming hypotheses may differ slightly from offline batch
    encoding on the same audio.
    """

    def __init__(self, model: nn.Module, config: StreamingConfig) -> None:
        self.model = model
        self.config = config

        # Buffer to keep the running raw audio samples
        self.audio_buffer = torch.zeros(0, dtype=torch.float32)

        # Calculate samples counts
        self.sample_rate = config.sample_rate
        self.hop_samples = int(config.hop_ms * self.sample_rate / 1000.0)
        self.context_samples = int(config.context_ms * self.sample_rate / 1000.0)
        self.chunk_samples = int(config.chunk_ms * self.sample_rate / 1000.0)

        self.mel_hop_samples = int(self.sample_rate * config.hop_length_ms / 1000.0)
        self.subsampling = config.subsampling_factor

        # Calculate exactly how many encoder frames correspond to one hop step
        # new_frames = (hop_samples / mel_hop_samples) / subsampling
        self.new_enc_frames = int((self.hop_samples // self.mel_hop_samples) // self.subsampling)
        if self.new_enc_frames == 0:
            self.new_enc_frames = 1

        logger.info(
            "Initialized StreamingEncoderState with sample_rate={}, "
            "hop_samples={}, context_samples={}, chunk_samples={}, "
            "new_enc_frames_per_step={}",
            self.sample_rate,
            self.hop_samples,
            self.context_samples,
            self.chunk_samples,
            self.new_enc_frames,
        )

    def reset(self) -> None:
        """Reset the internal audio buffer."""
        self.audio_buffer = torch.zeros(0, dtype=torch.float32)

    def append_audio(self, samples: torch.Tensor) -> None:
        """Append raw 1D audio samples to the buffer."""
        if samples.dim() != 1:
            samples = samples.view(-1)
        self.audio_buffer = torch.cat([self.audio_buffer, samples.cpu()])

    @torch.no_grad()
    def process_next_chunk(self) -> torch.Tensor | None:
        """Extract features and encode the next step of audio from the buffer.

        Returns:
            The new encoder output frames of shape [1, new_enc_frames, encoder_dim],
            or None if there are not enough samples in the buffer to process a new step.
        """
        # We need at least hop_samples to produce the next output frames
        if len(self.audio_buffer) < self.hop_samples:
            return None

        # Left context plus one decode chunk (see StreamingConfig.chunk_ms / hop_ms).
        window_size = self.context_samples + self.chunk_samples
        if len(self.audio_buffer) > window_size:
            audio_window = self.audio_buffer[-window_size:]
        else:
            audio_window = self.audio_buffer

        # Run featurizer and encoder on CPU (or model's device)
        device = next(self.model.parameters()).device
        wav = audio_window.unsqueeze(0).to(device)  # Shape: [1, T_samples]
        wav_lens = torch.tensor([audio_window.shape[0]], dtype=torch.long, device=device)

        # 1. Feature extraction
        # Handle the featurizer either on lit model or nested network
        featurizer = getattr(self.model, "featurizer", None)
        if featurizer is None:
            raise AttributeError("Model does not have a 'featurizer' module.")

        feats, feat_lens = featurizer(wav, wav_lens)

        # 2. Conformer Encoding
        encoder = (
            getattr(self.model.net, "encoder", None)
            if hasattr(self.model, "net")
            else getattr(self.model, "encoder", None)
        )
        if encoder is None:
            raise AttributeError(
                "Model does not have an 'encoder' module in net or as root attribute."
            )

        enc, out_lengths = encoder(feats, feat_lens)  # Shape: [1, T_enc, D]

        # 3. Slice out the new frames at the end corresponding to the hop step
        t_enc = enc.size(1)
        if t_enc < self.new_enc_frames:
            # Not enough encoder frames yet, don't return anything, wait for more audio
            return None

        # Extract the last new_enc_frames which corresponds to the new hop step
        new_hop_enc = enc[:, -self.new_enc_frames :, :]

        # Consume/slide the audio buffer by removing the processed hop_samples
        # This keeps the buffer size bounded and prevents memory leaks!
        self.audio_buffer = self.audio_buffer[self.hop_samples :]

        return new_hop_enc

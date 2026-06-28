from __future__ import annotations

import torch
import torch.nn as nn
from loguru import logger

from SpeechToText.streaming.config import StreamingConfig


class StreamingEncoderState:
    """Manages raw audio buffers and sliding-window feature extraction + encoding.

    Each hop re-encodes the left-context + chunk window to provide stable
    Conformer contextual frame encodings over streaming audio signals.
    """

    def __init__(self, model: nn.Module, config: StreamingConfig) -> None:
        """Initializes the streaming encoder state and computes window constants.

        Args:
            model: The ASR model containing a featurizer and encoder.
            config: Configuration parameters for streaming processing.
        """
        self.model = model
        self.config = config
        self.audio_buffer = torch.zeros(0, dtype=torch.float32)

        self.sample_rate = config.sample_rate
        self.hop_samples = int(config.hop_ms * self.sample_rate / 1000.0)
        self.context_samples = int(config.context_ms * self.sample_rate / 1000.0)
        self.chunk_samples = int(config.chunk_ms * self.sample_rate / 1000.0)

        self.mel_hop_samples = int(self.sample_rate * config.hop_length_ms / 1000.0)
        self.subsampling = config.subsampling_factor

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
        """Resets the internal raw audio buffer to start a new stream."""
        self.audio_buffer = torch.zeros(0, dtype=torch.float32)

    def append_audio(self, samples: torch.Tensor) -> None:
        """Appends raw 1D audio waveform samples to the internal buffer.

        Args:
            samples: A 1D tensor containing raw audio samples.
        """
        if samples.dim() != 1:
            samples = samples.view(-1)
        self.audio_buffer = torch.cat([self.audio_buffer, samples.cpu()])

    @torch.no_grad()
    def process_next_chunk(self) -> torch.Tensor | None:
        """Extracts features and encodes the next audio hop from the buffer.

        Retrieves a contextual slice of audio, forwards it through the feature
        extractor and Conformer encoder, extracts the newly computed frame encodings
        corresponding to the hop step, and slides the audio buffer forward.

        Returns:
            The new encoder output frames of shape [1, new_enc_frames, encoder_dim],
            or None if the buffer contains fewer than hop_samples.

        Raises:
            AttributeError: If model lacks a 'featurizer' or 'encoder' module.
        """
        if len(self.audio_buffer) < self.hop_samples:
            return None

        window_size = self.context_samples + self.chunk_samples
        if len(self.audio_buffer) > window_size:
            audio_window = self.audio_buffer[-window_size:]
        else:
            audio_window = self.audio_buffer

        device = next(self.model.parameters()).device
        wav = audio_window.unsqueeze(0).to(device)
        wav_lens = torch.tensor([audio_window.shape[0]], dtype=torch.long, device=device)

        featurizer = getattr(self.model, "featurizer", None)
        if featurizer is None:
            raise AttributeError("Model does not have a 'featurizer' module.")

        feats, feat_lens = featurizer(wav, wav_lens)

        encoder = (
            getattr(self.model.net, "encoder", None)
            if hasattr(self.model, "net")
            else getattr(self.model, "encoder", None)
        )
        if encoder is None:
            raise AttributeError(
                "Model does not have an 'encoder' module in net or as root attribute."
            )

        enc, out_lengths = encoder(feats, feat_lens)

        t_enc = enc.size(1)
        if t_enc < self.new_enc_frames:
            return None

        new_hop_enc = enc[:, -self.new_enc_frames :, :]

        self.audio_buffer = self.audio_buffer[self.hop_samples :]

        return new_hop_enc

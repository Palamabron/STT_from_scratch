from __future__ import annotations

import time

import torch
import torch.nn as nn
from sentencepiece import SentencePieceProcessor

from SpeechToText.streaming.config import StreamingConfig
from SpeechToText.streaming.encoder_state import StreamingEncoderState
from SpeechToText.streaming.latency import LatencyTracker
from SpeechToText.streaming.streaming_ctc import StreamingCTCDecoder
from SpeechToText.streaming.streaming_tdt import StreamingTDTDecoder


def _is_transducer_net(net: nn.Module) -> bool:
    """Detect RNN-T/TDT heads without confusing CTC+Attention transformer decoders."""
    return hasattr(net, "joint") or hasattr(net, "tdt_joint")


class StreamingSession:
    """Unified API for running real-time stateful ASR streaming sessions."""

    def __init__(
        self, model: nn.Module, sp: SentencePieceProcessor, config: StreamingConfig | None = None
    ) -> None:
        self.model = model
        self.sp = sp
        self.config = config or StreamingConfig()
        self.blank_id = int(getattr(model, "blank_id", 0))

        self.encoder_state = StreamingEncoderState(self.model, self.config)
        self.latency_tracker = LatencyTracker()

        net = getattr(self.model, "net", self.model)
        if _is_transducer_net(net):
            self.decoder = StreamingTDTDecoder(
                self.model,
                blank_id=self.blank_id,
                max_symbols_per_t=self.config.max_symbols_per_t,
            )
            self.model_type = "transducer"
        else:
            self.decoder = StreamingCTCDecoder(self.model, blank_id=self.blank_id)
            self.model_type = "ctc"

        self.accumulated_text = ""
        self.emitted_ids: list[int] = []
        self.full_audio_buffer = torch.zeros(0, dtype=torch.float32)

    def reset(self) -> None:
        """Reset the entire session state (buffers, decoders, latency trackers)."""
        self.encoder_state.reset()
        self.decoder.reset()
        self.latency_tracker.reset()
        self.accumulated_text = ""
        self.emitted_ids = []
        self.full_audio_buffer = torch.zeros(0, dtype=torch.float32)

    def append_audio(self, samples: torch.Tensor) -> None:
        """Append a new chunk of raw mono audio waveform."""
        if samples.dim() != 1:
            samples = samples.view(-1)
        self.encoder_state.append_audio(samples)
        self.full_audio_buffer = torch.cat([self.full_audio_buffer, samples.cpu()])

    def process_step(self) -> str:
        """Process next pending audio step from the buffer.

        Returns:
            The newly transcribed text from this step (empty if no new tokens or not enough audio).
        """
        start_time = time.perf_counter()

        enc_chunk = self.encoder_state.process_next_chunk()
        if enc_chunk is None:
            return ""

        new_token_ids = self.decoder.decode_chunk(enc_chunk)

        elapsed = time.perf_counter() - start_time
        chunk_duration_sec = self.config.hop_ms / 1000.0
        self.latency_tracker.record_step(chunk_duration_sec, elapsed)

        if not new_token_ids:
            return ""

        self.emitted_ids.extend(new_token_ids)

        sp_ids = [tok_id - 1 for tok_id in self.emitted_ids if tok_id > 0]
        full_text = "" if not sp_ids else self.sp.decode_ids(sp_ids)

        new_text = full_text[len(self.accumulated_text) :]
        self.accumulated_text = full_text

        return new_text

    def get_full_transcript(self) -> str:
        """Get the full accumulated transcript text."""
        return self.accumulated_text

    def get_latency_metrics(self) -> dict[str, float]:
        """Get the latency metrics (RTF, P50, P95)."""
        return {
            "rtf": self.latency_tracker.rtf(),
            "mean_latency_ms": self.latency_tracker.mean_latency_ms(),
            "p50_latency_ms": self.latency_tracker.p50_latency_ms(),
            "p95_latency_ms": self.latency_tracker.p95_latency_ms(),
        }

    def finish_stream_rescore(self) -> str:
        """Run second-pass attention rescoring on the full buffered audio.

        Re-encodes the complete utterance in one pass for accurate attention decoding,
        instead of concatenating per-chunk streaming encoder outputs.

        Returns:
            The final rescored transcript string (or the streaming CTC transcript if unsupported).
        """
        if self.full_audio_buffer.numel() == 0:
            return self.accumulated_text

        net = getattr(self.model, "net", self.model)
        has_attention_decoder = hasattr(net, "decoder") and not _is_transducer_net(net)
        has_ctc_head = hasattr(net, "ctc_proj") or hasattr(net, "proj")
        if not has_attention_decoder or not has_ctc_head or self.model_type == "transducer":
            return self.accumulated_text

        try:
            from SpeechToText.models.common.inference import (
                ctc_attention_special_tokens,
                decode_ctc_attention_attention_greedy,
            )

            device = next(self.model.parameters()).device
            wav = self.full_audio_buffer.unsqueeze(0).to(device)
            wav_lens = torch.tensor([wav.size(1)], dtype=torch.long, device=device)
            feats, feat_lens = self.model.featurizer(wav, wav_lens)
            enc, out_lengths, _aux = net.encode(feats, feat_lens)

            tokens = ctc_attention_special_tokens(self.model)
            rescored_text = decode_ctc_attention_attention_greedy(
                self.model, enc, out_lengths, sample_index=0, sp=self.sp, tokens=tokens
            )

            self.accumulated_text = rescored_text
            self.emitted_ids = []
            return rescored_text
        except (RuntimeError, ValueError, IndexError, AttributeError) as exc:
            import logging

            logging.warning(
                "Error during attention rescoring: %s. Falling back to streaming hypothesis.",
                exc,
            )
            return self.accumulated_text

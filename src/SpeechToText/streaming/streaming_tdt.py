from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class StreamingTDTDecoder:
    """Stateful streaming greedy Transducer/TDT decoder.

    Supports both traditional RNN-T (greedy frame-by-frame) and Time-Depth Transducer (TDT)
    with frame-skipping across chunk boundaries.
    """

    def __init__(self, model: nn.Module, blank_id: int = 0, max_symbols_per_t: int = 10) -> None:
        self.model = model
        self.blank_id = blank_id
        self.max_symbols_per_t = max_symbols_per_t

        # State variables
        self.emitted: list[int] = []
        self.dec_tokens = torch.tensor([[blank_id]], dtype=torch.long)
        self.pred: torch.Tensor | None = None
        self.skip_frames = 0  # Number of frames to skip at the start of the next chunk

    def reset(self) -> None:
        """Reset decoder state."""
        self.emitted = []
        self.dec_tokens = torch.tensor([[self.blank_id]], dtype=torch.long)
        self.pred = None
        self.skip_frames = 0

    def _get_joint_and_decoder(self) -> tuple[nn.Module, nn.Module]:
        """Locate decoder and joint net modules from model."""
        net = getattr(self.model, "net", self.model)

        decoder = getattr(net, "decoder", None) or getattr(net, "tdt_decoder", None)
        joint = getattr(net, "joint", None) or getattr(net, "tdt_joint", None)

        if decoder is None or joint is None:
            raise AttributeError(
                "Model must have transducer 'decoder'/'joint' or 'tdt_decoder'/'tdt_joint' modules."
            )

        return decoder, joint

    @torch.no_grad()
    def decode_chunk(self, enc_chunk: torch.Tensor) -> list[int]:
        """Decode a new encoder output chunk statefully.

        Args:
            enc_chunk: Encoder output chunk of shape [1, T, D].

        Returns:
            The newly emitted token IDs from this chunk.
        """
        device = enc_chunk.device
        decoder, joint = self._get_joint_and_decoder()

        # Move state tensors to the correct device if needed
        self.dec_tokens = self.dec_tokens.to(device)

        # Initialize predictor output if it's the first run
        if self.pred is None:
            self.pred = decoder(self.dec_tokens)

        assert self.pred is not None
        self.pred = self.pred.to(device)

        t_len = enc_chunk.size(1)
        new_emitted: list[int] = []

        t = 0
        u = len(self.emitted)

        # Check if joint network supports TDT duration prediction
        is_tdt = getattr(joint, "duration_out", None) is not None

        while t < t_len:
            # Track t at start of iteration to detect and prevent infinite loops
            t_start = t

            # Handle cross-boundary skips
            if self.skip_frames > 0:
                skip_now = min(self.skip_frames, t_len - t)
                t += skip_now
                self.skip_frames -= skip_now
                if t >= t_len:
                    break
                t_start = t

            n_emit = 0

            while n_emit < self.max_symbols_per_t and t < t_len:
                # 1. Forward step through joint network
                enc_frame = enc_chunk[0, t, :].unsqueeze(0).unsqueeze(1)
                pred_frame = self.pred[0, u, :].unsqueeze(0).unsqueeze(1)

                joint_out = joint(enc_frame, pred_frame)
                if isinstance(joint_out, tuple):
                    token_logits, duration_logits = joint_out
                    token_log_probs = F.log_softmax(token_logits.squeeze(1).squeeze(1), dim=-1)
                    duration_log_probs = F.log_softmax(
                        duration_logits.squeeze(1).squeeze(1), dim=-1
                    )
                else:
                    token_log_probs = F.log_softmax(joint_out.squeeze(1).squeeze(1), dim=-1)
                    duration_log_probs = None

                token_id = int(torch.argmax(token_log_probs).item())

                if token_id == self.blank_id:
                    # Advancing on blank
                    if is_tdt and duration_log_probs is not None:
                        duration = int(torch.argmax(duration_log_probs).item())
                        step = max(1, duration + 1)
                    else:
                        step = 1

                    # Calculate if the step goes beyond the current chunk boundary
                    if t + step < t_len:
                        t += step
                    else:
                        self.skip_frames = (t + step) - t_len
                        t = t_len  # Force exit loop

                    break

                # Emit non-blank token
                new_emitted.append(token_id)
                self.emitted.append(token_id)
                u += 1
                n_emit += 1

                # Update decoder/predictor input and state
                token_tensor = torch.tensor([[token_id]], dtype=torch.long, device=device)
                self.dec_tokens = torch.cat([self.dec_tokens, token_tensor], dim=1)
                self.pred = decoder(self.dec_tokens)

                # In TDT, non-blank decisions can also predict frame duration/skip
                if is_tdt and duration_log_probs is not None:
                    duration = int(torch.argmax(duration_log_probs).item())
                    step = max(1, duration + 1)
                    if t + step < t_len:
                        t += step
                    else:
                        self.skip_frames = (t + step) - t_len
                        t = t_len
                    break

            # Universal guard to prevent infinite loop: if t didn't advance, step it forward
            if t == t_start:
                t += 1

        return new_emitted

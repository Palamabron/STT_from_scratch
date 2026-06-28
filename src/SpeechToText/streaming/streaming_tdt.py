from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class StreamingTDTDecoder:
    """Stateful streaming greedy Transducer/TDT decoder.

    Supports traditional Recurrent Neural Network Transducer (RNN-T) decoding
    and Time-Depth Transducer (TDT) decoding with stateful frame-skipping across
    chunk boundaries.
    """

    def __init__(self, model: nn.Module, blank_id: int = 0, max_symbols_per_t: int = 10) -> None:
        """Initializes the streaming transducer decoder.

        Args:
            model: The ASR model containing a joint and predictor networks.
            blank_id: The vocabulary ID corresponding to the blank/null token.
            max_symbols_per_t: Max non-blank symbols allowed per frame step.
        """
        self.model = model
        self.blank_id = blank_id
        self.max_symbols_per_t = max_symbols_per_t

        self.emitted: list[int] = []
        self.dec_tokens = torch.tensor([[blank_id]], dtype=torch.long)
        self.pred: torch.Tensor | None = None
        self.skip_frames = 0

    def reset(self) -> None:
        """Resets the internal decoder and predictor state for a new session."""
        self.emitted = []
        self.dec_tokens = torch.tensor([[self.blank_id]], dtype=torch.long)
        self.pred = None
        self.skip_frames = 0

    def _get_joint_and_decoder(self) -> tuple[nn.Module, nn.Module]:
        """Locates decoder and joint network modules from the parent model structure.

        Returns:
            A tuple of (decoder, joint) PyTorch submodules.

        Raises:
            AttributeError: If transducer modules cannot be resolved.
        """
        net = getattr(self.model, "net", self.model)
        decoder = getattr(net, "decoder", None) or getattr(net, "tdt_decoder", None)
        joint = getattr(net, "joint", None) or getattr(net, "tdt_joint", None)

        if decoder is None or joint is None:
            raise AttributeError(
                "Model must have transducer 'decoder'/'joint' or 'tdt_decoder'/'tdt_joint' modules."
            )

        return decoder, joint

    def _initialize_decoder_state(self, device: torch.device, decoder: nn.Module) -> None:
        """Sets up prediction tensors and moves historical states to active device.

        Args:
            device: Active computation device.
            decoder: The predictor/decoder network module.
        """
        self.dec_tokens = self.dec_tokens.to(device)
        if self.pred is None:
            self.pred = decoder(self.dec_tokens)
        assert self.pred is not None
        self.pred = self.pred.to(device)

    def _apply_cross_boundary_skips(self, t: int, t_len: int) -> int:
        """Applies remaining skip offsets across chunk boundaries.

        Args:
            t: Current encoder frame index.
            t_len: Current chunk frame length.

        Returns:
            The advanced encoder frame index.
        """
        if self.skip_frames > 0:
            skip_now = min(self.skip_frames, t_len - t)
            t += skip_now
            self.skip_frames -= skip_now
        return t

    def _decode_single_frame(
        self,
        enc_chunk: torch.Tensor,
        t: int,
        u: int,
        t_len: int,
        decoder: nn.Module,
        joint: nn.Module,
        is_tdt: bool,
        device: torch.device,
        new_emitted: list[int],
    ) -> tuple[int, int, bool]:
        """Performs a single joint prediction step at frame index t.

        Args:
            enc_chunk: Encoder output chunk of shape [1, T, D].
            t: Current encoder frame index.
            u: Index of the current target symbol index in prediction history.
            t_len: Current chunk frame length.
            decoder: Predictor network submodule.
            joint: Joint network submodule.
            is_tdt: Whether the joint network predicts token duration/skips.
            device: Computing device.
            new_emitted: Accumulating list of newly emitted tokens.

        Returns:
            A tuple containing (updated_t, updated_u, loop_should_break).
        """
        assert self.pred is not None
        enc_frame = enc_chunk[0, t, :].unsqueeze(0).unsqueeze(1)
        pred_frame = self.pred[0, u, :].unsqueeze(0).unsqueeze(1)

        joint_out = joint(enc_frame, pred_frame)
        if isinstance(joint_out, tuple):
            token_logits, duration_logits = joint_out
            token_log_probs = F.log_softmax(token_logits.squeeze(1).squeeze(1), dim=-1)
            duration_log_probs = F.log_softmax(duration_logits.squeeze(1).squeeze(1), dim=-1)
        else:
            token_log_probs = F.log_softmax(joint_out.squeeze(1).squeeze(1), dim=-1)
            duration_log_probs = None

        token_id = int(torch.argmax(token_log_probs).item())

        if token_id == self.blank_id:
            if is_tdt and duration_log_probs is not None:
                duration = int(torch.argmax(duration_log_probs).item())
                step = max(1, duration + 1)
            else:
                step = 1

            if t + step < t_len:
                t += step
            else:
                self.skip_frames = (t + step) - t_len
                t = t_len

            return t, u, True

        new_emitted.append(token_id)
        self.emitted.append(token_id)
        u += 1

        token_tensor = torch.tensor([[token_id]], dtype=torch.long, device=device)
        self.dec_tokens = torch.cat([self.dec_tokens, token_tensor], dim=1)
        self.pred = decoder(self.dec_tokens)

        if is_tdt and duration_log_probs is not None:
            duration = int(torch.argmax(duration_log_probs).item())
            step = max(1, duration + 1)
            if t + step < t_len:
                t += step
            else:
                self.skip_frames = (t + step) - t_len
                t = t_len
            return t, u, True

        return t, u, False

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
        self._initialize_decoder_state(device, decoder)

        t_len = enc_chunk.size(1)
        new_emitted: list[int] = []

        t = 0
        u = len(self.emitted)
        is_tdt = getattr(joint, "duration_out", None) is not None

        while t < t_len:
            t_start = t
            t = self._apply_cross_boundary_skips(t, t_len)
            if t >= t_len:
                break

            n_emit = 0
            while n_emit < self.max_symbols_per_t and t < t_len:
                old_u = u
                t, u, should_break = self._decode_single_frame(
                    enc_chunk=enc_chunk,
                    t=t,
                    u=u,
                    t_len=t_len,
                    decoder=decoder,
                    joint=joint,
                    is_tdt=is_tdt,
                    device=device,
                    new_emitted=new_emitted,
                )
                if u > old_u:
                    n_emit += 1
                if should_break:
                    break

            if t == t_start:
                t += 1

        return new_emitted

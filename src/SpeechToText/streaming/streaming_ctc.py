from __future__ import annotations

import torch
import torch.nn as nn


class StreamingCTCDecoder:
    """Stateful streaming greedy CTC decoder that collapses repeats across chunks."""

    def __init__(self, model: nn.Module, blank_id: int = 0) -> None:
        self.model = model
        self.blank_id = blank_id

        # State variables
        self.last_raw_id = -1
        self.decoded_ids: list[int] = []

    def reset(self) -> None:
        """Reset decoder state."""
        self.last_raw_id = -1
        self.decoded_ids = []

    @torch.no_grad()
    def decode_chunk(self, enc_chunk: torch.Tensor) -> list[int]:
        """Decode a new encoder chunk and return the new token IDs.

        Args:
            enc_chunk: Encoder output chunk of shape [1, T, D].

        Returns:
            The newly emitted token IDs from this chunk.
        """
        # 1. Get CTC log probabilities / logits
        # CTC projection head is usually net.ctc_proj, net.proj, or ctc_proj
        ctc_proj = getattr(self.model, "ctc_proj", None)
        if ctc_proj is None and hasattr(self.model, "net"):
            ctc_proj = getattr(self.model.net, "ctc_proj", None)
        if ctc_proj is None:
            # Fallback for standalone CTC models
            ctc_proj = getattr(self.model, "proj", None)
            if ctc_proj is None and hasattr(self.model, "net"):
                ctc_proj = getattr(self.model.net, "proj", None)

        if ctc_proj is None:
            raise AttributeError("Model does not have a 'ctc_proj' or 'proj' module.")

        logits = ctc_proj(enc_chunk)  # Shape: [1, T, V]
        preds = torch.argmax(logits, dim=-1).squeeze(0).cpu()  # Shape: [T]

        new_emitted: list[int] = []
        for token_id in preds:
            current_id = int(token_id.item())
            if current_id != self.last_raw_id:
                if current_id != self.blank_id:
                    new_emitted.append(current_id)
                    self.decoded_ids.append(current_id)
                self.last_raw_id = current_id

        return new_emitted

from __future__ import annotations

import torch
import torch.nn as nn


class StreamingCTCDecoder:
    """Stateful streaming greedy Connectionist Temporal Classification (CTC) decoder.

    Collapses repeats and filters blank tokens across incoming chunks to maintain
    streaming state and produce a continuous hypothesis transcript.
    """

    def __init__(self, model: nn.Module, blank_id: int = 0) -> None:
        """Initializes the streaming CTC decoder.

        Args:
            model: The ASR model containing a CTC projection head.
            blank_id: The vocabulary ID corresponding to the CTC blank token.
        """
        self.model = model
        self.blank_id = blank_id
        self.last_raw_id = -1
        self.decoded_ids: list[int] = []

    def reset(self) -> None:
        """Resets the internal decoder state to start a new decoding session."""
        self.last_raw_id = -1
        self.decoded_ids = []

    @torch.no_grad()
    def decode_chunk(self, enc_chunk: torch.Tensor) -> list[int]:
        """Decodes a new encoder chunk statefully.

        Locates the CTC projection head dynamically, computes the frame-level
        predictions, collapses repeats, removes blank tokens, and returns newly
        emitted token IDs.

        Args:
            enc_chunk: Encoder output chunk tensor of shape [1, T, D].

        Returns:
            A list of new token IDs emitted from this chunk.

        Raises:
            AttributeError: If no CTC projection head can be located on the model.
        """
        ctc_proj = getattr(self.model, "ctc_proj", None)
        if ctc_proj is None and hasattr(self.model, "net"):
            ctc_proj = getattr(self.model.net, "ctc_proj", None)
        if ctc_proj is None:
            ctc_proj = getattr(self.model, "proj", None)
            if ctc_proj is None and hasattr(self.model, "net"):
                ctc_proj = getattr(self.model.net, "proj", None)

        if ctc_proj is None:
            raise AttributeError("Model does not have a 'ctc_proj' or 'proj' module.")

        logits = ctc_proj(enc_chunk)
        preds = torch.argmax(logits, dim=-1).squeeze(0).cpu()

        new_emitted: list[int] = []
        for token_id in preds:
            current_id = int(token_id.item())
            if current_id != self.last_raw_id:
                if current_id != self.blank_id:
                    new_emitted.append(current_id)
                    self.decoded_ids.append(current_id)
                self.last_raw_id = current_id

        return new_emitted

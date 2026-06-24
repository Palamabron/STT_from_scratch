from __future__ import annotations

import torch
from sentencepiece import SentencePieceProcessor


def greedy_ctc_decode(
    log_probs: torch.Tensor,
    out_lengths: torch.Tensor,
    blank_id: int = 0,
) -> list[list[int]]:
    """Greedy CTC decode with repeat collapse and blank removal.

    Args:
        log_probs: Log-probabilities with shape ``[batch, time, vocab]``.
        out_lengths: Valid time steps per utterance with shape ``[batch]``.
        blank_id: Index of the CTC blank symbol.

    Returns:
        Decoded token-id sequences, one list per batch element.
    """
    preds = torch.argmax(log_probs, dim=-1).cpu()
    out_lengths_cpu = out_lengths.cpu()

    decoded: list[list[int]] = []
    for seq, length in zip(preds, out_lengths_cpu, strict=True):
        valid_steps = int(length.item())
        previous_id = -1
        tokens: list[int] = []
        for token_id in seq[:valid_steps]:
            current_id = int(token_id.item())
            if current_id != previous_id and current_id != blank_id:
                tokens.append(current_id)
            previous_id = current_id
        decoded.append(tokens)
    return decoded


def ctc_ids_to_texts_spm(sp: SentencePieceProcessor, decoded_ids: list[list[int]]) -> list[str]:
    """Convert shifted model token ids back to SentencePiece text.

    Model ids reserve ``0`` for the CTC blank; SentencePiece token ``i`` is
    stored as model id ``i + 1``.

    Args:
        sp: SentencePiece processor used during training.
        decoded_ids: Batch of decoded model token-id sequences.

    Returns:
        Decoded transcript strings.
    """
    pred_texts: list[str] = []
    for seq in decoded_ids:
        sp_ids = [token_id - 1 for token_id in seq if token_id > 0]
        pred_texts.append("" if not sp_ids else sp.decode_ids(sp_ids))
    return pred_texts

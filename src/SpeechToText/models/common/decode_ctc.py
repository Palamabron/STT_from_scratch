from __future__ import annotations

import torch
from sentencepiece import SentencePieceProcessor


def greedy_ctc_decode(
    log_probs: torch.Tensor,  # [B, T, V]
    out_lengths: torch.Tensor,
    blank_id: int,
) -> list[list[int]]:
    preds = torch.argmax(log_probs, dim=-1).cpu()
    out_lengths_cpu = out_lengths.cpu()

    decoded: list[list[int]] = []
    for seq, L in zip(preds, out_lengths_cpu, strict=True):
        T = int(L.item())
        prev = -1
        tokens: list[int] = []
        for p in seq[:T]:
            p_int = int(p.item())
            if p_int != prev and p_int != blank_id:
                tokens.append(p_int)
            prev = p_int
        decoded.append(tokens)
    return decoded


def ctc_ids_to_texts_spm(sp: SentencePieceProcessor, decoded_ids: list[list[int]]) -> list[str]:
    pred_texts: list[str] = []
    for seq in decoded_ids:
        # convention in your repo: blank=0, tokens=sp_id+1
        sp_ids = [i - 1 for i in seq if i > 0]
        pred_texts.append("" if not sp_ids else sp.decode_ids(sp_ids))
    return pred_texts

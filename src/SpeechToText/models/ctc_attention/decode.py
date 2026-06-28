"""CTC + Attention decoding: prefix scoring and joint beam search."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import torch

_LOG_ZERO = -1e10


def _ctc_forward_score(log_probs: torch.Tensor, token_ids: list[int], *, blank_id: int) -> float:
    """Computes the log CTC forward score for a given label sequence.

    Loops over label positions (S=2n+1) interleaved by blanks, allowing
    efficient forward score calculation over prefix tokens during beam search.
    """
    labels: list[int] = []
    for token_id in token_ids:
        labels.extend([blank_id, token_id])
    labels.append(blank_id)
    num_labels = len(labels)
    time_steps = log_probs.size(0)

    alpha = torch.full(
        (time_steps, num_labels),
        _LOG_ZERO,
        device=log_probs.device,
        dtype=log_probs.dtype,
    )
    alpha[0, 0] = log_probs[0, labels[0]]
    if num_labels > 1:
        alpha[0, 1] = log_probs[0, labels[1]]

    for t in range(1, time_steps):
        for s in range(num_labels):
            lp = log_probs[t, labels[s]]
            candidates = [alpha[t - 1, s]]
            if s >= 1:
                candidates.append(alpha[t - 1, s - 1])
            if s >= 2 and labels[s] != blank_id and labels[s] != labels[s - 2]:
                candidates.append(alpha[t - 1, s - 2])
            alpha[t, s] = torch.logsumexp(torch.stack(candidates), dim=0) + lp

    return float(torch.logsumexp(alpha[time_steps - 1, num_labels - 2 : num_labels], dim=0).item())


def ctc_prefix_score(
    log_probs: torch.Tensor,
    out_len: int,
    token_ids: list[int],
    *,
    blank_id: int,
) -> float:
    """Return log P_ctc(prefix=token_ids | X)."""
    if not token_ids:
        truncated = log_probs[:out_len]
        blank_lp = truncated[:, blank_id]
        state = blank_lp[0]
        for t in range(1, truncated.size(0)):
            state = (
                torch.logsumexp(
                    torch.stack([state, torch.tensor(_LOG_ZERO, device=truncated.device)]),
                    dim=0,
                )
                + blank_lp[t]
            )
        return float(state.item())
    return _ctc_forward_score(log_probs[:out_len], token_ids, blank_id=blank_id)


def ctc_prefix_score_bruteforce(
    log_probs: torch.Tensor,
    out_len: int,
    token_ids: list[int],
    *,
    blank_id: int,
) -> float:
    """Alias kept for tests (same implementation)."""
    return ctc_prefix_score(log_probs, out_len, token_ids, blank_id=blank_id)


class CtcAttentionNet(Protocol):
    def decode(
        self,
        enc: torch.Tensor,
        out_lengths: torch.Tensor,
        decoder_input: torch.Tensor,
    ) -> torch.Tensor: ...


@dataclass(order=True)
class _BeamHyp:
    sort_score: float
    attn_score: float = field(compare=False)
    tokens: list[int] = field(compare=False)
    finished: bool = field(compare=False, default=False)


def attention_greedy_decode(
    net: CtcAttentionNet,
    enc: torch.Tensor,
    out_lengths: torch.Tensor,
    *,
    bos_id: int,
    eos_id: int,
    max_len: int,
) -> list[int]:
    """Autoregressive greedy decode from the attention decoder (batch size 1)."""
    device = enc.device
    prefix = torch.tensor([[bos_id]], device=device, dtype=torch.long)
    output_tokens: list[int] = []

    for _ in range(max_len):
        log_probs = net.decode(enc, out_lengths, prefix)
        next_id = int(log_probs[0, -1].argmax().item())
        if next_id == eos_id:
            break
        output_tokens.append(next_id)
        prefix = torch.cat(
            [prefix, torch.tensor([[next_id]], device=device, dtype=torch.long)],
            dim=1,
        )
    return output_tokens


def joint_beam_search(
    net: CtcAttentionNet,
    enc: torch.Tensor,
    out_lengths: torch.Tensor,
    ctc_log_probs: torch.Tensor,
    *,
    alpha: float,
    beam_size: int,
    top_k: int | None,
    bos_id: int,
    eos_id: int,
    blank_id: int,
    max_len: int,
    length_penalty: float = 0.0,
) -> list[int]:
    """Joint CTC/Attention beam search with TOP-N pruning on attention logits."""
    if enc.size(0) != 1:
        raise ValueError("joint_beam_search currently supports batch size 1")

    candidate_k = max(1, top_k if top_k is not None else beam_size)
    time_steps = int(out_lengths[0].item())
    log_probs = ctc_log_probs[0, :time_steps]

    beams: list[_BeamHyp] = [
        _BeamHyp(sort_score=0.0, attn_score=0.0, tokens=[], finished=False),
    ]

    for _step in range(max_len):
        candidates: list[_BeamHyp] = []
        for hyp in beams:
            if hyp.finished:
                candidates.append(hyp)
                continue

            prefix = torch.tensor([[bos_id, *hyp.tokens]], device=enc.device, dtype=torch.long)
            dec_log_probs = net.decode(enc, out_lengths, prefix)
            step_log_probs = dec_log_probs[0, -1]

            top_scores, top_ids = torch.topk(
                step_log_probs, k=min(candidate_k, step_log_probs.numel())
            )
            for log_p_attn, token_id in zip(top_scores.tolist(), top_ids.tolist(), strict=True):
                token = int(token_id)
                if token == eos_id:
                    ctc_score = ctc_prefix_score(
                        log_probs, time_steps, hyp.tokens, blank_id=blank_id
                    )
                    length = max(len(hyp.tokens), 1)
                    total = (
                        (1.0 - alpha) * hyp.attn_score + alpha * ctc_score - length_penalty * length
                    )
                    candidates.append(
                        _BeamHyp(
                            sort_score=total,
                            attn_score=hyp.attn_score,
                            tokens=list(hyp.tokens),
                            finished=True,
                        )
                    )
                    continue

                new_tokens = [*hyp.tokens, token]
                new_attn = hyp.attn_score + float(log_p_attn)
                ctc_score = ctc_prefix_score(log_probs, time_steps, new_tokens, blank_id=blank_id)
                length = max(len(new_tokens), 1)
                total = (1.0 - alpha) * new_attn + alpha * ctc_score - length_penalty * length
                candidates.append(
                    _BeamHyp(
                        sort_score=total,
                        attn_score=new_attn,
                        tokens=new_tokens,
                        finished=False,
                    )
                )

        candidates.sort(key=lambda item: item.sort_score, reverse=True)
        beams = candidates[:beam_size]
        if beams and all(item.finished for item in beams):
            break

    if not beams:
        return []

    best = max(beams, key=lambda item: item.sort_score)
    return best.tokens

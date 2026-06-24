from __future__ import annotations

from collections.abc import Sequence
from typing import Any, TypeAlias, cast

import editdistance
import numpy as np
import torch
from sentencepiece import SentencePieceProcessor

from SpeechToText.models.common import greedy_ctc_decode

MetricKey: TypeAlias = tuple[str, int | None, float | None, float | None, str]
MetricState: TypeAlias = dict[str, float]


def edit_distance(tokens_ref: Sequence[str], tokens_hyp: Sequence[str]) -> int:
    """Return Levenshtein distance between token sequences."""
    return int(editdistance.eval(tokens_ref, tokens_hyp))


def compute_wer_cer(ref_text: str, hyp_text: str) -> tuple[int, int, int, int]:
    """Return (wer_num, wer_den, cer_num, cer_den) for ref and hyp."""
    ref_clean = ref_text.strip().lower()
    hyp_clean = hyp_text.strip().lower()
    if not ref_clean:
        return 0, 0, 0, 0

    ref_words = ref_clean.split()
    hyp_words = hyp_clean.split()
    wer_num = edit_distance(ref_words, hyp_words)
    wer_den = len(ref_words)

    ref_chars = list(ref_clean)
    hyp_chars = list(hyp_clean)
    cer_num = edit_distance(ref_chars, hyp_chars)
    cer_den = len(ref_chars)

    return wer_num, wer_den, cer_num, cer_den


def greedy_decode_single(
    log_probs: torch.Tensor,
    out_len: int,
    tokenizer: SentencePieceProcessor,
    blank_id: int,
) -> str:
    """Greedy CTC decoding for a single sequence."""
    truncated = log_probs[:out_len].unsqueeze(0)
    out_lengths = torch.tensor([out_len], device=truncated.device, dtype=torch.long)

    decoded_ids_batch = greedy_ctc_decode(truncated, out_lengths, blank_id=blank_id)
    token_ids = decoded_ids_batch[0]

    sp_ids = [idx - 1 for idx in token_ids if idx > 0]
    if not sp_ids:
        return ""
    return cast(str, tokenizer.decode_ids(sp_ids))


def decode_batch_with_greedy(
    batch_log_probs: torch.Tensor,
    batch_lengths: torch.Tensor,
    batch_refs: Sequence[str],
    batch_langs: Sequence[str],
    tokenizer: SentencePieceProcessor,
    blank_id: int,
    metrics: dict[MetricKey, MetricState],
) -> None:
    """Run greedy decoding for a batch and update metrics."""
    batch_size = batch_log_probs.size(0)

    for idx in range(batch_size):
        log_probs = batch_log_probs[idx]
        out_len = int(batch_lengths[idx].item())
        ref = batch_refs[idx]
        lang = batch_langs[idx]

        hyp = greedy_decode_single(log_probs, out_len, tokenizer, blank_id=blank_id)
        wernum, werden, cernum, cerden = compute_wer_cer(ref, hyp)

        if werden == 0 and cerden == 0:
            continue

        for lang_key in ("all", lang):
            key = ("greedy", None, None, None, lang_key)
            state = metrics.setdefault(
                key, {"wer_num": 0.0, "wer_den": 0.0, "cer_num": 0.0, "cer_den": 0.0, "count": 0.0}
            )
            state["wer_num"] += float(wernum)
            state["wer_den"] += float(werden)
            state["cer_num"] += float(cernum)
            state["cer_den"] += float(cerden)
            state["count"] += 1.0


def collect_probs_for_beam(
    batch_log_probs: torch.Tensor,
    batch_lengths: torch.Tensor,
    batch_refs: Sequence[str],
    batch_langs: Sequence[str],
    vocab_size_with_blank: int,
) -> tuple[list[Any], list[str], list[str]]:
    """Prepare per-example probability matrices for pyctcdecode."""
    probs_list: list[Any] = []
    refs_list: list[str] = []
    langs_list: list[str] = []

    batch_size = batch_log_probs.size(0)
    for i in range(batch_size):
        out_len = int(batch_lengths[i].item())
        logp = batch_log_probs[i, :out_len, :vocab_size_with_blank]
        probs = torch.exp(logp).detach().cpu().to(torch.float32).numpy()
        probs_list.append(cast(np.ndarray, probs))
        refs_list.append(str(batch_refs[i]))
        langs_list.append(str(batch_langs[i]))

    return probs_list, refs_list, langs_list


def decode_batch_with_beam(
    decode_types: Sequence[str],
    beam_widths: Sequence[int],
    alphas: Sequence[float],
    betas: Sequence[float],
    probs_per_example: list[Any],
    refs: Sequence[str],
    langs: Sequence[str],
    decoder_ctc: Any | None,
    decoders_kenlm: dict[tuple[float, float], Any],
    pool: Any | None,
    metrics: dict[tuple[str, int | None, float | None, float | None, str], dict[str, float]],
) -> None:
    """Run beam and beam+KenLM decoding and update metrics."""
    if not probs_per_example:
        return

    def _decode_batch(decoder: Any, probs: list[Any], beam_width: int) -> list[str]:
        if pool is not None:
            return cast(list[str], decoder.decode_batch(pool, probs, beam_width=beam_width))
        return [cast(str, decoder.decode(p, beam_width=beam_width)) for p in probs]

    if "beam" in decode_types and decoder_ctc is not None:
        for beam_width in beam_widths:
            hyps = _decode_batch(decoder_ctc, probs_per_example, beam_width)

            for hyp, ref, lang in zip(hyps, refs, langs, strict=False):
                wernum, werden, cernum, cerden = compute_wer_cer(ref, hyp)
                if werden == 0 and cerden == 0:
                    continue

                for lang_key in ("all", lang):
                    key: MetricKey = ("beam", beam_width, None, None, lang_key)
                    state = metrics.setdefault(
                        key,
                        {
                            "wer_num": 0.0,
                            "wer_den": 0.0,
                            "cer_num": 0.0,
                            "cer_den": 0.0,
                            "count": 0.0,
                        },
                    )
                    state["wer_num"] += float(wernum)
                    state["wer_den"] += float(werden)
                    state["cer_num"] += float(cernum)
                    state["cer_den"] += float(cerden)
                    state["count"] += 1.0

    if "beam_kenlm" in decode_types and decoders_kenlm:
        for alpha in alphas:
            for beta in betas:
                decoder = decoders_kenlm.get((alpha, beta))
                if decoder is None:
                    continue

                for beam_width in beam_widths:
                    hyps = _decode_batch(decoder, probs_per_example, beam_width)

                    for hyp, ref, lang in zip(hyps, refs, langs, strict=False):
                        wernum, werden, cernum, cerden = compute_wer_cer(ref, hyp)
                        if werden == 0 and cerden == 0:
                            continue

                        for lang_key in ("all", lang):
                            key = ("beam_kenlm", beam_width, alpha, beta, lang_key)
                            state = metrics.setdefault(
                                key,
                                {
                                    "wer_num": 0.0,
                                    "wer_den": 0.0,
                                    "cer_num": 0.0,
                                    "cer_den": 0.0,
                                    "count": 0.0,
                                },
                            )
                            state["wer_num"] += float(wernum)
                            state["wer_den"] += float(werden)
                            state["cer_num"] += float(cernum)
                            state["cer_den"] += float(cerden)
                            state["count"] += 1.0

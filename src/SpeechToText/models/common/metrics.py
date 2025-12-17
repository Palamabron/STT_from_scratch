from __future__ import annotations

from jiwer import cer as jiwer_cer
from jiwer import wer as jiwer_wer


def wer_cer(refs: list[str], hyps: list[str]) -> tuple[float, float]:
    return float(jiwer_wer(refs, hyps)), float(jiwer_cer(refs, hyps))


def wer_cer_by_lang(
    refs: list[str],
    hyps: list[str],
    langs: list[str] | None,
) -> dict[str, float]:
    if langs is None:
        langs = ["unknown"] * len(refs)

    overall_wer, overall_cer = wer_cer(refs, hyps)
    out: dict[str, float] = {"wer/overall": overall_wer, "cer/overall": overall_cer}

    en_refs: list[str] = []
    en_hyps: list[str] = []
    pl_refs: list[str] = []
    pl_hyps: list[str] = []

    for lang, r, h in zip(langs, refs, hyps, strict=True):
        if lang == "en":
            en_refs.append(r)
            en_hyps.append(h)
        elif lang == "pl":
            pl_refs.append(r)
            pl_hyps.append(h)

    if en_refs:
        w, c = wer_cer(en_refs, en_hyps)
        out["wer/en"] = w
        out["cer/en"] = c
    if pl_refs:
        w, c = wer_cer(pl_refs, pl_hyps)
        out["wer/pl"] = w
        out["cer/pl"] = c

    return out

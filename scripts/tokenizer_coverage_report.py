from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import sentencepiece as spm
import tyro
from loguru import logger

POLISH_TAIL_CHARS = "ąćęłńóśźżĄĆĘŁŃÓŚŹŻ"


@dataclass(slots=True)
class CoverageReportConfig:
    manifests: tuple[str, ...]
    model_path: str
    output_json: str = "results/tokenizer_coverage.json"


def _load_texts_by_lang(manifests: tuple[str, ...]) -> dict[str, list[str]]:
    by_lang: dict[str, list[str]] = defaultdict(list)
    for manifest in manifests:
        path = Path(manifest)
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                obj = json.loads(line)
                text = (obj.get("text") or "").strip()
                if not text:
                    continue
                lang = str(obj.get("language", "unknown"))
                by_lang[lang].append(text)
    return by_lang


def _char_coverage(texts: list[str], chars: str) -> dict[str, dict[str, float | int]]:
    counts: Counter[str] = Counter()
    for text in texts:
        counts.update(text)
    report: dict[str, dict[str, float | int]] = {}
    for char in chars:
        report[char] = {
            "count": int(counts.get(char, 0)),
            "present": bool(counts.get(char, 0)),
        }
    return report


def _piece_coverage(processor: spm.SentencePieceProcessor, texts: list[str]) -> dict[str, int]:
    unknown = 0
    total = 0
    for text in texts:
        ids = processor.encode(text, out_type=int)
        total += len(ids)
        unknown += sum(1 for token_id in ids if token_id == processor.unk_id())
    return {"tokens": total, "unk_tokens": unknown}


def main(cfg: CoverageReportConfig) -> None:
    by_lang = _load_texts_by_lang(cfg.manifests)
    processor = spm.SentencePieceProcessor()
    processor.load(cfg.model_path)

    report: dict[str, object] = {
        "model_path": cfg.model_path,
        "manifests": list(cfg.manifests),
        "languages": {},
        "notes": (
            "Character balancing by total char count can oversample short English tokens "
            "relative to Polish digraphs. Inspect polish_tail_chars and unk_tokens/pl after "
            "training an 8k balanced tokenizer."
        ),
    }

    languages = report["languages"]
    assert isinstance(languages, dict)

    for lang, texts in sorted(by_lang.items()):
        char_stats = _char_coverage(texts, POLISH_TAIL_CHARS if lang == "pl" else "")
        piece_stats = _piece_coverage(processor, texts)
        languages[lang] = {
            "utterances": len(texts),
            "characters": sum(len(text) for text in texts),
            "piece_stats": piece_stats,
            "polish_tail_chars": char_stats if lang == "pl" else {},
        }
        if lang == "pl":
            logger.info(
                "[{}] utterances={} chars={} unk_tokens={}/{}",
                lang,
                len(texts),
                sum(len(text) for text in texts),
                piece_stats["unk_tokens"],
                piece_stats["tokens"],
            )

    output_path = Path(cfg.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Wrote tokenizer coverage report to {}", output_path)


if __name__ == "__main__":
    main(tyro.cli(CoverageReportConfig))

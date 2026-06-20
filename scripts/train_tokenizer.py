from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import sentencepiece as spm
import tyro
from loguru import logger


@dataclass(slots=True)
class TokenizerConfig:
    manifests: tuple[str, ...]
    corpus_out: str
    model_prefix: str
    vocab_size: int = 4096
    model_type: str = "unigram"
    character_coverage: float = 1.0
    max_lines: int | None = None
    shuffle: bool = False
    seed: int = 42
    balance_languages: bool = False
    control_symbols: tuple[str, ...] = ("<pad>", "<s>", "</s>")


def _load_manifest_lines(manifests: tuple[str, ...]) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for manifest in manifests:
        path = Path(manifest)
        if not path.exists():
            raise FileNotFoundError(f"Manifest not found: {path}")
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                obj = json.loads(line)
                text = (obj.get("text") or "").strip()
                if not text:
                    continue
                lang = str(obj.get("language", "unknown"))
                rows.append((lang, text.replace("\n", " ")))
    return rows


def _balance_by_character_count(rows: list[tuple[str, str]], seed: int) -> list[str]:
    by_lang: dict[str, list[str]] = defaultdict(list)
    for lang, text in rows:
        by_lang[lang].append(text)

    if len(by_lang) < 2:
        return [text for _, text in rows]

    import random

    rng = random.Random(seed)
    target_chars = max(sum(len(text) for text in texts) for texts in by_lang.values())
    balanced: list[str] = []

    for lang, texts in by_lang.items():
        if not texts:
            continue
        rng.shuffle(texts)
        corpus: list[str] = []
        char_count = 0
        index = 0
        while char_count < target_chars:
            corpus.append(texts[index % len(texts)])
            char_count += len(texts[index % len(texts)])
            index += 1
        balanced.extend(corpus)
        logger.info("Balanced language {} to {} chars (target={})", lang, char_count, target_chars)

    rng.shuffle(balanced)
    return balanced


def _build_corpus(
    manifests: tuple[str, ...],
    corpus_out: Path,
    max_lines: int | None,
    *,
    balance_languages: bool,
    seed: int,
) -> int:
    corpus_out.parent.mkdir(parents=True, exist_ok=True)
    rows = _load_manifest_lines(manifests)
    texts = _balance_by_character_count(rows, seed) if balance_languages else [t for _, t in rows]

    with corpus_out.open("w", encoding="utf-8") as handle:
        for index, text in enumerate(texts):
            if max_lines is not None and index >= max_lines:
                break
            handle.write(text + "\n")
    return min(len(texts), max_lines or len(texts))


def main(cfg: TokenizerConfig) -> None:
    corpus_path = Path(cfg.corpus_out)
    line_count = _build_corpus(
        cfg.manifests,
        corpus_path,
        cfg.max_lines,
        balance_languages=cfg.balance_languages,
        seed=cfg.seed,
    )

    out_prefix = Path(cfg.model_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    args = (
        f"--input={corpus_path} "
        f"--model_prefix={out_prefix} "
        f"--vocab_size={cfg.vocab_size} "
        f"--model_type={cfg.model_type} "
        f"--character_coverage={cfg.character_coverage} "
        f"--control_symbols={','.join(cfg.control_symbols)} "
        f"--bos_id=1 --eos_id=2 --pad_id=3 --unk_id=0"
    )

    spm.SentencePieceTrainer.Train(args)

    logger.info("Wrote corpus lines: {}", line_count)
    logger.info("Saved: {}.model", out_prefix)
    logger.info("Saved: {}.vocab", out_prefix)


if __name__ == "__main__":
    main(tyro.cli(TokenizerConfig))

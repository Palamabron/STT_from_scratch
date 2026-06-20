from __future__ import annotations

import json
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
    control_symbols: tuple[str, ...] = ("<pad>", "<s>", "</s>")


def _build_corpus(manifests: tuple[str, ...], corpus_out: Path, max_lines: int | None) -> int:
    corpus_out.parent.mkdir(parents=True, exist_ok=True)

    n = 0
    with corpus_out.open("w", encoding="utf-8") as w:
        for m in manifests:
            mp = Path(m)
            if not mp.exists():
                raise FileNotFoundError(f"Manifest not found: {mp}")
            with mp.open("r", encoding="utf-8") as f:
                for line in f:
                    obj = json.loads(line)
                    txt = (obj.get("text") or "").strip()
                    if not txt:
                        continue
                    w.write(txt.replace("\n", " ") + "\n")
                    n += 1
                    if max_lines is not None and n >= max_lines:
                        return n
    return n


def main(cfg: TokenizerConfig) -> None:
    corpus_path = Path(cfg.corpus_out)
    n = _build_corpus(cfg.manifests, corpus_path, cfg.max_lines)

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

    logger.info("Wrote corpus lines: {}", n)
    logger.info("Saved: {}.model", out_prefix)
    logger.info("Saved: {}.vocab", out_prefix)


if __name__ == "__main__":
    main(tyro.cli(TokenizerConfig))

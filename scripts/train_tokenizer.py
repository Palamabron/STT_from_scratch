from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import sentencepiece as spm
import tyro


@dataclass
class TokenizerConfig:
    corpus_path: str
    model_prefix: str
    vocab_size: int = 1000
    model_type: str = "unigram"
    character_coverage: float = 1.0


def main(cfg: TokenizerConfig):
    corpus = Path(cfg.corpus_path)
    if not corpus.exists():
        raise FileNotFoundError(f"Corpus not found: {corpus}")
    out_prefix = Path(cfg.model_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    args = (
        f"--input={corpus} "
        f"--model_prefix={out_prefix} "
        f"--vocab_size={cfg.vocab_size} "
        f"--model_type={cfg.model_type} "
        f"--character_coverage={cfg.character_coverage}"
    )
    spm.SentencePieceTrainer.Train(args)


if __name__ == "__main__":
    cfg = tyro.cli(TokenizerConfig)
    main(cfg)

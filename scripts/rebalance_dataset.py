from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import tyro


@dataclass
class RebalanceConfig:
    in_manifests: list[Path]
    out_train: Path
    out_val: Path
    train_frac: float = 0.7
    seed: int = 42
    balance_langs: bool = False


def main(cfg: RebalanceConfig):
    random.seed(cfg.seed)

    all_examples: list[dict] = []
    for p in cfg.in_manifests:
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                all_examples.append(json.loads(line))

    by_lang: dict[str, list[dict]] = defaultdict(list)
    for ex in all_examples:
        lang = ex.get("language", "unknown")
        by_lang[lang].append(ex)

    if cfg.balance_langs:
        min_n = min(len(v) for v in by_lang.values())
    else:
        min_n = None

    train, val = [], []
    for lang, items in by_lang.items():
        random.shuffle(items)
        if min_n is not None and len(items) > min_n:
            items = items[:min_n]
        n_total = len(items)
        n_train = int(n_total * cfg.train_frac)
        train.extend(items[:n_train])
        val.extend(items[n_train:])
        print(f"[{lang}] total={n_total}, train={n_train}, val={n_total - n_train}")

    def write_manifest(path: Path, data: list[dict]):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for ex in data:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    write_manifest(cfg.out_train, train)
    write_manifest(cfg.out_val, val)
    print(f"Final train={len(train)}, val={len(val)}")


if __name__ == "__main__":
    cfg = tyro.cli(RebalanceConfig)
    main(cfg)

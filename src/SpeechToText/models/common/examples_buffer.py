from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ExamplesBuffer:
    per_lang: int = 2
    buf: dict[str, list[tuple[str, str]]] = field(default_factory=lambda: {"en": [], "pl": []})

    def add(self, lang: str, ref: str, hyp: str) -> None:
        if lang not in self.buf:
            return
        if len(self.buf[lang]) < self.per_lang:
            self.buf[lang].append((ref, hyp))

    def pop_all(self) -> dict[str, list[tuple[str, str]]]:
        out = self.buf
        self.buf = {"en": [], "pl": []}
        return out

import json
from pathlib import Path

from SpeechToText.dataset import (
    DataConfig,
    FeatureConfig,
    FilterConfig,
    LoaderConfig,
    ManifestDataset,
    ManifestPaths,
)


class _FakeTokenizer:
    def encode(self, text: str, out_type: int = int) -> list[int]:
        return list(range(max(1, len(text.split()))))


def test_manifest_dataset_drops_ctc_too_long_without_loading_audio(tmp_path: Path) -> None:
    manifest = tmp_path / "train.jsonl"
    long_text = "word " * 200
    entries = [
        {"audio_filepath": "/tmp/ok.wav", "text": "hi there", "duration": 2.0, "language": "en"},
        {"audio_filepath": "/tmp/bad.wav", "text": long_text, "duration": 0.2, "language": "en"},
    ]
    manifest.write_text("\n".join(json.dumps(entry) for entry in entries) + "\n", encoding="utf-8")

    config = DataConfig(
        manifests=ManifestPaths(train=str(manifest), val=str(manifest)),
        tokenizer_model="unused.model",
        features=FeatureConfig(),
        loader=LoaderConfig(),
        filter=FilterConfig(subsampling_factor=8, max_duration=16.0),
    )

    tokenizer = _FakeTokenizer()
    dataset = ManifestDataset(str(manifest), tokenizer, config, split="train")  # type: ignore[arg-type]
    assert len(dataset) == 1
    assert dataset.texts == ["hi there"]

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
import tyro
from loguru import logger
from sentencepiece import SentencePieceProcessor


@dataclass(slots=True)
class InitEncoderConfig:
    source_checkpoint: str
    tokenizer_model: str
    target: Literal["rnnt", "ctc_attention"]
    output: str


def _extract_encoder_state(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    prefix = "net.encoder."
    return {key: value for key, value in state_dict.items() if key.startswith(prefix)}


def _build_target_module(
    target: Literal["rnnt", "ctc_attention"],
    *,
    sp: SentencePieceProcessor,
) -> torch.nn.Module:
    vocab_size = int(sp.get_piece_size()) + 1

    if target == "rnnt":
        from SpeechToText.models.tdt.lit import LitFastConformerTDT
        from SpeechToText.models.tdt.train import TrainConfig

        config = TrainConfig()
        config.model.decoder.vocab_size = vocab_size
        config.model.joint.vocab_size = vocab_size
        return LitFastConformerTDT(config, sp=sp, vocab_size=vocab_size)

    from SpeechToText.models.ctc_attention.lit import LitFastConformerCTCAttention
    from SpeechToText.models.ctc_attention.train import TrainConfig

    config = TrainConfig()
    return LitFastConformerCTCAttention(config=config, sp=sp)


def main(cfg: InitEncoderConfig) -> None:
    source_path = Path(cfg.source_checkpoint)
    if not source_path.is_file():
        raise FileNotFoundError(f"Source checkpoint not found: {source_path}")

    sp = SentencePieceProcessor()
    sp.load(cfg.tokenizer_model)

    source_ckpt = torch.load(source_path, map_location="cpu", weights_only=False)
    source_state = source_ckpt.get("state_dict", source_ckpt)
    encoder_weights = _extract_encoder_state(source_state)
    if not encoder_weights:
        raise ValueError(f"No net.encoder.* weights found in {source_path}")

    target_module = _build_target_module(cfg.target, sp=sp)
    target_state = target_module.state_dict()

    copied = 0
    skipped = 0
    for key, value in encoder_weights.items():
        if key not in target_state:
            logger.warning("Skipping encoder key missing in target: {}", key)
            skipped += 1
            continue
        if target_state[key].shape != value.shape:
            logger.warning(
                "Shape mismatch for {}: source {} vs target {}",
                key,
                tuple(value.shape),
                tuple(target_state[key].shape),
            )
            skipped += 1
            continue
        target_state[key] = value.clone()
        copied += 1

    target_module.load_state_dict(target_state)

    output_path = Path(cfg.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": target_module.state_dict(), "epoch": 0}, output_path)

    logger.info(
        "Copied {} encoder tensors ({} skipped) from {} into {} head -> {}",
        copied,
        skipped,
        source_path.name,
        cfg.target,
        output_path,
    )
    logger.info(
        "Start training with: --ckpt_path {} (decoder/joint heads remain randomly initialized)",
        output_path,
    )


if __name__ == "__main__":
    main(tyro.cli(InitEncoderConfig))

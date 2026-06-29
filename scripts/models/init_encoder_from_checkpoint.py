from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import lightning.pytorch as pl
import torch
import tyro
from loguru import logger
from sentencepiece import SentencePieceProcessor

from SpeechToText.models.common.checkpoint_io import load_lightning_checkpoint


@dataclass(slots=True)
class InitEncoderConfig:
    source_checkpoint: str
    tokenizer_model: str
    target: Literal["rnnt", "ctc_attention"]
    output: str
    use_tdt: bool = False


def _extract_encoder_state(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    prefix = "net.encoder."
    return {key: value for key, value in state_dict.items() if key.startswith(prefix)}


def _extract_ctc_attention_head_state(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Map CTC or CTC+Attention checkpoint keys to CTC+Attention head keys."""
    mapped: dict[str, torch.Tensor] = {}

    for key, value in state_dict.items():
        if key.startswith("net.ctc_proj."):
            mapped[key] = value

    for key, value in state_dict.items():
        if key.startswith("net.aux_projs."):
            mapped[key] = value

    if any(key.startswith("net.ctc_proj.") for key in mapped):
        return mapped

    proj = state_dict.get("net.proj.weight")
    proj_bias = state_dict.get("net.proj.bias")
    if proj is not None:
        mapped["net.ctc_proj.weight"] = proj
    if proj_bias is not None:
        mapped["net.ctc_proj.bias"] = proj_bias

    return mapped


def _copy_weights(
    source_weights: dict[str, torch.Tensor],
    target_state: dict[str, torch.Tensor],
    *,
    label: str,
) -> tuple[int, int]:
    copied = 0
    skipped = 0
    for key, value in source_weights.items():
        if key not in target_state:
            logger.warning("Skipping {} key missing in target: {}", label, key)
            skipped += 1
            continue
        if target_state[key].shape != value.shape:
            logger.warning(
                "Shape mismatch for {} {}: source {} vs target {}",
                label,
                key,
                tuple(value.shape),
                tuple(target_state[key].shape),
            )
            skipped += 1
            continue
        target_state[key] = value.clone()
        copied += 1
    return copied, skipped


def _build_target_module(
    target: Literal["rnnt", "ctc_attention"],
    *,
    sp: SentencePieceProcessor,
    source_ckpt: dict[str, object] | None = None,
    use_tdt: bool = False,
) -> torch.nn.Module:
    vocab_size = int(sp.get_piece_size()) + 1

    if target == "rnnt":
        from SpeechToText.models.tdt.config import TrainConfig
        from SpeechToText.models.tdt.lit import LitFastConformerTDT

        config = TrainConfig()
        if source_ckpt is not None:
            source_config = source_ckpt.get("hyper_parameters", {}).get("config")
            if source_config is not None and hasattr(source_config, "model"):
                config.model.encoder = source_config.model.encoder
        config.use_tdt = bool(use_tdt)
        config.model.joint.use_tdt = bool(use_tdt)
        config.model.decoder.vocab_size = vocab_size
        config.model.joint.vocab_size = vocab_size
        config.model.decoder.d_model = int(config.model.encoder.d_model)
        config.model.joint.enc_d = int(config.model.encoder.d_model)
        config.model.joint.pred_d = int(config.model.encoder.d_model)
        return LitFastConformerTDT(config, sp=sp, vocab_size=vocab_size)

    from SpeechToText.models.ctc_attention.config import TrainConfig
    from SpeechToText.models.ctc_attention.lit import LitFastConformerCTCAttention

    config = TrainConfig()
    if source_ckpt is not None:
        source_config = source_ckpt.get("hyper_parameters", {}).get("config")
        if source_config is not None and hasattr(source_config, "model"):
            config.model.encoder = source_config.model.encoder
            config.model.aux_layer = source_config.model.aux_layer
            config.model.aux_interval = source_config.model.aux_interval
    return LitFastConformerCTCAttention(config=config, sp=sp)


def _save_lightning_checkpoint(module: pl.LightningModule, output_path: Path) -> None:
    checkpoint: dict[str, Any] = {
        "epoch": 0,
        "global_step": 0,
        "state_dict": module.state_dict(),
        "hyper_parameters": {"config": module.config},
    }
    module.on_save_checkpoint(checkpoint)
    checkpoint["pytorch-lightning_version"] = pl.__version__
    torch.save(checkpoint, output_path)


def main(cfg: InitEncoderConfig) -> None:
    source_path = Path(cfg.source_checkpoint)
    if not source_path.is_file():
        raise FileNotFoundError(f"Source checkpoint not found: {source_path}")

    sp = SentencePieceProcessor()
    sp.load(cfg.tokenizer_model)

    source_ckpt = load_lightning_checkpoint(source_path)
    source_state = source_ckpt.get("state_dict", source_ckpt)
    encoder_weights = _extract_encoder_state(source_state)
    if not encoder_weights:
        raise ValueError(f"No net.encoder.* weights found in {source_path}")

    target_module = _build_target_module(
        cfg.target, sp=sp, source_ckpt=source_ckpt, use_tdt=cfg.use_tdt
    )
    target_state = target_module.state_dict()

    enc_copied, enc_skipped = _copy_weights(encoder_weights, target_state, label="encoder")

    ctc_copied = 0
    ctc_skipped = 0
    if cfg.target == "ctc_attention":
        ctc_head_weights = _extract_ctc_attention_head_state(source_state)
        if not ctc_head_weights:
            logger.warning("No CTC head weights (net.proj / net.aux_projs) in {}", source_path)
        else:
            ctc_copied, ctc_skipped = _copy_weights(
                ctc_head_weights, target_state, label="ctc_heads"
            )

    target_module.load_state_dict(target_state)

    if cfg.target == "ctc_attention" and ctc_copied == 0:
        raise ValueError(
            f"No CTC head weights were copied from {source_path}. "
            "Source must contain net.proj.* or net.ctc_proj.* keys."
        )

    output_path = Path(cfg.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _save_lightning_checkpoint(target_module, output_path)

    logger.info(
        "Copied {} encoder tensors ({} skipped), {} ctc_head tensors ({} skipped) "
        "from {} into {} -> {}",
        enc_copied,
        enc_skipped,
        ctc_copied,
        ctc_skipped,
        source_path.name,
        cfg.target,
        output_path,
    )
    if cfg.target == "ctc_attention":
        logger.info(
            "Start training with: --ckpt_path {} (attention decoder remains randomly initialized)",
            output_path,
        )
    else:
        logger.info(
            "Start training with: --ckpt_path {} (decoder/joint heads remain randomly initialized)",
            output_path,
        )


if __name__ == "__main__":
    main(tyro.cli(InitEncoderConfig))

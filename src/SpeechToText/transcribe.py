from __future__ import annotations

from dataclasses import dataclass

import torch
import torchaudio
import tyro
from loguru import logger
from sentencepiece import SentencePieceProcessor

from SpeechToText.models.common.inference import ModelType, load_lit_module, transcribe_batch


@dataclass
class TranscribeConfig:
    checkpoint: str
    tokenizer_model: str
    sample_rate: int = 16_000
    device: str = "auto"
    model_type: ModelType = "auto"
    val_max_symbols_per_t: int = 4


def get_device(device_str: str) -> torch.device:
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_str)


@torch.inference_mode()
def transcribe_files(cfg: TranscribeConfig, audio_paths: list[str]) -> list[str]:
    device = get_device(cfg.device)
    sp = SentencePieceProcessor()
    sp.load(cfg.tokenizer_model)

    model, resolved_type = load_lit_module(
        cfg.checkpoint,
        sp=sp,
        model_type=cfg.model_type,
    )
    model.eval()
    model.to(device)
    logger.info("Loaded model type: {}", resolved_type)

    transcripts: list[str] = []
    for path in audio_paths:
        wav, sr = torchaudio.load(path)
        if wav.dim() == 2 and wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sr != cfg.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, cfg.sample_rate)

        audio = wav.squeeze(0).unsqueeze(0).to(device)
        audio_lengths = torch.tensor([audio.size(1)], device=device, dtype=torch.long)
        text = transcribe_batch(
            model,
            audio,
            audio_lengths,
            sp=sp,
            model_type=resolved_type,
            val_max_symbols_per_t=cfg.val_max_symbols_per_t,
        )[0]
        transcripts.append(text)

    return transcripts


def main(audio_paths: list[str], cfg: TranscribeConfig) -> None:
    texts = transcribe_files(cfg, audio_paths)
    for path, text in zip(audio_paths, texts, strict=False):
        logger.info("{}: {}", path, text)


if __name__ == "__main__":
    tyro.cli(main)

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import cast

import torch
import torchaudio
import tyro
from loguru import logger
from sentencepiece import SentencePieceProcessor

from .train import LitFastConformerCTC, greedy_decoder


@dataclass
class TranscribeConfig:
    checkpoint: str
    audio: str
    tokenizer_model: str = "models/sp_en_pl_unigram_2k_lower.model"
    device: str = "auto"
    sample_rate: int = 16_000


def load_audio(path: str, target_sr: int, device: torch.device) -> torch.Tensor:
    waveform, sr = torchaudio.load(path)
    if waveform.dim() == 2 and waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
        waveform = cast(torch.Tensor, resampler(waveform))
    waveform = waveform.to(device)
    waveform = waveform.squeeze(0)
    return cast(torch.Tensor, waveform)


def decode_text(
    log_probs: torch.Tensor,
    tokenizer: SentencePieceProcessor,
    blank_token_id: int,
) -> str:
    batch_size, T, _ = log_probs.shape
    out_lengths = torch.full(
        (batch_size,),
        T,
        dtype=torch.long,
        device=log_probs.device,
    )

    decoded_ids_batch = greedy_decoder(
        log_probs,
        out_lengths,
        blank_id=blank_token_id,
    )
    seq = decoded_ids_batch[0]

    sp_ids = [i - 1 for i in seq if i > 0]
    if not sp_ids:
        return ""
    text = tokenizer.decode_ids(sp_ids)
    return cast(str, text)


def load_tokenizer(tokenizer_model_path: str) -> SentencePieceProcessor:
    sp = SentencePieceProcessor()
    sp.load(tokenizer_model_path)
    return sp


def main(cfg: TranscribeConfig) -> None:
    if cfg.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(cfg.device)

    logger.info(f"Using device: {device}")

    tokenizer = load_tokenizer(cfg.tokenizer_model)

    logger.info(f"Loading checkpoint from {cfg.checkpoint}")
    model = LitFastConformerCTC.load_from_checkpoint(
        cfg.checkpoint,
        sp=tokenizer,
    )
    model.eval()
    model.to(device)

    blank_id = 0

    audio_path = Path(cfg.audio)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    waveform = load_audio(str(audio_path), target_sr=cfg.sample_rate, device=device)
    with torch.no_grad():
        batch = waveform.unsqueeze(0)
        log_probs, out_lengths, _ = model(
            batch,
            torch.tensor([batch.size(1)], device=device),
        )

    transcription = decode_text(log_probs, tokenizer, blank_token_id=blank_id)
    print(transcription)


if __name__ == "__main__":
    tyro.cli(main)

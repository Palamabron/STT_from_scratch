from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torchaudio
import tyro
from loguru import logger

from .dataset import get_tokenizer
from .train import LitTranscribeModel, TrainConfig, greedy_decoder


@dataclass
class TranscribeConfig:
    checkpoint: str
    audio: str
    device: str = "auto"
    sample_rate: int = 16_000


def load_audio(path: str, target_sr: int, device: torch.device) -> torch.Tensor:
    waveform, sr = torchaudio.load(path)
    if waveform.dim() == 2 and waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
        waveform = resampler(waveform)
    waveform = waveform.to(device)
    waveform = waveform.squeeze(0)
    return waveform


def decode_text(
    log_probs: torch.Tensor,
    tokenizer,
    blank_token_id: int,
) -> str:
    decoded_ids_batch = greedy_decoder(log_probs, blank_token=blank_token_id)
    seq = decoded_ids_batch[0]

    tokens = []
    for tid in seq:
        token = tokenizer.id_to_token(int(tid))
        if token in ("<pad>", "<blk>", "<unk>", "<s>", "</s>"):
            continue
        tokens.append(token)
    return "".join(tokens)


def main(cfg: TranscribeConfig):
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

    train_cfg = TrainConfig()
    tokenizer = get_tokenizer(train_cfg.tokenizer_path)

    logger.info(f"Loading checkpoint from {cfg.checkpoint}")
    model = LitTranscribeModel.load_from_checkpoint(cfg.checkpoint, cfg=train_cfg)
    model.eval()
    model.to(device)

    blank_id = tokenizer.token_to_id("<blk>")

    audio_path = Path(cfg.audio)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    waveform = load_audio(str(audio_path), target_sr=cfg.sample_rate, device=device)
    with torch.no_grad():
        batch = waveform.unsqueeze(0)
        log_probs, _ = model(batch)

    transcription = decode_text(log_probs, tokenizer, blank_token_id=blank_id)
    print(transcription)


if __name__ == "__main__":
    tyro.cli(main)

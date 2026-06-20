from __future__ import annotations

from dataclasses import dataclass

import torch
import torchaudio
import tyro
from loguru import logger
from sentencepiece import SentencePieceProcessor

from SpeechToText.models.common import ctc_ids_to_texts_spm, greedy_ctc_decode
from SpeechToText.models.ctc_attention.lit import LitFastConformerCTCAttention


@dataclass
class TranscribeConfig:
    checkpoint: str
    tokenizer_model: str
    sample_rate: int = 16_000
    device: str = "auto"


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

    model = LitFastConformerCTCAttention.load_from_checkpoint(
        cfg.checkpoint,
        sp=sp,
        weights_only=False,
    )
    model.eval()
    model.to(device)

    transcripts: list[str] = []
    blank_id = 0

    for path in audio_paths:
        wav, sr = torchaudio.load(path)
        if wav.dim() == 2 and wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sr != cfg.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, cfg.sample_rate)

        audio = wav.squeeze(0).unsqueeze(0).to(device)
        audio_lengths = torch.tensor([audio.size(1)], device=device, dtype=torch.long)

        feats, feat_lens = model.featurizer(audio, audio_lengths)
        out = model(feats, feat_lens, decoder_input=None)

        decoded = greedy_ctc_decode(out.ctc_log_probs, out.out_lengths, blank_id=blank_id)
        text = ctc_ids_to_texts_spm(sp, decoded)[0]
        transcripts.append(text)

    return transcripts


def main(audio_paths: list[str], cfg: TranscribeConfig) -> None:
    texts = transcribe_files(cfg, audio_paths)
    for path, text in zip(audio_paths, texts, strict=False):
        logger.info("{}: {}", path, text)


if __name__ == "__main__":
    tyro.cli(main)

from __future__ import annotations

import torch
import torchaudio
import tyro
from dataset import DataConfig
from sentencepiece import SentencePieceProcessor

from SpeechToText.models.ctc_attention.train import LitFastConformerCTCAttention
from SpeechToText.utils.audio import build_feature_transforms, extract_features
from SpeechToText.utils.decoding import greedy_decode_single


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


def load_model_and_frontend(
    cfg: TranscribeConfig,
) -> tuple[
    LitFastConformerCTCAttention,
    SentencePieceProcessor,
    DataConfig,
    torchaudio.transforms.MelSpectrogram,
    torchaudio.transforms.AmplitudeToDB,
]:
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

    data_config = DataConfig(
        train_manifest="",
        val_manifest="",
        tokenizer_model=cfg.tokenizer_model,
        sample_rate=cfg.sample_rate,
    )
    mel_spec, amplitude_to_db = build_feature_transforms(data_config)
    mel_spec.to(device)
    amplitude_to_db.to(device)

    return model, sp, data_config, mel_spec, amplitude_to_db


@torch.inference_mode()
def transcribe_files(cfg: TranscribeConfig, audio_paths: list[str]) -> list[str]:
    device = get_device(cfg.device)
    model, sp, data_config, mel_spec, amplitude_to_db = load_model_and_frontend(cfg)
    blank_id = 0

    transcripts: list[str] = []
    for path in audio_paths:
        mel, feat_len = extract_features(
            audio_path=path,
            data_config=data_config,
            mel_spec=mel_spec,
            amplitude_to_db=amplitude_to_db,
            device=device,
        )
        feats = mel.unsqueeze(0)
        feat_lengths = torch.tensor([feat_len], device=device, dtype=torch.long)

        outputs = model(feats, feat_lengths)
        log_probs = outputs[0][0]
        out_len = int(outputs[1][0].item())

        text = greedy_decode_single(log_probs, out_len, sp, blank_id=blank_id)
        transcripts.append(text)

    return transcripts


def main(audio_paths: list[str], cfg: TranscribeConfig) -> None:
    texts = transcribe_files(cfg, audio_paths)
    for path, text in zip(audio_paths, texts, strict=False):
        print(f"{path}: {text}")


if __name__ == "__main__":
    tyro.cli(main)

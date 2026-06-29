from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch
import torchaudio
from loguru import logger
from pyctcdecode import build_ctcdecoder
from sentencepiece import SentencePieceProcessor

from SpeechToText.dataset import _pcm_to_float32
from SpeechToText.models.common.inference import (
    ModelType,
    forward_ctc_log_probs,
    load_lit_module,
)
from SpeechToText.streaming import StreamingConfig, StreamingSession

_MODEL_CACHE: dict[str, tuple[torch.nn.Module, str]] = {}
_MODEL_CACHE_LOCK = threading.Lock()
_SPM_PROCESSOR: SentencePieceProcessor | None = None
_KENLM_DECODER: object | None = None

SPM_MODEL_PATH = "models/spm_unigram_2k_trainval.model"
KENLM_MODEL_PATH = "lm/kenlm_en_pl_5gram.arpa"
BEAM_WIDTH = 32
BEAM_ALPHA = 0.5
BEAM_BETA = 1.5
DEFAULT_STREAM_MODEL = "FastConformer CTC+Attn v9"

MODEL_CHECKPOINTS = {
    "FastConformer CTC+Attn v9": "checkpoints/ctc_attn_4090_65m_v9/012-val_wer=0.27.ckpt",
    "FastConformer CTC v9": "checkpoints/ctc_4090_65m_v9/026-val_wer=0.27.ckpt",
    "FastConformer CTC v8": "checkpoints/ctc_4090_65m_v8/078-val_wer=0.27.ckpt",
    "FastConformer TDT (RNN-T)": "checkpoints/tdt_4090_65m/000-val_wer=0.66.ckpt",
}


def get_sentencepiece_processor() -> SentencePieceProcessor:
    """Load and return the global SentencePiece processor."""
    global _SPM_PROCESSOR
    if _SPM_PROCESSOR is None:
        _SPM_PROCESSOR = SentencePieceProcessor()
        _SPM_PROCESSOR.load(SPM_MODEL_PATH)
        logger.info("Loaded global SentencePiece tokenizer from {}", SPM_MODEL_PATH)
    return _SPM_PROCESSOR


def _get_kenlm_decoder(sp: SentencePieceProcessor) -> object | None:
    """Build or return the cached KenLM-backed CTC decoder."""
    global _KENLM_DECODER
    if _KENLM_DECODER is not None:
        return _KENLM_DECODER

    lm_path = Path(KENLM_MODEL_PATH)
    if not lm_path.is_file():
        logger.warning("KenLM model not found at {}; beam decode unavailable", lm_path)
        return None

    labels = [""] + [sp.id_to_piece(index) for index in range(sp.get_piece_size())]
    _KENLM_DECODER = build_ctcdecoder(
        labels=labels,
        kenlm_model_path=str(lm_path),
        alpha=BEAM_ALPHA,
        beta=BEAM_BETA,
    )
    logger.info("Loaded KenLM decoder from {}", lm_path)
    return _KENLM_DECODER


def get_model(model_name: str) -> tuple[torch.nn.Module, str]:
    """Load or retrieve from cache the acoustic model module."""
    global _MODEL_CACHE
    if model_name not in MODEL_CHECKPOINTS:
        raise ValueError(f"Unknown model selection: {model_name}")

    with _MODEL_CACHE_LOCK:
        if model_name not in _MODEL_CACHE:
            ckpt_path = MODEL_CHECKPOINTS[model_name]
            if not Path(ckpt_path).exists():
                raise FileNotFoundError(f"Checkpoint file does not exist: {ckpt_path}")

            sp = get_sentencepiece_processor()
            logger.info("Loading model '{}' from checkpoint '{}'...", model_name, ckpt_path)

            device = "cuda" if torch.cuda.is_available() else "cpu"

            model, resolved_type = load_lit_module(ckpt_path, sp=sp)
            model.to(device)
            model.eval()

            _MODEL_CACHE[model_name] = (model, resolved_type)
            logger.info("Model '{}' loaded successfully on {}!", model_name, device)

        return _MODEL_CACHE[model_name]


def _decode_beam_kenlm(
    model: torch.nn.Module,
    wav: torch.Tensor,
    wav_lens: torch.Tensor,
    *,
    sp: SentencePieceProcessor,
    model_type: ModelType,
) -> str:
    decoder = _get_kenlm_decoder(sp)
    if decoder is None:
        raise FileNotFoundError(
            f"KenLM model not found at {KENLM_MODEL_PATH}. "
            "Use Greedy CTC Decode or place the 5-gram LM file."
        )
    if model_type == "tdt":
        raise ValueError("Beam Search + KenLM is not supported for transducer models.")

    log_probs, out_lengths = forward_ctc_log_probs(model, wav, wav_lens, model_type)
    out_len = int(out_lengths[0].item())
    probs = torch.exp(log_probs[0, :out_len]).detach().cpu().to(torch.float32).numpy()
    return cast(str, cast(Any, decoder).decode(probs, beam_width=BEAM_WIDTH))


def run_offline_transcribe(audio_path: str, model_name: str, decode_mode: str) -> str:
    """Run full-file offline transcription of an audio file on the selected model."""
    try:
        if not audio_path:
            return "Please upload or record an audio file."

        model, resolved_type = get_model(model_name)
        sp = get_sentencepiece_processor()

        wav, sr = torchaudio.load(audio_path)
        if wav.dim() == 2 and wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)

        if sr != 16000:
            wav = torchaudio.functional.resample(wav, sr, 16000)

        wav = _pcm_to_float32(wav.squeeze(0))
        device = next(model.parameters()).device
        wav = wav.to(device)
        wav_lens = torch.tensor([wav.shape[0]], dtype=torch.long, device=device)

        if decode_mode == "Greedy Attention Decode":
            if resolved_type not in ("ctc_attention", "shared"):
                return (
                    "Greedy Attention Decode is only available for CTC+Attention checkpoints. "
                    "Select a hybrid model or choose a CTC decoding mode."
                )

            from SpeechToText.models.common.inference import ctc_attention_special_tokens
            from SpeechToText.models.ctc_attention.decode import (
                CtcAttentionNet,
                attention_greedy_decode,
            )

            featurizer = getattr(model, "featurizer", None)
            if featurizer is None:
                raise AttributeError("Model does not have a 'featurizer' module.")
            feats, feat_lens = featurizer(wav.unsqueeze(0), wav_lens)
            encoder_net = cast(Any, getattr(model, "net", model))
            enc, out_lengths, _ = encoder_net.encode(feats, feat_lens)
            tokens = ctc_attention_special_tokens(model)
            token_ids = attention_greedy_decode(
                cast(CtcAttentionNet, encoder_net),
                enc,
                out_lengths,
                bos_id=tokens.bos_id,
                eos_id=tokens.eos_id,
                max_len=tokens.max_decode_len,
            )
            sp_ids = [tok_id - 1 for tok_id in token_ids if tok_id > 0]
            return "" if not sp_ids else sp.decode_ids(sp_ids)

        if decode_mode == "Beam Search + KenLM 5-gram":
            return _decode_beam_kenlm(
                model,
                wav.unsqueeze(0),
                wav_lens,
                sp=sp,
                model_type=cast(ModelType, resolved_type),
            )

        from SpeechToText.models.common.inference import transcribe_batch

        blank_id = int(getattr(model, "blank_id", 0))
        texts = transcribe_batch(
            model,
            wav.unsqueeze(0),
            wav_lens,
            sp=sp,
            model_type=cast(ModelType, resolved_type),
            blank_id=blank_id,
        )
        return texts[0]

    except Exception as exc:
        logger.exception("Error during offline transcription")
        return f"Error during transcription: {exc}"


def init_streaming_session(model_name: str) -> StreamingSession:
    """Initialize a brand new streaming session for the selected model."""
    model, _ = get_model(model_name)
    sp = get_sentencepiece_processor()
    config = StreamingConfig()
    return StreamingSession(model, sp, config)


def normalize_stream_audio(
    sample_rate: int,
    samples: torch.Tensor,
    *,
    target_sr: int = 16_000,
) -> torch.Tensor:
    """Convert a streaming mic chunk to mono float32 at the model sample rate."""
    if samples.dim() > 1:
        samples = samples.mean(dim=-1)
    if sample_rate != target_sr:
        samples = torchaudio.functional.resample(
            samples.unsqueeze(0),
            sample_rate,
            target_sr,
        ).squeeze(0)
    return samples


@dataclass
class StreamingState:
    session: StreamingSession | None = None
    model_name: str | None = None


def _parse_stream_audio(audio: tuple[int, Any] | str) -> torch.Tensor:
    if isinstance(audio, str):
        wav, sample_rate = torchaudio.load(audio)
        if wav.dim() == 2 and wav.size(0) > 1:
            wav = wav.mean(dim=0)
        samples = _pcm_to_float32(wav.squeeze(0))
        return normalize_stream_audio(sample_rate, samples)

    sample_rate, data = audio
    if not isinstance(data, torch.Tensor):
        data = torch.from_numpy(data)
    samples = data.to(torch.float32)
    if samples.numel() == 0:
        return samples
    if samples.dim() > 1:
        samples = samples.mean(dim=-1)
    samples = _pcm_to_float32(samples)
    return normalize_stream_audio(sample_rate, samples)


def run_streaming_step(
    audio: tuple[int, Any] | str | None,
    state: StreamingState | None,
    model_name: str,
) -> tuple[str, StreamingState]:
    """Process one streaming microphone chunk and return the live transcript."""
    current_state = state or StreamingState()

    if audio is None:
        if current_state.session is not None:
            transcript = current_state.session.finish_stream_rescore()
            return transcript, StreamingState()
        return "", StreamingState()

    try:
        if not model_name:
            model_name = DEFAULT_STREAM_MODEL

        if current_state.session is None or current_state.model_name != model_name:
            current_state = StreamingState(
                session=init_streaming_session(model_name),
                model_name=model_name,
            )

        samples = _parse_stream_audio(audio)
        if samples.numel() == 0:
            transcript = (
                current_state.session.get_full_transcript() if current_state.session else ""
            )
            return transcript, current_state

        assert current_state.session is not None
        current_state.session.append_audio(samples)
        current_state.session.drain_pending_steps()
        return current_state.session.get_full_transcript(), current_state
    except Exception as exc:
        logger.exception("Error during streaming transcription")
        if current_state.session is not None:
            current_state.session.reset()
            current_state = StreamingState(model_name=current_state.model_name)
        return f"Error during streaming transcription: {exc}", current_state

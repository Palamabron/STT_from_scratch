from __future__ import annotations

from typing import Literal, cast

import torch
from sentencepiece import SentencePieceProcessor

from SpeechToText.models.common.checkpoint_io import load_lightning_checkpoint
from SpeechToText.models.common.rnnt import transducer_greedy_decode_one

ModelType = Literal["auto", "ctc", "ctc_attention", "tdt"]


def detect_model_type(checkpoint_path: str) -> ModelType:
    """Infer model head from Lightning checkpoint state-dict keys."""
    ckpt = load_lightning_checkpoint(checkpoint_path)
    state = ckpt.get("state_dict", ckpt)
    keys = state.keys() if hasattr(state, "keys") else []

    if any(key.startswith("net.joint.") for key in keys):
        return "tdt"
    if any(key.startswith("net.ctc_proj.") for key in keys) or any(
        key.startswith("net.decoder.") for key in keys
    ):
        return "ctc_attention"
    if any(key.startswith("net.proj.") for key in keys):
        return "ctc"

    raise ValueError(
        f"Could not detect model type from checkpoint: {checkpoint_path}. "
        "Pass --model_type explicitly (ctc, ctc_attention, or tdt)."
    )


def resolve_model_type(checkpoint_path: str, model_type: ModelType) -> ModelType:
    if model_type == "auto":
        return detect_model_type(checkpoint_path)
    return model_type


def load_lit_module(
    checkpoint_path: str,
    *,
    sp: SentencePieceProcessor,
    model_type: ModelType = "auto",
) -> tuple[torch.nn.Module, ModelType]:
    """Load a Lightning module from checkpoint for inference."""
    ckpt = load_lightning_checkpoint(checkpoint_path)
    config = ckpt.get("hyper_parameters", {}).get("config")
    if config is None:
        raise ValueError(f"Checkpoint missing hyper_parameters.config: {checkpoint_path}")

    state_dict = ckpt.get("state_dict")
    if state_dict is None:
        raise ValueError(f"Checkpoint missing state_dict: {checkpoint_path}")

    resolved = resolve_model_type(checkpoint_path, model_type)
    vocab_size = int(sp.get_piece_size()) + 1

    module: torch.nn.Module
    if resolved == "ctc":
        from SpeechToText.models.ctc.lit import LitFastConformerCTC

        module = LitFastConformerCTC(config, sp=sp)
    elif resolved == "ctc_attention":
        from SpeechToText.models.ctc_attention.lit import LitFastConformerCTCAttention

        module = LitFastConformerCTCAttention(config, sp=sp)
    elif resolved == "tdt":
        from SpeechToText.models.tdt.lit import LitFastConformerTDT

        module = LitFastConformerTDT(config, sp=sp, vocab_size=vocab_size)
    else:
        raise ValueError(f"Unsupported model type: {resolved}")

    module.load_state_dict(state_dict, strict=True)
    return module, resolved


@torch.inference_mode()
def forward_ctc_log_probs(
    module: torch.nn.Module,
    audio: torch.Tensor,
    audio_lengths: torch.Tensor,
    model_type: ModelType,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run a forward pass and return CTC log-probabilities with output lengths."""
    if model_type == "ctc":
        from SpeechToText.models.ctc.lit import LitFastConformerCTC

        ctc_lit = cast(LitFastConformerCTC, module)
        feats, feat_lens = ctc_lit.featurizer(audio, audio_lengths)
        out = ctc_lit.net(feats, feat_lens)
        return cast(torch.Tensor, out.log_probs), out.out_lengths

    if model_type == "ctc_attention":
        from SpeechToText.models.ctc_attention.lit import LitFastConformerCTCAttention

        attn_lit = cast(LitFastConformerCTCAttention, module)
        feats, feat_lens = attn_lit.featurizer(audio, audio_lengths)
        out = attn_lit(feats, feat_lens, decoder_input=None)
        return out.ctc_log_probs, out.out_lengths

    raise ValueError(f"Model type '{model_type}' does not expose CTC log-probabilities")


@torch.inference_mode()
def encode_transducer(
    module: torch.nn.Module,
    audio: torch.Tensor,
    audio_lengths: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the transducer encoder and return encoder outputs with lengths."""
    from SpeechToText.models.tdt.lit import LitFastConformerTDT

    lit = cast(LitFastConformerTDT, module)
    feats, feat_lens = lit.featurizer(audio, audio_lengths)
    enc, out_lengths = lit.net.encoder(feats, feat_lens)
    return enc, out_lengths


@torch.inference_mode()
def transducer_greedy_decode_batch(
    module: torch.nn.Module,
    audio: torch.Tensor,
    audio_lengths: torch.Tensor,
    *,
    blank_id: int = 0,
    val_max_symbols_per_t: int = 10,
) -> list[list[int]]:
    """Greedy RNN-T/TDT decode with incremental predictor history."""
    from SpeechToText.models.tdt.lit import LitFastConformerTDT

    lit = cast(LitFastConformerTDT, module)
    enc, out_lengths = encode_transducer(module, audio, audio_lengths)

    decoded: list[list[int]] = []
    for index in range(int(enc.size(0))):
        out_len = int(out_lengths[index].item())
        ids = transducer_greedy_decode_one(
            enc[index : index + 1],
            out_len,
            decoder=lit.net.decoder,
            joint=lit.net.joint,
            blank_id=blank_id,
            max_symbols_per_t=val_max_symbols_per_t,
        )
        decoded.append(ids)
    return decoded


@torch.inference_mode()
def transcribe_batch(
    module: torch.nn.Module,
    audio: torch.Tensor,
    audio_lengths: torch.Tensor,
    *,
    sp: SentencePieceProcessor,
    model_type: ModelType,
    blank_id: int = 0,
    val_max_symbols_per_t: int = 10,
) -> list[str]:
    """Decode a batch of waveforms into text transcripts."""
    from SpeechToText.models.common import ctc_ids_to_texts_spm, greedy_ctc_decode

    if model_type == "tdt":
        decoded = transducer_greedy_decode_batch(
            module,
            audio,
            audio_lengths,
            blank_id=blank_id,
            val_max_symbols_per_t=val_max_symbols_per_t,
        )
        texts: list[str] = []
        for ids in decoded:
            sp_ids = [token_id - 1 for token_id in ids if token_id != blank_id and token_id > 0]
            texts.append("" if not sp_ids else sp.decode_ids(sp_ids))
        return texts

    log_probs, out_lengths = forward_ctc_log_probs(module, audio, audio_lengths, model_type)
    decoded = greedy_ctc_decode(log_probs, out_lengths, blank_id=blank_id)
    return ctc_ids_to_texts_spm(sp, decoded)

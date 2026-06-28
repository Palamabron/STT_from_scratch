from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, cast

import torch
import torch.nn.functional as F
from sentencepiece import SentencePieceProcessor

from SpeechToText.models.common.checkpoint_io import load_lightning_checkpoint
from SpeechToText.models.common.decode_ctc import ctc_ids_to_texts_spm
from SpeechToText.models.common.rnnt import transducer_greedy_decode_one
from SpeechToText.models.ctc_attention.decode import attention_greedy_decode, joint_beam_search

ModelType = Literal["auto", "ctc", "ctc_attention", "tdt"]


@dataclass(frozen=True)
class CtcAttentionSpecialTokens:
    blank_id: int
    bos_id: int
    eos_id: int
    pad_id: int
    max_decode_len: int


def detect_model_type(checkpoint_path: str) -> ModelType:
    """Infer model head from Lightning checkpoint state-dict keys."""
    ckpt = load_lightning_checkpoint(checkpoint_path)
    state = ckpt.get("state_dict", ckpt)
    keys = state.keys() if hasattr(state, "keys") else []

    if any(key.startswith("net.joint.") for key in keys):
        return "tdt"
    if any(
        key.startswith("net.ctc_proj.")
        or key.startswith("net.attention.")
        or key.startswith("net.decoder.")
        for key in keys
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


def ctc_attention_special_tokens(module: torch.nn.Module) -> CtcAttentionSpecialTokens:
    """Token ids aligned with LitFastConformerCTCAttention (blank=0, SPM ids +1)."""
    from SpeechToText.models.ctc_attention.lit import LitFastConformerCTCAttention

    lit = cast(LitFastConformerCTCAttention, module)
    return CtcAttentionSpecialTokens(
        blank_id=int(lit.blank_id),
        bos_id=int(lit.bos_id),
        eos_id=int(lit.eos_id),
        pad_id=int(lit.pad_id),
        max_decode_len=int(lit.net.cfg.decoder.max_len),
    )


@torch.inference_mode()
def encode_ctc_attention(
    module: torch.nn.Module,
    audio: torch.Tensor,
    audio_lengths: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Encode audio and return encoder outputs plus CTC log-probabilities (log_softmax)."""
    from SpeechToText.models.ctc_attention.lit import LitFastConformerCTCAttention

    lit = cast(LitFastConformerCTCAttention, module)
    feats, feat_lens = lit.featurizer(audio, audio_lengths)
    enc, out_lengths, _aux = lit.net.encode(feats, feat_lens)
    ctc_log_probs = F.log_softmax(lit.net.ctc_proj(enc), dim=-1)
    return enc, out_lengths, ctc_log_probs


@torch.inference_mode()
def decode_ctc_attention_attention_greedy(
    module: torch.nn.Module,
    enc: torch.Tensor,
    out_lengths: torch.Tensor,
    sample_index: int,
    *,
    sp: SentencePieceProcessor,
    tokens: CtcAttentionSpecialTokens,
) -> str:
    """Greedy attention decode for one utterance (decoder outputs log_softmax)."""
    from SpeechToText.models.ctc_attention.lit import LitFastConformerCTCAttention

    lit = cast(LitFastConformerCTCAttention, module)
    token_ids = attention_greedy_decode(
        lit.net,
        enc[sample_index : sample_index + 1],
        out_lengths[sample_index : sample_index + 1],
        bos_id=tokens.bos_id,
        eos_id=tokens.eos_id,
        max_len=tokens.max_decode_len,
    )
    return ctc_ids_to_texts_spm(sp, [token_ids])[0]


@torch.inference_mode()
def decode_ctc_attention_joint_beam(
    module: torch.nn.Module,
    enc: torch.Tensor,
    out_lengths: torch.Tensor,
    ctc_log_probs: torch.Tensor,
    sample_index: int,
    *,
    sp: SentencePieceProcessor,
    tokens: CtcAttentionSpecialTokens,
    alpha: float,
    beam_size: int,
    top_k: int | None,
    length_penalty: float,
) -> str:
    """Joint CTC/attention beam search for one utterance."""
    from SpeechToText.models.ctc_attention.lit import LitFastConformerCTCAttention

    lit = cast(LitFastConformerCTCAttention, module)
    token_ids = joint_beam_search(
        lit.net,
        enc[sample_index : sample_index + 1],
        out_lengths[sample_index : sample_index + 1],
        ctc_log_probs[sample_index : sample_index + 1],
        alpha=alpha,
        beam_size=beam_size,
        top_k=top_k,
        bos_id=tokens.bos_id,
        eos_id=tokens.eos_id,
        blank_id=tokens.blank_id,
        max_len=tokens.max_decode_len,
        length_penalty=length_penalty,
    )
    return ctc_ids_to_texts_spm(sp, [token_ids])[0]


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

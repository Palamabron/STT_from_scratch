from __future__ import annotations

from typing import Literal, cast

import torch
import torch.nn.functional as F
from sentencepiece import SentencePieceProcessor

ModelType = Literal["auto", "ctc", "ctc_attention", "tdt"]


def detect_model_type(checkpoint_path: str) -> ModelType:
    """Infer model head from Lightning checkpoint state-dict keys."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
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
    resolved = resolve_model_type(checkpoint_path, model_type)

    module: torch.nn.Module
    if resolved == "ctc":
        from SpeechToText.models.ctc.lit import LitFastConformerCTC

        module = LitFastConformerCTC.load_from_checkpoint(
            checkpoint_path,
            sp=sp,
            weights_only=False,
        )
    elif resolved == "ctc_attention":
        from SpeechToText.models.ctc_attention.lit import LitFastConformerCTCAttention

        module = LitFastConformerCTCAttention.load_from_checkpoint(
            checkpoint_path,
            sp=sp,
            weights_only=False,
        )
    elif resolved == "tdt":
        from SpeechToText.models.tdt.lit import LitFastConformerTDT

        module = LitFastConformerTDT.load_from_checkpoint(
            checkpoint_path,
            sp=sp,
            weights_only=False,
        )
    else:
        raise ValueError(f"Unsupported model type: {resolved}")

    return module, resolved


def module_uses_tdt(module: torch.nn.Module) -> bool:
    """Return True when the loaded module has a duration head (true TDT)."""
    from SpeechToText.models.tdt.lit import LitFastConformerTDT

    if not isinstance(module, LitFastConformerTDT):
        return False
    if bool(getattr(module.config, "use_tdt", False)):
        return True
    return getattr(module.net.joint, "duration_out", None) is not None


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
def forward_tdt_joint(
    module: torch.nn.Module,
    audio: torch.Tensor,
    audio_lengths: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
    """Run RNN-T/TDT encoder + joint network for greedy decoding."""
    from SpeechToText.models.tdt.lit import LitFastConformerTDT

    lit = cast(LitFastConformerTDT, module)
    feats, feat_lens = lit.featurizer(audio, audio_lengths)
    enc, out_lengths = lit.net.encoder(feats, feat_lens)

    batch_size = int(enc.size(0))
    u_max = max(1, int(out_lengths.max().item()))
    dec_in = torch.full(
        (batch_size, u_max + 1),
        lit.blank_id,
        dtype=torch.long,
        device=enc.device,
    )
    dec = lit.net.decoder(dec_in)
    joint_out = lit.net.joint.forward_chunked(
        enc,
        dec,
        fused_batch_size=lit.config.joint_fused_batch_size,
    )

    if isinstance(joint_out, tuple):
        token_logits, duration_logits = joint_out
        return (
            F.log_softmax(token_logits, dim=-1),
            F.log_softmax(duration_logits, dim=-1),
            out_lengths,
        )

    return F.log_softmax(joint_out, dim=-1), None, out_lengths


def forward_tdt_log_probs(
    module: torch.nn.Module,
    audio: torch.Tensor,
    audio_lengths: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Backward-compatible wrapper returning token log-probs only."""
    token_log_probs, _, out_lengths = forward_tdt_joint(module, audio, audio_lengths)
    return token_log_probs, out_lengths


@torch.inference_mode()
def transcribe_batch(
    module: torch.nn.Module,
    audio: torch.Tensor,
    audio_lengths: torch.Tensor,
    *,
    sp: SentencePieceProcessor,
    model_type: ModelType,
    blank_id: int = 0,
    val_max_symbols_per_t: int = 4,
) -> list[str]:
    """Decode a batch of waveforms into text transcripts."""
    from SpeechToText.models.common import ctc_ids_to_texts_spm, greedy_ctc_decode
    from SpeechToText.models.common.rnnt import greedy_rnnt_path_decode_one, greedy_tdt_decode_one

    batch_size = int(audio.size(0))

    if model_type == "tdt":
        token_log_probs, duration_log_probs, out_lengths = forward_tdt_joint(
            module, audio, audio_lengths
        )
        use_tdt = duration_log_probs is not None and module_uses_tdt(module)
        texts: list[str] = []
        for index in range(batch_size):
            out_len = int(out_lengths[index].item())
            if use_tdt and duration_log_probs is not None:
                ids = greedy_tdt_decode_one(
                    token_log_probs[index : index + 1],
                    duration_log_probs[index : index + 1],
                    out_length=out_len,
                    max_symbols_per_t=val_max_symbols_per_t,
                    blank_id=blank_id,
                )
            else:
                ids = greedy_rnnt_path_decode_one(
                    token_log_probs[index : index + 1],
                    out_length=out_len,
                    max_symbols_per_t=val_max_symbols_per_t,
                    blank_id=blank_id,
                )
            sp_ids = [token_id - 1 for token_id in ids if token_id != blank_id and token_id > 0]
            texts.append("" if not sp_ids else sp.decode_ids(sp_ids))
        return texts

    log_probs, out_lengths = forward_ctc_log_probs(module, audio, audio_lengths, model_type)
    decoded = greedy_ctc_decode(log_probs, out_lengths, blank_id=blank_id)
    return ctc_ids_to_texts_spm(sp, decoded)

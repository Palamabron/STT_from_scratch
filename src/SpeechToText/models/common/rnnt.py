from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, cast

import torch
import torch.nn.functional as F
from torchaudio.functional import rnnt_loss

warnings.filterwarnings(
    "ignore",
    message=r".*_rnnt_loss has been deprecated.*",
    category=UserWarning,
)

if TYPE_CHECKING:
    from SpeechToText.models.tdt.decoder import TDTDecoder
    from SpeechToText.models.tdt.joint import JointNet


def targets_1d_to_padded_2d(
    targets_1d: torch.Tensor,
    target_lengths: torch.Tensor,
    pad_value: int,
) -> torch.Tensor:
    device = targets_1d.device
    b = int(target_lengths.shape[0])
    max_u = int(target_lengths.max().item()) if b > 0 else 0

    out = torch.full((b, max_u), pad_value, dtype=torch.long, device=device)
    off = 0
    for i in range(b):
        u = int(target_lengths[i].item())
        if u > 0:
            out[i, :u] = targets_1d[off : off + u]
            off += u
    return out


def _joint_step_log_probs(
    joint: JointNet,
    enc_t: torch.Tensor,
    pred_u: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Return token (and optional duration) log-probs at a single ``(t, u)`` step."""
    enc = enc_t.unsqueeze(0).unsqueeze(1)
    pred = pred_u.unsqueeze(0).unsqueeze(1)
    out = joint.forward(enc, pred)
    if isinstance(out, tuple):
        token_logits, duration_logits = out
        return (
            F.log_softmax(token_logits.squeeze(1).squeeze(1), dim=-1),
            F.log_softmax(duration_logits.squeeze(1).squeeze(1), dim=-1),
        )
    return F.log_softmax(out.squeeze(1).squeeze(1), dim=-1), None


def greedy_rnnt_decode_incremental(
    enc: torch.Tensor,
    out_length: int,
    *,
    decoder: TDTDecoder,
    joint: JointNet,
    blank_id: int,
    max_symbols_per_t: int,
) -> list[int]:
    """Greedy RNN-T decode with predictor state updated from the emitted prefix."""
    if enc.dim() != 3 or enc.size(0) != 1:
        raise ValueError("incremental decode expects encoder output with shape [1, T, D]")

    device = enc.device
    emitted: list[int] = []
    dec_tokens = torch.tensor([[blank_id]], dtype=torch.long, device=device)
    pred = decoder(dec_tokens)

    t = 0
    u = 0
    max_u = out_length + max(1, max_symbols_per_t) * out_length

    while t < out_length and u < max_u:
        n_emit = 0
        while n_emit < max_symbols_per_t and u < max_u and t < out_length:
            token_log_probs, _ = _joint_step_log_probs(joint, enc[0, t, :], pred[0, u, :])
            token_id = int(torch.argmax(token_log_probs).item())
            if token_id == blank_id:
                break

            emitted.append(token_id)
            u += 1
            n_emit += 1
            dec_tokens = torch.cat(
                [dec_tokens, torch.tensor([[token_id]], dtype=torch.long, device=device)],
                dim=1,
            )
            pred = decoder(dec_tokens)
        t += 1

    return emitted


def greedy_tdt_decode_incremental(
    enc: torch.Tensor,
    out_length: int,
    *,
    decoder: TDTDecoder,
    joint: JointNet,
    blank_id: int,
    max_symbols_per_t: int,
) -> list[int]:
    """Greedy TDT decode with frame-skipping and incremental predictor history."""
    if enc.dim() != 3 or enc.size(0) != 1:
        raise ValueError("incremental decode expects encoder output with shape [1, T, D]")

    device = enc.device
    emitted: list[int] = []
    dec_tokens = torch.tensor([[blank_id]], dtype=torch.long, device=device)
    pred = decoder(dec_tokens)

    t = 0
    u = 0
    max_u = out_length + max(1, max_symbols_per_t) * out_length

    while t < out_length and u < max_u:
        n_emit = 0
        while n_emit < max_symbols_per_t and u < max_u and t < out_length:
            token_log_probs, duration_log_probs = _joint_step_log_probs(
                joint, enc[0, t, :], pred[0, u, :]
            )
            assert duration_log_probs is not None

            token_id = int(torch.argmax(token_log_probs).item())
            if token_id == blank_id:
                duration = int(torch.argmax(duration_log_probs).item())
                t += max(1, duration + 1)
                break

            emitted.append(token_id)
            u += 1
            n_emit += 1
            dec_tokens = torch.cat(
                [dec_tokens, torch.tensor([[token_id]], dtype=torch.long, device=device)],
                dim=1,
            )
            pred = decoder(dec_tokens)
            duration = int(torch.argmax(duration_log_probs).item())
            t += max(1, duration + 1)

        if n_emit == 0 and t < out_length:
            t += 1

    return emitted


def transducer_greedy_decode_one(
    enc: torch.Tensor,
    out_length: int,
    *,
    decoder: TDTDecoder,
    joint: JointNet,
    blank_id: int,
    max_symbols_per_t: int,
) -> list[int]:
    """Greedy transducer decode with incremental predictor history."""
    if joint.duration_out is not None:
        return greedy_tdt_decode_incremental(
            enc,
            out_length,
            decoder=decoder,
            joint=joint,
            blank_id=blank_id,
            max_symbols_per_t=max_symbols_per_t,
        )
    return greedy_rnnt_decode_incremental(
        enc,
        out_length,
        decoder=decoder,
        joint=joint,
        blank_id=blank_id,
        max_symbols_per_t=max_symbols_per_t,
    )


def rnnt_loss_mean(
    *,
    logits: torch.Tensor,
    out_lengths: torch.Tensor,
    targets_1d_or_2d: torch.Tensor,
    target_lengths: torch.Tensor,
    blank_id: int,
    clamp: float,
    fused_log_softmax: bool,
) -> torch.Tensor:
    targets_2d = (
        targets_1d_to_padded_2d(targets_1d_or_2d, target_lengths, pad_value=blank_id)
        if targets_1d_or_2d.dim() == 1
        else targets_1d_or_2d
    ).to(dtype=torch.int32)

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r".*_rnnt_loss has been deprecated.*",
            category=UserWarning,
        )
        loss_any = rnnt_loss(
            logits=logits.float(),
            targets=targets_2d,
            logit_lengths=out_lengths.to(dtype=torch.int32),
            target_lengths=target_lengths.to(dtype=torch.int32),
            blank=int(blank_id),
            clamp=float(clamp),
            reduction="mean",
            fused_log_softmax=bool(fused_log_softmax),
        )
    return cast(torch.Tensor, loss_any)

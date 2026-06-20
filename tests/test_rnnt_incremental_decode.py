from __future__ import annotations

import torch
import torch.nn as nn

from SpeechToText.models.common.rnnt import (
    greedy_rnnt_decode_incremental,
    greedy_tdt_decode_incremental,
)


class _PrefixAwareDecoder(nn.Module):
    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        batch_size, length = tokens.shape
        states = torch.zeros(batch_size, length, 2, device=tokens.device)
        for index in range(length):
            states[:, index, 0] = float(index)
            states[:, index, 1] = tokens[:, index].float()
        return states


class _PrefixAwareJoint(nn.Module):
    duration_out = None

    def forward(self, enc: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
        batch_size, time_steps, _ = enc.shape
        _, label_steps, _ = pred.shape
        vocab_size = 3
        logits = torch.full(
            (batch_size, time_steps, label_steps, vocab_size), -10.0, device=enc.device
        )
        logits[..., 0] = 0.0

        for time_index in range(time_steps):
            for label_index in range(label_steps):
                time_value = int(enc[0, time_index, 0].item())
                label_value = int(pred[0, label_index, 0].item())
                prefix_token = int(pred[0, label_index, 1].item())

                if time_value == 0 and label_value == 0:
                    logits[0, time_index, label_index, 1] = 5.0
                    logits[0, time_index, label_index, 0] = -10.0
                elif time_value == 1 and label_value == 1 and prefix_token == 1:
                    logits[0, time_index, label_index, 2] = 5.0
                    logits[0, time_index, label_index, 0] = -10.0
                elif time_value == 1 and label_value == 0:
                    logits[0, time_index, label_index, 1] = 5.0
                    logits[0, time_index, label_index, 0] = -10.0
        return logits


def test_rnnt_incremental_decode_uses_emitted_prefix() -> None:
    decoder = _PrefixAwareDecoder()
    joint = _PrefixAwareJoint()
    enc = torch.zeros(1, 3, 4)
    for time_index in range(3):
        enc[0, time_index, 0] = float(time_index)

    decoded = greedy_rnnt_decode_incremental(
        enc,
        out_length=3,
        decoder=decoder,  # type: ignore[arg-type]
        joint=joint,  # type: ignore[arg-type]
        blank_id=0,
        max_symbols_per_t=2,
    )

    assert decoded == [1, 2]


class _PrefixAwareTdtJoint(nn.Module):
    duration_out = object()

    def forward(
        self, enc: torch.Tensor, pred: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, time_steps, _ = enc.shape
        _, label_steps, _ = pred.shape
        token_vocab = 3
        duration_classes = 3
        token_logits = torch.full(
            (batch_size, time_steps, label_steps, token_vocab), -10.0, device=enc.device
        )
        duration_logits = torch.full(
            (batch_size, time_steps, label_steps, duration_classes), -10.0, device=enc.device
        )
        token_logits[..., 0] = 0.0

        for time_index in range(time_steps):
            for label_index in range(label_steps):
                time_value = int(enc[0, time_index, 0].item())
                label_value = int(pred[0, label_index, 0].item())
                prefix_token = int(pred[0, label_index, 1].item())

                if time_value == 0 and label_value == 0:
                    token_logits[0, time_index, label_index, 1] = 5.0
                    token_logits[0, time_index, label_index, 0] = -10.0
                    duration_logits[0, time_index, label_index, 2] = 5.0
                elif time_value == 3 and label_value == 0:
                    token_logits[0, time_index, label_index, 1] = 5.0
                    token_logits[0, time_index, label_index, 0] = -10.0
                    duration_logits[0, time_index, label_index, 0] = 5.0
                elif time_value == 3 and label_value == 1 and prefix_token == 1:
                    token_logits[0, time_index, label_index, 2] = 5.0
                    token_logits[0, time_index, label_index, 0] = -10.0
                    duration_logits[0, time_index, label_index, 0] = 5.0
        return token_logits, duration_logits


def test_tdt_incremental_decode_uses_emitted_prefix_and_duration() -> None:
    decoder = _PrefixAwareDecoder()
    joint = _PrefixAwareTdtJoint()
    enc = torch.zeros(1, 6, 4)
    enc[0, 0, 0] = 0.0
    enc[0, 3, 0] = 3.0

    decoded = greedy_tdt_decode_incremental(
        enc,
        out_length=6,
        decoder=decoder,  # type: ignore[arg-type]
        joint=joint,  # type: ignore[arg-type]
        blank_id=0,
        max_symbols_per_t=2,
    )

    assert decoded == [1, 2]

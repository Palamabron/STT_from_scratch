import pytest
import torch

from SpeechToText.models.ctc_attention.decode import (
    attention_greedy_decode,
    ctc_prefix_score,
    ctc_prefix_score_bruteforce,
    joint_beam_search,
)


def _random_log_probs(time_steps: int, vocab: int, *, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    logits = torch.randn(time_steps, vocab, generator=generator)
    return torch.log_softmax(logits, dim=-1)


def test_ctc_prefix_score_matches_bruteforce() -> None:
    blank_id = 0
    log_probs = _random_log_probs(10, 5, seed=7)

    for token_ids in [[], [1], [2, 3], [1, 1], [2, 1, 2]]:
        fast = ctc_prefix_score(log_probs, out_len=10, token_ids=token_ids, blank_id=blank_id)
        ref = ctc_prefix_score_bruteforce(
            log_probs, out_len=10, token_ids=token_ids, blank_id=blank_id
        )
        assert fast == pytest.approx(ref)


def test_ctc_prefix_repeat_token_uses_blank_path() -> None:
    blank_id = 0
    log_probs = torch.full((4, 3), 1e-3).log()
    log_probs[:, blank_id] = torch.tensor([0.0, -0.5, -0.5, -0.5])
    log_probs[:, 1] = torch.tensor([-0.5, 0.0, 0.0, 0.0])

    once = ctc_prefix_score(log_probs, out_len=4, token_ids=[1], blank_id=blank_id)
    twice = ctc_prefix_score(log_probs, out_len=4, token_ids=[1, 1], blank_id=blank_id)
    assert twice < once


class _MockAttentionNet:
    def __init__(self, steps: list[dict[int, float]], *, eos_id: int) -> None:
        self.steps = steps
        self.eos_id = eos_id
        self._call = 0

    def decode(
        self,
        enc: torch.Tensor,
        out_lengths: torch.Tensor,
        decoder_input: torch.Tensor,
    ) -> torch.Tensor:
        del enc, out_lengths
        step = self.steps[min(self._call, len(self.steps) - 1)]
        self._call += 1
        vocab = max(step) + 1
        logits = torch.full((1, decoder_input.size(1), vocab), -20.0)
        for token_id, value in step.items():
            logits[0, -1, token_id] = value
        return torch.log_softmax(logits, dim=-1)


def test_attention_greedy_reaches_eos() -> None:
    eos_id = 4
    net = _MockAttentionNet(
        steps=[{1: 0.0, eos_id: -5.0}, {2: 0.0, eos_id: -5.0}, {eos_id: 0.0}],
        eos_id=eos_id,
    )
    enc = torch.zeros(1, 8, 4)
    out_lengths = torch.tensor([8])
    tokens = attention_greedy_decode(
        net,
        enc,
        out_lengths,
        bos_id=3,
        eos_id=eos_id,
        max_len=10,
    )
    assert tokens == [1, 2]


def test_joint_beam_combines_attn_and_ctc() -> None:
    blank_id = 0
    bos_id = 3
    eos_id = 4
    time_steps = 10
    vocab = 5
    log_probs = _random_log_probs(time_steps, vocab, seed=3)
    ctc_log_probs = log_probs.unsqueeze(0)

    net = _MockAttentionNet(
        steps=[
            {1: 0.0, 2: -1.0, eos_id: -10.0},
            {2: 0.0, eos_id: -1.0},
            {eos_id: 0.0},
        ],
        eos_id=eos_id,
    )
    enc = torch.zeros(1, time_steps, 4)
    out_lengths = torch.tensor([time_steps])

    tokens = joint_beam_search(
        net,
        enc,
        out_lengths,
        ctc_log_probs,
        alpha=0.3,
        beam_size=3,
        top_k=2,
        bos_id=bos_id,
        eos_id=eos_id,
        blank_id=blank_id,
        max_len=5,
    )
    assert tokens == [1, 2]

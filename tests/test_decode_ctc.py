import torch

from SpeechToText.models.common.decode_ctc import ctc_ids_to_texts_spm, greedy_ctc_decode


class _FakeSentencePiece:
    def decode_ids(self, ids: list[int]) -> str:
        return "".join(chr(ord("a") + i) for i in ids)


def test_greedy_ctc_decode_collapses_repeats_and_blanks() -> None:
    log_probs = torch.tensor(
        [
            [[0.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]],
        ]
    )
    out_lengths = torch.tensor([4])
    decoded = greedy_ctc_decode(log_probs, out_lengths, blank_id=0)
    assert decoded == [[1, 2]]


def test_ctc_ids_to_texts_spm_shifts_back_to_sentencepiece() -> None:
    sp = _FakeSentencePiece()
    texts = ctc_ids_to_texts_spm(sp, [[1, 3]])
    assert texts == ["ac"]

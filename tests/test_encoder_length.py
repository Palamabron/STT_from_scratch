import pytest
import torch

from SpeechToText.models.conformer.subsampling import subsample_lengths


@pytest.mark.parametrize(
    ("length", "factor", "expected"),
    [
        (100, 2, 50),
        (101, 2, 51),
        (100, 4, 25),
        (100, 8, 12),
    ],
)
def test_subsample_lengths_scalar(length: int, factor: int, expected: int) -> None:
    assert subsample_lengths(length, factor) == expected


def test_subsample_lengths_tensor_matches_scalar() -> None:
    lengths = torch.tensor([100, 101, 64])
    tensor_out = subsample_lengths(lengths, 2)
    assert tensor_out.tolist() == [
        subsample_lengths(100, 2),
        subsample_lengths(101, 2),
        subsample_lengths(64, 2),
    ]


def test_subsample_lengths_rejects_unknown_factor() -> None:
    with pytest.raises(ValueError, match="Unsupported subsampling_factor"):
        subsample_lengths(10, 3)

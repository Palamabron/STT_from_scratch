from SpeechToText.dataset import estimate_encoder_output_length
from SpeechToText.models.conformer.subsampling import subsample_lengths


def test_estimate_encoder_output_length_matches_manual_pipeline() -> None:
    duration_sec = 2.0
    sample_rate = 16_000
    hop_length_ms = 10.0
    subsampling_factor = 4

    hop_length = int(sample_rate * hop_length_ms / 1000.0)
    audio_samples = int(duration_sec * sample_rate)
    feat_len = (audio_samples // hop_length) + 1
    expected = int(subsample_lengths(feat_len, subsampling_factor))

    assert (
        estimate_encoder_output_length(
            duration_sec,
            sample_rate=sample_rate,
            hop_length_ms=hop_length_ms,
            subsampling_factor=subsampling_factor,
        )
        == expected
    )


def test_estimate_encoder_output_length_applies_speed_margin() -> None:
    without_margin = estimate_encoder_output_length(
        2.0,
        sample_rate=16_000,
        hop_length_ms=10.0,
        subsampling_factor=2,
        min_speed_factor=1.0,
    )
    with_margin = estimate_encoder_output_length(
        2.0,
        sample_rate=16_000,
        hop_length_ms=10.0,
        subsampling_factor=2,
        min_speed_factor=0.95,
    )
    assert with_margin <= without_margin

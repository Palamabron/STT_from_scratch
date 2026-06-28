from __future__ import annotations

import torch
import torch.nn as nn
from sentencepiece import SentencePieceProcessor

from SpeechToText.streaming.config import StreamingConfig
from SpeechToText.streaming.session import StreamingSession


# 1. Define robust Mock Modules for testing
class MockFeaturizer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.sample_rate = 16000
        self.hop_length = 160

    def forward(
        self, wav: torch.Tensor, wav_lens: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # wav shape: [1, T_samples]
        # mel shape: [1, T_mel, 80] where T_mel = T_samples // 160 + 1
        t_samples = wav.size(1)
        t_mel = (t_samples // self.hop_length) + 1
        feats = torch.randn(1, t_mel, 80, device=wav.device)
        feat_lens = torch.tensor([t_mel], dtype=torch.long, device=wav.device)
        return feats, feat_lens


class MockEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(
        self, feats: torch.Tensor, feat_lens: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # subsampling by factor of 8
        t_mel = feats.size(1)
        t_enc = max(1, t_mel // 8)
        enc = torch.randn(1, t_enc, 256, device=feats.device)
        out_lengths = torch.tensor([t_enc], dtype=torch.long, device=feats.device)
        return enc, out_lengths


class MockCTCModel(nn.Module):
    def __init__(self, vocab_size: int = 2000) -> None:
        super().__init__()
        self.featurizer = MockFeaturizer()

        class Net(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.encoder = MockEncoder()
                self.ctc_proj = nn.Linear(256, vocab_size)

        self.net = Net()


class MockStandaloneCTCModel(nn.Module):
    """Standalone CTC layout using net.proj instead of net.ctc_proj."""

    def __init__(self, vocab_size: int = 2000) -> None:
        super().__init__()
        self.featurizer = MockFeaturizer()
        self.blank_id = 0

        class Net(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.encoder = MockEncoder()
                self.proj = nn.Linear(256, vocab_size)

        self.net = Net()


class MockDecoderNet(nn.Module):
    def __init__(self, vocab_size: int = 2000) -> None:
        super().__init__()
        # predictor network
        self.embed = nn.Embedding(vocab_size, 256)
        self.lstm = nn.LSTM(256, 256, batch_first=True)

    def forward(self, dec_tokens: torch.Tensor) -> torch.Tensor:
        # dec_tokens: [1, U]
        x = self.embed(dec_tokens)
        out, _ = self.lstm(x)
        return out


class MockJointNet(nn.Module):
    def __init__(self, vocab_size: int = 2000, is_tdt: bool = False) -> None:
        super().__init__()
        self.fc = nn.Linear(512, vocab_size)
        self.duration_out = nn.Linear(512, 4) if is_tdt else None

    def forward(
        self, enc: torch.Tensor, pred: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        # enc: [1, 1, 256], pred: [1, 1, 256]
        cat = torch.cat([enc, pred], dim=-1)
        token_out = self.fc(cat)
        if self.duration_out is not None:
            dur_out = self.duration_out(cat)
            return token_out, dur_out
        return token_out


class MockTransducerModel(nn.Module):
    def __init__(self, vocab_size: int = 2000, is_tdt: bool = False) -> None:
        super().__init__()
        self.blank_id = 0
        self.featurizer = MockFeaturizer()

        class Net(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.encoder = MockEncoder()
                self.decoder = MockDecoderNet(vocab_size)
                self.joint = MockJointNet(vocab_size, is_tdt)

        self.net = Net()


# 2. Test Cases
def test_streaming_standalone_ctc_proj_session() -> None:
    sp = SentencePieceProcessor()
    sp.load("models/spm_unigram_2k_trainval.model")

    model = MockStandaloneCTCModel(vocab_size=len(sp) + 1)
    session = StreamingSession(model, sp, StreamingConfig())
    session.append_audio(torch.randn(16000))
    _ = session.process_step()
    assert isinstance(session.get_full_transcript(), str)


def test_streaming_ctc_session() -> None:
    # Load real SPM
    sp = SentencePieceProcessor()
    sp.load("models/spm_unigram_2k_trainval.model")

    # Instantiate Mock CTC model
    vocab_size = len(sp) + 1  # +1 for CTC blank
    model = MockCTCModel(vocab_size=vocab_size)

    config = StreamingConfig()
    session = StreamingSession(model, sp, config)

    session.reset()

    # Append some audio (e.g. 1.0 second of audio)
    audio = torch.randn(16000)
    session.append_audio(audio)

    # Feed more audio
    _ = session.process_step()

    # Verify that processing step records latency metrics
    metrics = session.get_latency_metrics()
    assert metrics["rtf"] >= 0.0
    assert metrics["mean_latency_ms"] >= 0.0

    # Feed more audio to trigger actual step transcribing
    for _ in range(5):
        # Feed more audio
        _ = session.process_step()

    full_transcript = session.get_full_transcript()
    assert isinstance(full_transcript, str)


def test_streaming_transducer_session() -> None:
    # Load real SPM
    sp = SentencePieceProcessor()
    sp.load("models/spm_unigram_2k_trainval.model")

    vocab_size = len(sp) + 1
    model = MockTransducerModel(vocab_size=vocab_size, is_tdt=False)

    config = StreamingConfig()
    session = StreamingSession(model, sp, config)

    session.reset()

    # 1.0 second of audio
    audio = torch.randn(16000)
    session.append_audio(audio)
    # Feed more audio
    _ = session.process_step()

    # Feed more audio
    _ = session.process_step()
    _ = session.process_step()

    metrics = session.get_latency_metrics()
    assert "rtf" in metrics
    assert metrics["mean_latency_ms"] >= 0.0


def test_streaming_tdt_session() -> None:
    # Load real SPM
    sp = SentencePieceProcessor()
    sp.load("models/spm_unigram_2k_trainval.model")

    vocab_size = len(sp) + 1
    # Create with is_tdt=True to test the Time-Depth Transducer skipping branches
    model = MockTransducerModel(vocab_size=vocab_size, is_tdt=True)

    config = StreamingConfig()
    session = StreamingSession(model, sp, config)

    session.reset()

    audio = torch.randn(16000)
    session.append_audio(audio)
    # Feed more audio
    _ = session.process_step()

    # Feed more audio
    _ = session.process_step()
    _ = session.process_step()

    metrics = session.get_latency_metrics()
    assert metrics["p50_latency_ms"] >= 0.0

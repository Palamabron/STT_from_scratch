from __future__ import annotations

from unittest.mock import MagicMock

import torch

from SpeechToText.demo.transcribe_logic import StreamingState, run_streaming_step


def test_run_streaming_step_resets_on_none_audio() -> None:
    transcript, state = run_streaming_step(
        None, StreamingState(session=MagicMock(), model_name="x"), "x"
    )
    assert transcript == ""
    assert state.session is None
    assert state.model_name is None


def test_run_streaming_step_appends_audio_chunk(monkeypatch) -> None:
    session = MagicMock()
    session.get_full_transcript.return_value = "hello"
    session.process_step.return_value = "lo"

    monkeypatch.setattr(
        "SpeechToText.demo.transcribe_logic.init_streaming_session",
        lambda model_name: session,
    )

    audio = (16_000, torch.zeros(1_600))
    transcript, state = run_streaming_step(audio, StreamingState(), "FastConformer CTC v9")

    session.append_audio.assert_called_once()
    session.process_step.assert_called_once()
    assert transcript == "hello"
    assert state.session is session
    assert state.model_name == "FastConformer CTC v9"

from __future__ import annotations

from unittest.mock import MagicMock

import torch

from SpeechToText.demo.transcribe_logic import StreamingState, run_streaming_step


def test_run_streaming_step_finalizes_on_none_audio() -> None:
    session = MagicMock()
    session.finish_stream_rescore.return_value = "final text"

    transcript, state = run_streaming_step(
        None, StreamingState(session=session, model_name="x"), "x"
    )
    assert transcript == "final text"
    session.finish_stream_rescore.assert_called_once()
    assert state.session is None
    assert state.model_name is None


def test_run_streaming_step_appends_audio_chunk(monkeypatch) -> None:
    session = MagicMock()
    session.get_full_transcript.return_value = "hello"

    monkeypatch.setattr(
        "SpeechToText.demo.transcribe_logic.init_streaming_session",
        lambda model_name: session,
    )

    audio = (16_000, torch.zeros(1_600))
    transcript, state = run_streaming_step(audio, StreamingState(), "FastConformer CTC v9")

    session.append_audio.assert_called_once()
    session.drain_pending_steps.assert_called_once()
    assert transcript == "hello"
    assert state.session is session
    assert state.model_name == "FastConformer CTC v9"

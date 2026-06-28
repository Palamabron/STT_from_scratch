from __future__ import annotations

import argparse

import gradio as gr
import numpy as np
import torch
from loguru import logger

from SpeechToText.demo.analytics_tab import create_analytics_tab
from SpeechToText.demo.transcribe_logic import (
    MODEL_CHECKPOINTS,
    init_streaming_session,
    normalize_stream_audio,
    run_offline_transcribe,
)


def build_app() -> gr.Blocks:
    """Build the full multi-tab Gradio app."""
    with gr.Blocks() as demo:
        gr.HTML(
            """
            <div style="text-align: center; margin-bottom: 20px;">
                <h1 style="color: #2c3e50; font-family: sans-serif;">🎤 Multilingual SpeechToText ASR Demo</h1>
                <p style="font-size: 16px; color: #7f8c8d; font-family: sans-serif;">
                    FastConformer model suite trained on ~800 hours of English and Polish data.
                </p>
            </div>
            """
        )

        with gr.Tabs():
            # 1. Transcribe Tab (Offline)
            with gr.Tab("🎙️ Transcribe Offline"):
                gr.Markdown("### Offline Audio Transcription")
                gr.Markdown(
                    "Upload an audio file (WAV/MP3/FLAC) or record from your microphone. "
                    "Select the acoustic model and decoding algorithm."
                )
                with gr.Row():
                    with gr.Column():
                        audio_input = gr.Audio(type="filepath", label="Input Audio")
                        model_selector = gr.Dropdown(
                            choices=list(MODEL_CHECKPOINTS.keys()),
                            value="FastConformer CTC+Attn v9",
                            label="Select Model",
                        )
                        decode_selector = gr.Dropdown(
                            choices=[
                                "Greedy CTC Decode",
                                "Beam Search + KenLM 5-gram",
                                "Greedy Attention Decode",
                            ],
                            value="Greedy CTC Decode",
                            label="Decoding Algorithm",
                        )
                        transcribe_btn = gr.Button("Transcribe", variant="primary")
                    with gr.Column():
                        text_output = gr.Textbox(
                            label="ASR Hypothesis Transcript",
                            placeholder="Transcript will appear here...",
                        )
                        with gr.Accordion("Evaluation Metrics", open=False):
                            gr.Textbox(label="Reference Text (Optional)")
                            gr.Number(label="WER (%)", precision=2)

                transcribe_btn.click(
                    run_offline_transcribe,
                    inputs=[audio_input, model_selector, decode_selector],
                    outputs=[text_output],
                )

            # 2. Streaming Tab (Real-Time)
            with gr.Tab("⚡ Real-Time Streaming"):
                gr.Markdown("### Real-Time Streaming ASR (Chunk-based)")
                gr.Markdown(
                    "Low-latency streaming mode (chunk size = 320ms, hop = 160ms). "
                    "Speak into the microphone and the system will decode partial hypotheses in real-time."
                )

                # Gradio state: (StreamingSession | None, active model name | None)
                session_state = gr.State(None)

                with gr.Row():
                    with gr.Column():
                        streaming_mic = gr.Audio(
                            sources=["microphone"], streaming=True, label="Microphone (Streaming)"
                        )
                        stream_model = gr.Dropdown(
                            choices=list(MODEL_CHECKPOINTS.keys()),
                            value="FastConformer CTC v9",
                            label="Acoustic Model",
                        )
                        reset_stream_btn = gr.Button("Reset Session", variant="secondary")
                    with gr.Column():
                        streaming_output = gr.Textbox(
                            label="Live Hypothesis Transcript", placeholder="Start speaking..."
                        )
                        latency_box = gr.Textbox(
                            label="Latency & RTF Metrics", value="RTF: - | Latency: -"
                        )
                        rescore_btn = gr.Button(
                            "Finish & Run 2-Pass Attention Rescoring", variant="primary"
                        )

                # Define the streaming audio process function
                def process_audio_stream(audio, state, model_name):
                    if audio is None:
                        return gr.skip(), gr.skip(), state

                    sr, y = audio

                    # Convert to float32 mono tensor
                    if y.dtype == np.int16:
                        y_float = torch.from_numpy(y).float() / 32768.0
                    elif y.dtype == np.int32:
                        y_float = torch.from_numpy(y).float() / 2147483648.0
                    else:
                        y_float = torch.from_numpy(y).float()

                    y_float = normalize_stream_audio(sr, y_float)

                    session, active_model = state if state is not None else (None, None)
                    if session is None or active_model != model_name:
                        session = init_streaming_session(model_name)
                        active_model = model_name

                    # Append audio samples to buffer
                    session.append_audio(y_float)

                    # Process next chunks if available
                    session.process_step()

                    # Get accumulated text and latency metrics
                    full_transcript = session.get_full_transcript()
                    m = session.get_latency_metrics()

                    metrics_str = (
                        f"RTF: {m['rtf']:.3f} | "
                        f"P50 Latency: {m['p50_latency_ms']:.1f}ms | "
                        f"P95 Latency: {m['p95_latency_ms']:.1f}ms"
                    )

                    return full_transcript, metrics_str, (session, active_model)

                # Define reset action
                def reset_stream_session():
                    return "", "RTF: - | Latency: -", None

                # Define attention rescoring action (second pass)
                def run_attention_rescoring(state):
                    if state is None:
                        return "No active streaming session found. Record some audio first."

                    session, _active_model = state
                    logger.info("Running 2nd-pass Attention rescoring on full stream...")
                    rescored_text = session.finish_stream_rescore()
                    return rescored_text

                # Wire components together
                streaming_mic.stream(
                    process_audio_stream,
                    inputs=[streaming_mic, session_state, stream_model],
                    outputs=[streaming_output, latency_box, session_state],
                )

                reset_stream_btn.click(
                    reset_stream_session,
                    inputs=[],
                    outputs=[streaming_output, latency_box, session_state],
                )

                rescore_btn.click(
                    run_attention_rescoring, inputs=[session_state], outputs=[streaming_output]
                )

            # 3. KenLM & LLM Tuning Tab (Structure)
            with gr.Tab("⚙️ Decoder & LLM Tuning"):
                gr.Markdown("### Hybrid Decoding with KenLM & LLM Post-Processing")
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("#### KenLM Hyperparameters")
                        gr.Slider(
                            minimum=0.0, maximum=2.0, value=0.5, step=0.1, label="Alpha (LM Weight)"
                        )
                        gr.Slider(
                            minimum=0.0,
                            maximum=3.0,
                            value=1.5,
                            step=0.1,
                            label="Beta (Word Length Bonus)",
                        )
                        gr.Markdown("#### LLM Post-Processing (Error Correction)")
                        gr.Textbox(
                            value="Correct ASR errors in the following Polish-English bilingual transcription, maintaining the original meaning: {text}",
                            label="LLM Prompt Template",
                            lines=3,
                        )
                    with gr.Column():
                        gr.Markdown("#### Live Evaluation Panel")
                        gr.Textbox(
                            label="Input (raw ASR text)", placeholder="Enter text with errors..."
                        )
                        gr.Button("Run LLM Correction")
                        gr.Textbox(
                            label="Corrected Text", placeholder="Awaiting corrected output..."
                        )

            # 4. Analytics & Benchmark Tab (Fully Implemented)
            with gr.Tab("📈 Analytics & Benchmarks"):
                create_analytics_tab()

    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description="ASR Gradio Demo Application")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host address to bind to")
    parser.add_argument("--port", type=int, default=7860, help="Port to run the app on")
    parser.add_argument("--share", action="store_true", help="Generate public link")
    args = parser.parse_args()

    logger.info("Initializing SpeechToText Gradio application...")
    app = build_app()

    logger.info(f"Launching Gradio server on {args.host}:{args.port}...")
    app.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        theme=gr.themes.Default(primary_hue="blue", secondary_hue="gray"),
    )


if __name__ == "__main__":
    main()

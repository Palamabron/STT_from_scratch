from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import gradio as gr
import tyro
from loguru import logger

from SpeechToText.demo.analytics_tab import create_analytics_tab

# Re-export key streaming logic
from SpeechToText.demo.transcribe_logic import (
    MODEL_CHECKPOINTS,
    run_offline_transcribe,
)
from SpeechToText.demo.transcribe_logic import (
    run_offline_transcribe as run_streaming_step,  # Placeholder
)


def build_app() -> gr.Blocks:
    """Builds the full multi-tab Gradio app."""
    with gr.Blocks(title="ASR Demo") as demo:
        gr.Markdown("# SpeechToText ASR Demo")

        with gr.Tabs():
            # 1. Transcribe Tab (Offline)
            with gr.Tab("📁 File Transcribe"):
                file_input = gr.Audio(type="filepath", label="Upload Audio")
                model_dropdown = gr.Dropdown(
                    choices=list(MODEL_CHECKPOINTS.keys()),
                    value="FastConformer CTC+Attn v9",
                    label="Select Model",
                )
                decode_dropdown = gr.Dropdown(
                    choices=[
                        "Greedy CTC Decode",
                        "Beam Search + KenLM 5-gram",
                        "Greedy Attention Decode",
                    ],
                    value="Greedy CTC Decode",
                    label="Decoding Algorithm",
                )
                transcribe_btn = gr.Button("Transcribe")
                output_text = gr.Textbox(label="Transcript")

                transcribe_btn.click(
                    run_offline_transcribe,
                    inputs=[file_input, model_dropdown, decode_dropdown],
                    outputs=[output_text],
                )

            # 2. Streaming Tab (Real-Time)
            with gr.Tab("🎙️ Streaming"):
                stream_mic = gr.Audio(sources="microphone", streaming=True)
                stream_output = gr.Textbox(label="Live Transcript")
                stream_state = gr.State((None, None))
                stream_mic.stream(
                    run_streaming_step,
                    inputs=[stream_mic, stream_state],
                    outputs=[stream_output, stream_state],
                )

            # 3. Analytics & Benchmark Tab (Fully Implemented)
            with gr.Tab("📈 Analytics & Benchmarks"):
                create_analytics_tab()

    return cast(gr.Blocks, demo)


def main() -> None:
    @dataclass
    class AppArgs:
        host: str = "127.0.0.1"
        port: int = 7860
        share: bool = False

    args = tyro.cli(AppArgs)

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

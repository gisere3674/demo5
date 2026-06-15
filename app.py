from __future__ import annotations

import importlib.util
import tempfile
from pathlib import Path
from typing import Tuple

import gradio as gr

from eraecho_core import DECADES, array_from_path, build_report, transform_audio, wav_bytes_from_array

TITLE = "EraEcho"
DESCRIPTION = "Transpose modern music into the style of any historical decade."


def _try_modal(audio_path: str, decade: str) -> Tuple[str, str] | None:
    if importlib.util.find_spec("modal") is None:
        return None

    import modal

    try:
        fn = modal.Function.from_name("eraecho", "transform_with_era_ai")
        with open(audio_path, "rb") as file:
            audio_bytes = file.read()
        transformed_bytes, report = fn.remote(audio_bytes, Path(audio_path).name, decade)
        output = tempfile.NamedTemporaryFile(delete=False, suffix=f"_{decade}_eraecho.wav")
        output.write(transformed_bytes)
        output.close()
        return output.name, report
    except Exception:
        return None


def transform(audio_path: str | None, decade: str, use_modal: bool) -> Tuple[str | None, str]:
    if not audio_path:
        return None, "Please upload an audio file first."

    if use_modal:
        modal_result = _try_modal(audio_path, decade)
        if modal_result is not None:
            return modal_result

    audio, sample_rate = array_from_path(audio_path)
    processed, analysis, profile = transform_audio(audio, sample_rate, decade)
    output = tempfile.NamedTemporaryFile(delete=False, suffix=f"_{decade}_eraecho.wav")
    output.write(wav_bytes_from_array(processed, sample_rate))
    output.close()
    report = build_report(decade, analysis, profile)
    if use_modal:
        report += "\n\n> Modal was requested but unavailable, so EraEcho used the local fallback DSP pipeline."
    return output.name, report


with gr.Blocks(theme=gr.themes.Soft(), title=TITLE) as demo:
    gr.Markdown(f"# {TITLE}\n{DESCRIPTION}")
    gr.Markdown(
        "Upload a song, choose a decade, and EraEcho returns a historically flavored audio pass plus a fidelity report. "
        "The Space can call Modal for Qwen3-14B notes, then falls back to local DSP when Modal is not configured."
    )
    with gr.Row():
        with gr.Column():
            audio = gr.Audio(label="Upload music", type="filepath")
            decade = gr.Dropdown(DECADES, value="1950s", label="Target decade")
            use_modal = gr.Checkbox(value=True, label="Use Modal AI backend when available")
            button = gr.Button("Transpose through time", variant="primary")
        with gr.Column():
            output_audio = gr.Audio(label="EraEcho transformed audio", type="filepath")
            report = gr.Markdown(label="Historical fidelity report")

    button.click(transform, inputs=[audio, decade, use_modal], outputs=[output_audio, report])


if __name__ == "__main__":
    demo.launch()

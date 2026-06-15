from __future__ import annotations

import tempfile
from pathlib import Path

import modal

MODEL_ID = "Qwen/Qwen3-14B"
APP_NAME = "eraecho"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libsndfile1", "ffmpeg")
    .pip_install(
        "numpy>=1.26.0",
        "scipy>=1.11.0",
        "soundfile>=0.12.1",
        "librosa>=0.10.2.post1",
        "torch>=2.3.0",
        "transformers>=4.51.0",
        "accelerate>=0.33.0",
        "bitsandbytes>=0.43.0",
    )
    .add_local_file("eraecho_core.py", remote_path="/root/eraecho_core.py")
)

app = modal.App(APP_NAME, image=image)


@app.function(gpu="A10G", timeout=900, scaledown_window=300)
def transform_with_era_ai(audio_bytes: bytes, filename: str, decade: str) -> tuple[bytes, str]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from eraecho_core import array_from_path, build_report, transform_audio, wav_bytes_from_array

    suffix = Path(filename or "upload.wav").suffix or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix) as input_file:
        input_file.write(audio_bytes)
        input_file.flush()
        audio, sample_rate = array_from_path(input_file.name)

    processed, analysis, profile = transform_audio(audio, sample_rate, decade)

    prompt = (
        "You are EraEcho, a concise music historian and producer. "
        f"Explain how to reinterpret a {analysis.tempo_bpm:.0f} BPM track with key hint {analysis.key_hint} "
        f"as {decade} music. Mention instrumentation, production limits, and artifacts. "
        "Keep it under 130 words."
    )

    model_notes = ""
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            device_map="auto",
            torch_dtype=torch.float16,
            load_in_4bit=True,
            trust_remote_code=True,
        )
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer([text], return_tensors="pt").to(model.device)
        output = model.generate(**inputs, max_new_tokens=180, temperature=0.7, do_sample=True)
        model_notes = tokenizer.decode(output[0][inputs.input_ids.shape[-1] :], skip_special_tokens=True)
    except Exception as exc:  # Model loading can fail on cold-start capacity; DSP still returns a useful demo.
        model_notes = f"Qwen notes unavailable in this run, so the deterministic era profile was used. Modal/model error: {exc}"

    return wav_bytes_from_array(processed, sample_rate), build_report(decade, analysis, profile, model_notes)

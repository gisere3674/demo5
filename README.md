# EraEcho

EraEcho is a Hugging Face Space prototype that transposes uploaded music into the style of a selected historical decade, from the 1920s through the 2020s.

The app uses:

- **Gradio** for the Hugging Face Space UI.
- **Modal.com** for remote AI/audio processing.
- **Qwen/Qwen3-14B** for era-aware transformation notes and historical fidelity reporting.
- Lightweight DSP filters for the 1-day prototype audio transformation.

All planned model choices stay at **32B parameters or under**.

## Features

- Upload a music file.
- Select a target decade from the 1920s to the 2020s.
- Generate an era-simulated audio file.
- Receive a historical fidelity report explaining instruments, production norms, and audio imperfections.

## Local Space development

```bash
pip install -r requirements.txt
python app.py
```

The app works locally with a deterministic fallback DSP pipeline. If Modal is configured, it will call the remote Modal function.

## Modal setup

1. Install and authenticate Modal:

   ```bash
   pip install modal
   modal setup
   ```

2. Deploy the Modal service:

   ```bash
   modal deploy modal_app.py
   ```

3. In Hugging Face Spaces, add any required Modal credentials/secrets according to Modal's client setup.

## Prototype roadmap

### 1-day prototype

- Qwen3-14B generates text-based musical change descriptions and a historical fidelity report.
- DSP adds decade-specific EQ, saturation, reverb, mono folding, vinyl crackle, tape hiss, or digital polish.

### 3-day polished version

- Add MusicNet-style audio analysis for tempo, key, and instrumentation hints.
- Add a small music generation or symbolic reconstruction model for deeper reinterpretation.
- Improve report scoring with explicit instrumentation counts and references.

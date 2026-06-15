from __future__ import annotations

import io
import math
from dataclasses import dataclass
from typing import Dict, Tuple

import librosa
import numpy as np
import soundfile as sf
from scipy import signal


DECADES = [f"{year}s" for year in range(1920, 2030, 10)]

ERA_PROFILES: Dict[str, Dict[str, object]] = {
    "1920s": {
        "instruments": ["upright piano", "banjo", "muted trumpet", "clarinet", "tuba"],
        "texture": "small acoustic jazz ensemble, mono recording, narrow bandwidth",
        "imperfections": ["shellac noise", "78 RPM crackle", "room bleed"],
        "lowpass": 4200,
        "highpass": 140,
        "mono": True,
        "noise": 0.018,
        "reverb": 0.08,
        "saturation": 1.9,
    },
    "1930s": {
        "instruments": ["swing brass", "walking bass", "brush drums", "clarinet", "stride piano"],
        "texture": "big-band swing arrangement with early microphone coloration",
        "imperfections": ["light disc crackle", "bandstand room tone"],
        "lowpass": 5200,
        "highpass": 110,
        "mono": True,
        "noise": 0.014,
        "reverb": 0.11,
        "saturation": 1.7,
    },
    "1940s": {
        "instruments": ["crooner strings", "horn section", "upright bass", "brush kit", "celeste"],
        "texture": "wartime radio warmth with intimate vocal-band balance",
        "imperfections": ["radio compression", "soft optical distortion"],
        "lowpass": 6100,
        "highpass": 90,
        "mono": True,
        "noise": 0.011,
        "reverb": 0.14,
        "saturation": 1.55,
    },
    "1950s": {
        "instruments": ["clean electric guitar", "upright piano", "trumpet", "saxophone", "slapback drums"],
        "texture": "early rock-and-roll or jazz-club recording with slapback echo",
        "imperfections": ["vinyl surface noise", "tube warmth", "needle lift"],
        "lowpass": 7600,
        "highpass": 70,
        "mono": True,
        "noise": 0.009,
        "reverb": 0.19,
        "saturation": 1.35,
    },
    "1960s": {
        "instruments": ["jangly guitar", "organ", "live drums", "Motown bass", "horn stabs"],
        "texture": "tape-era pop with plate reverb and tight band performance",
        "imperfections": ["tape hiss", "wow/flutter impression"],
        "lowpass": 9000,
        "highpass": 55,
        "mono": False,
        "noise": 0.007,
        "reverb": 0.22,
        "saturation": 1.22,
    },
    "1970s": {
        "instruments": ["Rhodes piano", "analog strings", "funk bass", "dry drums", "wah guitar"],
        "texture": "warm album-track production with analog console color",
        "imperfections": ["tape saturation", "soft hiss"],
        "lowpass": 11000,
        "highpass": 45,
        "mono": False,
        "noise": 0.005,
        "reverb": 0.18,
        "saturation": 1.16,
    },
    "1980s": {
        "instruments": ["FM synth", "gated snare", "chorus guitar", "DX-style bass", "digital pads"],
        "texture": "bright stereo mix with gated ambience and glossy synth layers",
        "imperfections": ["early digital grain", "chorus shimmer"],
        "lowpass": 13500,
        "highpass": 35,
        "mono": False,
        "noise": 0.003,
        "reverb": 0.32,
        "saturation": 1.06,
    },
    "1990s": {
        "instruments": ["sampled drums", "distorted guitar", "rompler piano", "sub bass", "turntable cuts"],
        "texture": "CD-era loudness with sampler-driven edges",
        "imperfections": ["sample grit", "mild digital clipping"],
        "lowpass": 15000,
        "highpass": 30,
        "mono": False,
        "noise": 0.002,
        "reverb": 0.15,
        "saturation": 1.12,
    },
    "2000s": {
        "instruments": ["Auto-Tune sheen", "compressed drums", "supersaw synth", "808 kick", "clean pop guitar"],
        "texture": "hyper-compressed digital pop with clean top-end",
        "imperfections": ["brickwall limiting", "edited precision"],
        "lowpass": 17000,
        "highpass": 25,
        "mono": False,
        "noise": 0.001,
        "reverb": 0.10,
        "saturation": 1.05,
    },
    "2010s": {
        "instruments": ["trap hats", "sidechain synth", "808 bass", "vocal chops", "wide pads"],
        "texture": "streaming-era width, sub bass, and polished transient control",
        "imperfections": ["intentional lo-fi risers", "tight quantization"],
        "lowpass": 18500,
        "highpass": 22,
        "mono": False,
        "noise": 0.0008,
        "reverb": 0.12,
        "saturation": 1.03,
    },
    "2020s": {
        "instruments": ["hybrid synths", "AI vocal textures", "spatial pads", "punchy 808s", "granular ear candy"],
        "texture": "modern streaming master with immersive, genre-fluid production",
        "imperfections": ["creative glitch edits", "ultra-clean limiting"],
        "lowpass": 20000,
        "highpass": 20,
        "mono": False,
        "noise": 0.0003,
        "reverb": 0.09,
        "saturation": 1.0,
    },
}


@dataclass
class AudioAnalysis:
    duration_seconds: float
    tempo_bpm: float
    key_hint: str
    peak: float


def analyze_audio(audio: np.ndarray, sample_rate: int) -> AudioAnalysis:
    mono = librosa.to_mono(audio.T) if audio.ndim == 2 else audio
    duration = float(len(mono) / sample_rate)
    tempo, _ = librosa.beat.beat_track(y=mono, sr=sample_rate)
    chroma = librosa.feature.chroma_cqt(y=mono, sr=sample_rate)
    pitch_classes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    key_hint = pitch_classes[int(np.argmax(np.mean(chroma, axis=1)))] if chroma.size else "Unknown"
    return AudioAnalysis(duration, float(np.atleast_1d(tempo)[0]), key_hint, float(np.max(np.abs(audio)) or 0.0))


def _butter_filter(audio: np.ndarray, sr: int, cutoff: float, kind: str) -> np.ndarray:
    nyquist = sr / 2
    cutoff = min(max(cutoff, 20), nyquist - 100)
    sos = signal.butter(4, cutoff / nyquist, btype=kind, output="sos")
    return signal.sosfiltfilt(sos, audio, axis=0).astype(np.float32)


def _simple_reverb(audio: np.ndarray, sr: int, amount: float) -> np.ndarray:
    if amount <= 0:
        return audio
    delay = max(1, int(sr * 0.055))
    wet = np.zeros_like(audio)
    wet[delay:] += audio[:-delay] * 0.55
    if delay * 2 < len(audio):
        wet[delay * 2 :] += audio[: -delay * 2] * 0.28
    return (audio * (1 - amount) + wet * amount).astype(np.float32)


def _add_noise(audio: np.ndarray, amount: float, seed: int = 7) -> np.ndarray:
    if amount <= 0:
        return audio
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, amount, size=audio.shape).astype(np.float32)
    crackle_mask = rng.random(audio.shape) > 0.9975
    crackle = rng.normal(0, amount * 7, size=audio.shape).astype(np.float32) * crackle_mask
    return (audio + noise + crackle).astype(np.float32)


def transform_audio(audio: np.ndarray, sample_rate: int, decade: str) -> Tuple[np.ndarray, AudioAnalysis, Dict[str, object]]:
    if decade not in ERA_PROFILES:
        raise ValueError(f"Unsupported decade: {decade}")

    profile = ERA_PROFILES[decade]
    analysis = analyze_audio(audio, sample_rate)

    processed = audio.astype(np.float32)
    if processed.ndim == 1:
        processed = processed[:, None]

    processed = _butter_filter(processed, sample_rate, float(profile["highpass"]), "highpass")
    processed = _butter_filter(processed, sample_rate, float(profile["lowpass"]), "lowpass")

    saturation = float(profile["saturation"])
    processed = np.tanh(processed * saturation) / max(math.tanh(saturation), 1e-6)
    processed = _simple_reverb(processed, sample_rate, float(profile["reverb"]))
    processed = _add_noise(processed, float(profile["noise"]))

    if bool(profile["mono"]):
        mono = np.mean(processed, axis=1, keepdims=True)
        processed = np.repeat(mono, 2, axis=1)

    peak = np.max(np.abs(processed)) or 1.0
    processed = (processed / peak * 0.92).astype(np.float32)
    return processed, analysis, profile


def build_report(decade: str, analysis: AudioAnalysis, profile: Dict[str, object], model_notes: str | None = None) -> str:
    instruments = list(profile["instruments"])
    imperfections = list(profile["imperfections"])
    score = min(100, 55 + len(instruments) * 6 + len(imperfections) * 5)
    lines = [
        f"# Historical fidelity report: {decade}",
        "",
        f"**Input analysis:** about {analysis.duration_seconds:.1f}s, {analysis.tempo_bpm:.0f} BPM, key center hint: {analysis.key_hint}.",
        f"**Era texture:** {profile['texture']}.",
        f"**Era instruments referenced ({len(instruments)}):** {', '.join(instruments)}.",
        f"**Era imperfections added ({len(imperfections)}):** {', '.join(imperfections)}.",
        f"**Fidelity score:** {score}/100 for this prototype DSP pass.",
    ]
    if model_notes:
        lines.extend(["", "## Qwen era-arrangement notes", model_notes.strip()])
    else:
        lines.extend([
            "",
            "## Prototype notes",
            "This quick version preserves the uploaded performance while applying decade-specific bandwidth, mono/stereo width, saturation, reverb, and noise. A polished version would regenerate the arrangement with dedicated music models.",
        ])
    return "\n".join(lines)


def wav_bytes_from_array(audio: np.ndarray, sample_rate: int) -> bytes:
    buffer = io.BytesIO()
    sf.write(buffer, audio, sample_rate, format="WAV")
    return buffer.getvalue()


def array_from_path(path: str, target_sr: int = 44100) -> Tuple[np.ndarray, int]:
    audio, sr = librosa.load(path, sr=target_sr, mono=False)
    if audio.ndim == 1:
        audio = audio[:, None]
    else:
        audio = audio.T
    return audio.astype(np.float32), sr

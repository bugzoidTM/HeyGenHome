"""Voz em português brasileiro com Kokoro-82M (independente do motor do avatar)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

SAMPLE_RATE = 24000
LANG_CODE = "p"  # português brasileiro

VOICES = {
    "Feminina — Dora (PT-BR)": "pf_dora",
    "Masculina — Alex (PT-BR)": "pm_alex",
    "Masculina — Santa (PT-BR)": "pm_santa",
}
DEFAULT_VOICE = "Feminina — Dora (PT-BR)"


class TTSError(RuntimeError):
    pass


_pipeline = None


def get_pipeline():
    global _pipeline
    if _pipeline is None:
        from kokoro import KPipeline  # import pesado (PyTorch): só na primeira voz

        _pipeline = KPipeline(lang_code=LANG_CODE)
    return _pipeline


def synthesize(text: str, voice_label: str, speed: float, out_path: str | Path) -> float:
    """Gera o WAV em ``out_path`` e devolve a duração em segundos."""
    if not text or not text.strip():
        raise TTSError("Digite um texto.")
    if voice_label not in VOICES:
        raise TTSError(f"Voz desconhecida: {voice_label}")

    chunks = [
        np.asarray(audio, dtype=np.float32)
        for _graphemes, _phonemes, audio in get_pipeline()(
            text.strip(), voice=VOICES[voice_label], speed=float(speed)
        )
    ]
    if not chunks:
        raise TTSError("O TTS não retornou áudio.")

    audio = np.concatenate(chunks)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(out_path, audio, SAMPLE_RATE)
    return len(audio) / SAMPLE_RATE

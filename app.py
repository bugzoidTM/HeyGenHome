import os
import sys
import time
import subprocess
from pathlib import Path

import gradio as gr
import numpy as np
import soundfile as sf
from kokoro import KPipeline

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# Configure these through environment variables or edit the defaults.
SADTALKER_DIR = Path(os.getenv("SADTALKER_DIR", r"C:\AI\SadTalker"))
SADTALKER_PYTHON = os.getenv(
    "SADTALKER_PYTHON",
    r"C:\AI\SadTalker\venv\Scripts\python.exe"
)

VOICES = {
    "Feminina — Dora (PT-BR)": "pf_dora",
    "Masculina — Alex (PT-BR)": "pm_alex",
    "Masculina — Santa (PT-BR)": "pm_santa",
}

_pipeline = None


def get_pipeline():
    global _pipeline
    if _pipeline is None:
        _pipeline = KPipeline(lang_code="p")
    return _pipeline


def synthesize(text: str, voice_label: str, speed: float) -> str:
    if not text or not text.strip():
        raise gr.Error("Digite um texto.")

    voice = VOICES[voice_label]
    pipeline = get_pipeline()

    chunks = []
    for _graphemes, _phonemes, audio in pipeline(
        text.strip(),
        voice=voice,
        speed=float(speed),
    ):
        chunks.append(np.asarray(audio, dtype=np.float32))

    if not chunks:
        raise gr.Error("O TTS não retornou áudio.")

    audio = np.concatenate(chunks)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    wav_path = OUTPUT_DIR / f"voz-{stamp}.wav"
    sf.write(wav_path, audio, 24000)
    return str(wav_path)


def newest_mp4(folder: Path):
    files = list(folder.rglob("*.mp4"))
    if not files:
        return None
    return max(files, key=lambda p: p.stat().st_mtime)


def animate(image_path: str, wav_path: str, mode: str) -> str:
    if not image_path:
        raise gr.Error("Envie uma imagem do avatar.")

    inference = SADTALKER_DIR / "inference.py"
    python_exe = Path(SADTALKER_PYTHON)

    if not inference.exists():
        raise gr.Error(
            f"SadTalker não encontrado em: {inference}. "
            "Configure SADTALKER_DIR."
        )
    if not python_exe.exists():
        raise gr.Error(
            f"Python do SadTalker não encontrado em: {python_exe}. "
            "Configure SADTALKER_PYTHON."
        )

    stamp = time.strftime("%Y%m%d-%H%M%S")
    job_dir = OUTPUT_DIR / f"video-{stamp}"
    job_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(python_exe),
        str(inference),
        "--driven_audio", str(wav_path),
        "--source_image", str(image_path),
        "--result_dir", str(job_dir),
        "--cpu",
    ]

    # "Apresentador" preserva mais a foto original.
    if mode == "Apresentador (mais estável)":
        cmd += ["--still", "--preprocess", "full"]
    else:
        cmd += ["--preprocess", "crop"]

    proc = subprocess.run(
        cmd,
        cwd=str(SADTALKER_DIR),
        capture_output=True,
        text=True,
    )

    if proc.returncode != 0:
        error = (proc.stderr or proc.stdout or "Erro desconhecido")[-5000:]
        raise gr.Error("SadTalker falhou:\n" + error)

    mp4 = newest_mp4(job_dir)
    if not mp4:
        raise gr.Error("O SadTalker terminou, mas nenhum MP4 foi encontrado.")

    return str(mp4)


def generate(image_path, text, voice_label, speed, mode):
    wav = synthesize(text, voice_label, speed)
    mp4 = animate(image_path, wav, mode)
    return wav, mp4


with gr.Blocks(title="Avatar Falante Local — CPU") as demo:
    gr.Markdown(
        """
# Avatar Falante Local
**Foto + texto → voz PT-BR → vídeo MP4**, rodando localmente e sem API paga.

Primeiro teste com textos curtos (1–3 frases), pois a animação em CPU é a etapa mais lenta.
"""
    )

    with gr.Row():
        image = gr.Image(
            label="Foto do avatar",
            type="filepath",
        )
        with gr.Column():
            text = gr.Textbox(
                label="Texto",
                lines=8,
                placeholder="Digite o que o avatar deve falar..."
            )
            voice = gr.Dropdown(
                list(VOICES.keys()),
                value="Feminina — Dora (PT-BR)",
                label="Voz"
            )
            speed = gr.Slider(
                minimum=0.8,
                maximum=1.25,
                value=1.0,
                step=0.05,
                label="Velocidade da voz"
            )
            mode = gr.Radio(
                ["Apresentador (mais estável)", "Rosto (mais movimento)"],
                value="Apresentador (mais estável)",
                label="Modo"
            )
            btn = gr.Button("Gerar vídeo", variant="primary")

    audio_out = gr.Audio(label="Áudio gerado")
    video_out = gr.Video(label="Vídeo final")

    btn.click(
        fn=generate,
        inputs=[image, text, voice, speed, mode],
        outputs=[audio_out, video_out],
    )

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch(
        server_name="127.0.0.1",
        server_port=7860,
        inbrowser=True,
    )

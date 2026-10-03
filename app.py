import inspect
import secrets
import time
from pathlib import Path

import gradio as gr

from heygenhome.engines import (
    FRAMING_CROP,
    FRAMING_FULL,
    AnimationOptions,
    EngineError,
    EngineRun,
    TalkingHeadBackend,
    create_engine,
)
from heygenhome.export import (
    BACKGROUND_BLACK,
    BACKGROUND_BLUR,
    FORMAT_ORIGINAL,
    FORMAT_VERTICAL,
    ExportResult,
    export_video,
)
from heygenhome.media import MediaError
from heygenhome.tts import DEFAULT_VOICE, VOICES, TTSError, synthesize

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# Motores configurados por variáveis de ambiente (veja o README).
ENGINE = create_engine()
DEFAULT_ENGINE = ENGINE.backends()[0].key

FRAMING_CHOICES = [
    ("Foto inteira (padrão)", FRAMING_FULL),
    ("Só o rosto (recorte)", FRAMING_CROP),
]
FORMAT_CHOICES = [
    ("Original", FORMAT_ORIGINAL),
    ("9:16 vertical (sem corte)", FORMAT_VERTICAL),
]
BACKGROUND_CHOICES = [
    ("Desfocado", BACKGROUND_BLUR),
    ("Preto", BACKGROUND_BLACK),
]
FORMAT_LABELS = {value: label for label, value in FORMAT_CHOICES}
FRAMING_SHORT = {FRAMING_FULL: "foto inteira", FRAMING_CROP: "só o rosto"}

# O player e a prévia da foto nunca cortam: mostram o quadro inteiro (contain).
CSS = """
#avatar-video video, #avatar-video img { object-fit: contain !important; background: #000; }
#avatar-photo img { object-fit: contain !important; }
"""


def size_choices(backend: TalkingHeadBackend):
    return [(f"{s} px" + (" (padrão)" if s == 512 else ""), s) for s in backend.capabilities.sizes]


def engine_markdown(backend: TalkingHeadBackend) -> str:
    status = backend.status()
    head = f"**{backend.label}** — {backend.name}. {backend.description}"
    if not status.ready:
        lines = [head, "", "❌ **Não instalado:**"] + [f"- {p}" for p in status.problems]
    else:
        lines = [head, "", "✅ Pronto."] + [f"- ⚠️ {w}" for w in status.warnings]
    return "\n".join(lines)


def on_engine_change(key: str):
    backend = ENGINE.get(key)
    caps = backend.capabilities
    size = 512 if 512 in caps.sizes else caps.sizes[0]
    return (
        engine_markdown(backend),
        gr.update(interactive=caps.still),
        gr.update(interactive=caps.enhancer),
        gr.update(choices=size_choices(backend), value=size),
    )


def new_job_dir(engine_key: str) -> Path:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    job_dir = OUTPUT_DIR / f"{stamp}-{engine_key}-{secrets.token_hex(2)}"
    job_dir.mkdir(parents=True)
    return job_dir


def fmt_seconds(seconds: float) -> str:
    minutes, sec = divmod(int(round(seconds)), 60)
    return f"{minutes}m{sec:02d}s" if minutes else f"{sec}s"


def build_report(
    run: EngineRun, export: ExportResult, audio_duration: float, timings: dict[str, float]
) -> str:
    req, img, raw, final = run.request, run.image, run.video, export.info
    options = [
        f"{req.size} px",
        FRAMING_SHORT[req.framing],
        "cabeça estável" if req.still else "cabeça livre",
        "com melhoria de rosto (GFPGAN)" if req.enhancer else "sem melhoria de rosto",
    ]
    photo = f"{img.size[0]}x{img.size[1]}"
    if img.size != img.original_size:
        photo += f" (enviada {img.original_size[0]}x{img.original_size[1]})"

    lines = [
        "### Relatório",
        "",
        "| Etapa | Resultado |",
        "|---|---|",
        f"| Motor | {run.backend.label} ({run.backend.name}) |",
        f"| Ajustes | {' · '.join(options)} |",
        f"| Foto | {photo} |",
        f"| Voz | {audio_duration:.2f}s |",
        f"| Saída do motor | {raw.resolution} · {raw.codec} · {raw.fps:g} fps · {raw.duration:.2f}s |",
        f"| MP4 final | **{final.resolution}** · {final.codec}/{final.audio_codec} · "
        f"{FORMAT_LABELS[export.fmt]} · {final.duration:.2f}s · ✅ resolução validada |",
        f"| Tempo | voz {fmt_seconds(timings['tts'])} · animação {fmt_seconds(timings['engine'])} · "
        f"exportação {fmt_seconds(timings['export'])} |",
    ]
    if run.notes:
        lines += ["", "**Observações**"] + [f"- {n}" for n in run.notes]
    warnings = run.warnings + export.warnings
    if warnings:
        lines += ["", "**Avisos**"] + [f"- ⚠️ {w}" for w in warnings]
    lines += ["", f"Arquivos em `{export.path.parent}`"]
    return "\n".join(lines)


def generate(
    photo, text, voice, speed, engine_key, framing, still, size, enhancer, fmt, background,
    progress=gr.Progress(),
):
    if not photo:
        raise gr.Error("Envie uma foto do avatar.")
    backend = ENGINE.get(engine_key)
    job_dir = new_job_dir(engine_key)
    timings = {}

    try:
        progress(0.05, desc="Gerando voz (Kokoro)…")
        start = time.monotonic()
        wav = job_dir / "voz.wav"
        audio_duration = synthesize(text, voice, speed, wav)
        timings["tts"] = time.monotonic() - start

        progress(0.2, desc=f"Animando o avatar ({backend.label} — {backend.name})…")
        options = AnimationOptions(
            size=int(size), framing=framing, still=bool(still), enhancer=bool(enhancer)
        )
        run = ENGINE.animate(engine_key, photo, wav, job_dir, options, audio_duration=audio_duration)
        timings["engine"] = run.seconds

        progress(0.9, desc="Exportando e validando o MP4…")
        start = time.monotonic()
        export = export_video(
            run.video.path,
            job_dir / f"avatar-{fmt}.mp4",
            fmt=fmt,
            audio=wav,
            # Foto inteira: o Original tem a resolução da foto. Recorte: a do vídeo do motor.
            original_size=run.image.size if run.request.framing == FRAMING_FULL else None,
            background=background,
            audio_duration=audio_duration,
        )
        timings["export"] = time.monotonic() - start
    except (TTSError, EngineError, MediaError) as exc:
        raise gr.Error(str(exc), duration=None) from exc

    report = build_report(run, export, audio_duration, timings)
    (job_dir / "relatorio.md").write_text(report, encoding="utf-8")
    return str(wav), str(export.path), report


default_backend = ENGINE.get(DEFAULT_ENGINE)
blocks_kwargs = {"title": "Avatar Falante Local — CPU"}
launch_kwargs = {}
# Gradio 5 recebe o CSS em Blocks(); o Gradio 6 em launch().
if "css" in inspect.signature(gr.Blocks.__init__).parameters:
    blocks_kwargs["css"] = CSS
else:
    launch_kwargs["css"] = CSS

with gr.Blocks(**blocks_kwargs) as demo:
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
            image_mode=None,  # entrega o arquivo original; o motor normaliza sem cortar
            elem_id="avatar-photo",
            height=420,
        )
        with gr.Column():
            text = gr.Textbox(
                label="Texto",
                lines=8,
                placeholder="Digite o que o avatar deve falar...",
            )
            voice = gr.Dropdown(list(VOICES.keys()), value=DEFAULT_VOICE, label="Voz")
            speed = gr.Slider(
                minimum=0.8, maximum=1.25, value=1.0, step=0.05, label="Velocidade da voz"
            )

    with gr.Row():
        with gr.Column():
            engine = gr.Radio(
                [(b.label, b.key) for b in ENGINE.backends()],
                value=DEFAULT_ENGINE,
                label="MOTOR DO AVATAR",
            )
            engine_info = gr.Markdown(engine_markdown(default_backend))
        with gr.Column():
            framing = gr.Radio(
                FRAMING_CHOICES,
                value=FRAMING_FULL,
                label="Enquadramento",
                info="Foto inteira preserva a resolução e a proporção da foto.",
            )
            fmt = gr.Radio(
                FORMAT_CHOICES,
                value=FORMAT_ORIGINAL,
                label="Formato do vídeo",
                info="O 9:16 encaixa o vídeo inteiro em 1080x1920, sem cortar.",
            )
            with gr.Row():
                still = gr.Checkbox(value=True, label="Cabeça estável (still)")
                enhancer = gr.Checkbox(value=False, label="Melhorar rosto (GFPGAN, mais lento)")
            with gr.Accordion("Mais ajustes", open=False):
                size = gr.Radio(
                    size_choices(default_backend), value=512, label="Resolução do rosto"
                )
                background = gr.Radio(
                    BACKGROUND_CHOICES, value=BACKGROUND_BLUR, label="Fundo das sobras no 9:16"
                )

    btn = gr.Button("Gerar vídeo", variant="primary")

    with gr.Row():
        with gr.Column(scale=1):
            audio_out = gr.Audio(label="Áudio gerado")
            report_out = gr.Markdown()
        with gr.Column(scale=1):
            video_out = gr.Video(label="Vídeo final", elem_id="avatar-video", height=560)

    # No carregamento da página também: o status reflete instalações feitas depois de abrir o app.
    for trigger in (engine.change, demo.load):
        trigger(
            fn=on_engine_change,
            inputs=engine,
            outputs=[engine_info, still, enhancer, size],
        )
    btn.click(
        fn=generate,
        inputs=[image, text, voice, speed, engine, framing, still, size, enhancer, fmt, background],
        outputs=[audio_out, video_out, report_out],
    )

if __name__ == "__main__":
    for backend in ENGINE.backends():
        status = backend.status()
        mark = "OK" if status.ready else "NÃO INSTALADO"
        print(f"[motor] {backend.label} ({backend.name}): {mark}")
        for line in status.problems + status.warnings:
            print(f"        - {line}")

    demo.queue(default_concurrency_limit=1).launch(
        server_name="127.0.0.1",
        server_port=7860,
        inbrowser=True,
        allowed_paths=[str(OUTPUT_DIR)],
        **launch_kwargs,
    )

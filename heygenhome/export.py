"""Exportação final do vídeo, independente do motor de animação.

Sempre reencoda para H.264 (yuv420p) + AAC com +faststart, que toca no
navegador, no celular e nas redes sociais. Nenhum formato corta o avatar:
quando a proporção do destino é diferente, o vídeo é encaixado (contain) e o
espaço que sobra é preenchido.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .media import MediaError, VideoInfo, even, probe_video, run_ffmpeg

FORMAT_ORIGINAL = "original"
FORMAT_VERTICAL = "9x16"
FORMATS = (FORMAT_ORIGINAL, FORMAT_VERTICAL)

BACKGROUND_BLUR = "blur"
BACKGROUND_BLACK = "black"

VERTICAL_SIZE = (1080, 1920)

# Diferença aceitável entre a duração do vídeo e a do áudio (segundos).
DURATION_TOLERANCE = 0.5


class ExportError(MediaError):
    """O MP4 exportado não ficou como esperado."""


@dataclass(frozen=True)
class ExportResult:
    path: Path
    info: VideoInfo
    fmt: str
    target: tuple[int, int]
    warnings: list[str] = field(default_factory=list)


def export_target(
    fmt: str, video: VideoInfo, original_size: tuple[int, int] | None = None
) -> tuple[int, int]:
    """Resolução final de cada formato.

    - ``original``: a resolução da foto (quando conhecida) ou a do vídeo do motor;
    - ``9x16``: 1080x1920.
    """
    if fmt == FORMAT_VERTICAL:
        return VERTICAL_SIZE
    if fmt == FORMAT_ORIGINAL:
        w, h = original_size or video.size
        return even(w), even(h)
    raise ValueError(f"Formato desconhecido: {fmt}")


def video_filter(target: tuple[int, int], background: str = BACKGROUND_BLACK) -> str:
    """Filtro que encaixa o vídeo inteiro em ``target`` sem cortar nem distorcer."""
    tw, th = target
    fit = f"scale={tw}:{th}:force_original_aspect_ratio=decrease:flags=lanczos,setsar=1"
    if background == BACKGROUND_BLUR:
        # O fundo é uma cópia ampliada e desfocada; o avatar fica inteiro por cima.
        return (
            "[0:v]split=2[bg][fg];"
            f"[bg]scale={tw}:{th}:force_original_aspect_ratio=increase,"
            f"crop={tw}:{th},boxblur=40:2,setsar=1[bgb];"
            f"[fg]{fit}[fgs];"
            "[bgb][fgs]overlay=(W-w)/2:(H-h)/2,format=yuv420p[v]"
        )
    return f"[0:v]{fit},pad={tw}:{th}:(ow-iw)/2:(oh-ih)/2:color=black,format=yuv420p[v]"


def validate_export(
    info: VideoInfo,
    target: tuple[int, int],
    *,
    expect_audio: bool = True,
    audio_duration: float | None = None,
) -> list[str]:
    """Confere o MP4 final. Erros graves levantam ExportError; o resto vira aviso."""
    errors = []
    if info.size != target:
        errors.append(f"resolução {info.resolution}, esperado {target[0]}x{target[1]}")
    if info.codec != "h264":
        errors.append(f"codec {info.codec}, esperado h264")
    if info.pix_fmt != "yuv420p":
        errors.append(f"pixel format {info.pix_fmt}, esperado yuv420p")
    if expect_audio and not info.has_audio:
        errors.append("sem faixa de áudio")
    if errors:
        raise ExportError(f"MP4 final inválido ({info.path.name}): " + "; ".join(errors) + ".")

    warnings = []
    if audio_duration and abs(info.duration - audio_duration) > DURATION_TOLERANCE:
        warnings.append(
            f"Duração do vídeo ({info.duration:.2f}s) difere do áudio ({audio_duration:.2f}s)."
        )
    return warnings


def export_video(
    video: str | Path,
    out_path: str | Path,
    *,
    fmt: str = FORMAT_ORIGINAL,
    audio: str | Path | None = None,
    original_size: tuple[int, int] | None = None,
    background: str = BACKGROUND_BLUR,
    audio_duration: float | None = None,
) -> ExportResult:
    """Gera o MP4 final no formato pedido e valida a resolução.

    ``audio`` deve ser o WAV original do TTS: alguns motores (ex.: SadTalker)
    reamostram o áudio para 16 kHz, então a versão original soa melhor.
    """
    source = probe_video(video)
    target = export_target(fmt, source, original_size)
    # No formato Original não há sobra para preencher; o fundo só importa no 9:16.
    bg = background if fmt == FORMAT_VERTICAL else BACKGROUND_BLACK

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    args = ["-i", str(video)]
    if audio:
        args += ["-i", str(audio)]
    args += ["-filter_complex", video_filter(target, bg), "-map", "[v]"]
    if audio:
        args += ["-map", "1:a:0"]
    elif source.has_audio:
        args += ["-map", "0:a:0"]
    args += [
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
        "-movflags", "+faststart", "-shortest",
        str(out_path),
    ]
    run_ffmpeg(args)

    info = probe_video(out_path)
    warnings = validate_export(
        info,
        target,
        expect_audio=bool(audio) or source.has_audio,
        audio_duration=audio_duration,
    )
    return ExportResult(path=out_path, info=info, fmt=fmt, target=target, warnings=warnings)

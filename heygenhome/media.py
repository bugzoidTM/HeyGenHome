"""Imagem e vídeo: preparo da foto, leitura de MP4 (ffprobe) e composição (FFmpeg).

Nada aqui depende do motor de animação; os motores e a exportação usam estas
funções para garantir que a resolução e a proporção da foto sejam preservadas.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageOps


class MediaError(RuntimeError):
    """Falha ao ler, preparar ou gerar imagem/vídeo."""


def find_tool(name: str) -> str:
    path = shutil.which(name)
    if not path:
        raise MediaError(
            f"'{name}' não foi encontrado no PATH. Instale o FFmpeg "
            "(ele inclui o ffprobe) e reinicie o terminal."
        )
    return path


def run_ffmpeg(args: list[str]) -> None:
    cmd = [find_tool("ffmpeg"), "-hide_banner", "-loglevel", "error", "-y", *args]
    proc = subprocess.run(
        cmd, capture_output=True, text=True, encoding="utf-8", errors="replace"
    )
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "erro desconhecido").strip()
        raise MediaError("FFmpeg falhou:\n" + detail[-3000:])


def even(value: int) -> int:
    """Menor número par >= value (H.264 em yuv420p exige dimensões pares)."""
    return value + (value % 2)


@dataclass(frozen=True)
class VideoInfo:
    path: Path
    width: int
    height: int
    codec: str
    pix_fmt: str
    fps: float
    duration: float
    audio_codec: str | None

    @property
    def size(self) -> tuple[int, int]:
        return self.width, self.height

    @property
    def resolution(self) -> str:
        return f"{self.width}x{self.height}"

    @property
    def has_audio(self) -> bool:
        return self.audio_codec is not None


def _parse_rate(rate: str | None) -> float:
    if not rate or rate == "0/0":
        return 0.0
    if "/" in rate:
        num, den = rate.split("/", 1)
        return float(num) / float(den) if float(den) else 0.0
    return float(rate)


def probe_video(path: str | Path) -> VideoInfo:
    """Lê resolução, codec, fps, duração e áudio de um vídeo com ffprobe."""
    path = Path(path)
    if not path.is_file():
        raise MediaError(f"Vídeo não encontrado: {path}")

    proc = subprocess.run(
        [
            find_tool("ffprobe"), "-v", "error", "-print_format", "json",
            "-show_streams", "-show_format", str(path),
        ],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    if proc.returncode != 0:
        raise MediaError(f"ffprobe não conseguiu ler {path.name}:\n{proc.stderr[-2000:]}")

    data = json.loads(proc.stdout or "{}")
    streams = data.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
    if video is None or not video.get("width") or not video.get("height"):
        raise MediaError(f"{path.name} não contém uma faixa de vídeo válida.")

    duration = data.get("format", {}).get("duration") or video.get("duration") or 0
    return VideoInfo(
        path=path,
        width=int(video["width"]),
        height=int(video["height"]),
        codec=video.get("codec_name", "?"),
        pix_fmt=video.get("pix_fmt", "?"),
        fps=_parse_rate(video.get("avg_frame_rate")) or _parse_rate(video.get("r_frame_rate")),
        duration=float(duration),
        audio_codec=audio.get("codec_name") if audio else None,
    )


@dataclass(frozen=True)
class PreparedImage:
    path: Path
    original_size: tuple[int, int]
    size: tuple[int, int]
    notes: list[str] = field(default_factory=list)


def prepare_image(source: str | Path, dest: str | Path, max_side: int = 0) -> PreparedImage:
    """Normaliza a foto do avatar sem cortar nada.

    - aplica a rotação EXIF (fotos de celular);
    - converte para RGB (transparência vira fundo branco);
    - reduz proporcionalmente se o maior lado passar de ``max_side`` (0 = nunca);
    - completa 1 px (repetindo a borda) quando largura/altura são ímpares;
    - salva como PNG, formato que todos os motores aceitam.
    """
    dest = Path(dest)
    notes: list[str] = []
    try:
        with Image.open(source) as opened:
            im = ImageOps.exif_transpose(opened)
            original_size = im.size

            if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
                rgba = im.convert("RGBA")
                im = Image.new("RGB", rgba.size, (255, 255, 255))
                im.paste(rgba, mask=rgba.getchannel("A"))
                notes.append("Transparência da foto preenchida com branco.")
            else:
                im = im.convert("RGB")
    except (OSError, ValueError) as exc:
        raise MediaError(f"Não foi possível abrir a imagem: {exc}") from exc

    if max_side and max(im.size) > max_side:
        scale = max_side / max(im.size)
        new_size = (max(1, round(im.width * scale)), max(1, round(im.height * scale)))
        im = im.resize(new_size, Image.Resampling.LANCZOS)
        notes.append(
            f"Foto reduzida de {original_size[0]}x{original_size[1]} para "
            f"{new_size[0]}x{new_size[1]} (limite HEYGEN_MAX_SIDE={max_side}), proporção mantida."
        )

    w, h = im.size
    if w % 2 or h % 2:
        padded = Image.new("RGB", (even(w), even(h)))
        padded.paste(im, (0, 0))
        if even(w) > w:
            padded.paste(im.crop((w - 1, 0, w, h)), (w, 0))
        if even(h) > h:
            padded.paste(padded.crop((0, h - 1, even(w), h)), (0, h))
        im = padded
        notes.append(
            f"Foto completada para {im.width}x{im.height} (+1 px de borda): "
            "H.264 exige largura e altura pares."
        )

    dest.parent.mkdir(parents=True, exist_ok=True)
    im.save(dest, "PNG")
    return PreparedImage(path=dest, original_size=original_size, size=im.size, notes=notes)


def feather_mask(size: tuple[int, int], inset: float, feather: float, dest: str | Path) -> Path:
    """Máscara em tons de cinza: branca no centro, transição suave até as bordas."""
    w, h = size
    pad_x, pad_y = int(w * inset), int(h * inset)
    radius = max(1.0, min(w, h) * feather)
    mask = Image.new("L", (w, h), 0)
    ImageDraw.Draw(mask).rectangle([pad_x, pad_y, w - 1 - pad_x, h - 1 - pad_y], fill=255)
    mask = mask.filter(ImageFilter.GaussianBlur(radius))
    dest = Path(dest)
    mask.save(dest, "PNG")
    return dest


def paste_back(
    base_image: str | Path,
    face_video: str | Path,
    rect: tuple[int, int, int, int],
    out_path: str | Path,
    *,
    inset: float = 0.12,
    feather: float = 0.06,
) -> Path:
    """Cola um vídeo do rosto (recortado) de volta na foto inteira, na resolução da foto.

    ``rect`` é (x0, y0, x1, y1) na foto. As bordas do recorte são suavizadas
    para não aparecer a "caixa" do rosto; o resto do quadro é a foto original.
    """
    out_path = Path(out_path)
    x0, y0, x1, y1 = (int(v) for v in rect)
    rw, rh = x1 - x0, y1 - y0
    if rw <= 0 or rh <= 0:
        raise MediaError(f"Região do rosto inválida: {rect}")

    face = probe_video(face_video)
    mask = feather_mask((rw, rh), inset, feather, out_path.with_name(out_path.stem + "_mask.png"))
    fps = f"{face.fps:.3f}" if face.fps else "25"

    run_ffmpeg([
        "-loop", "1", "-framerate", fps, "-i", str(base_image),
        "-i", str(face_video),
        "-i", str(mask),
        "-filter_complex",
        f"[1:v]scale={rw}:{rh}:flags=lanczos,format=rgba[fg];"
        "[2:v]format=gray[m];"
        "[fg][m]alphamerge[fga];"
        f"[0:v][fga]overlay={x0}:{y0}:shortest=1,format=yuv420p[v]",
        "-map", "[v]", "-an",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "16",
        str(out_path),
    ])
    return out_path

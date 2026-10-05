"""TalkingHeadEngine: a única porta entre o HeyGenHome e os motores de animação.

A interface, o Kokoro e a exportação conversam só com esta classe. Ela:
1. prepara a foto (EXIF, RGB, PNG, dimensões pares, sem cortar);
2. ajusta as opções ao que o motor suporta (e explica o que mudou);
3. executa o motor escolhido;
4. lê o MP4 gerado e confere o contrato de resolução/proporção.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from ..media import MediaError, PreparedImage, VideoInfo, prepare_image, probe_video
from .base import (
    FRAMING_CROP,
    FRAMING_FULL,
    FRAMINGS,
    MOTION_NATURAL,
    MOTIONS,
    AnimationOptions,
    AnimationRequest,
    EngineError,
    ProgressCallback,
    TalkingHeadBackend,
)

ASPECT_TOLERANCE = 0.01
DURATION_TOLERANCE = 0.5


@dataclass
class EngineRun:
    backend: TalkingHeadBackend
    request: AnimationRequest
    image: PreparedImage
    video: VideoInfo
    seconds: float
    log: Path | None = None
    notes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    details: dict = field(default_factory=dict)


class TalkingHeadEngine:
    def __init__(self, backends: Iterable[TalkingHeadBackend], max_side: int = 0):
        self._backends = {b.key: b for b in backends}
        self.max_side = max_side

    def backends(self) -> list[TalkingHeadBackend]:
        return list(self._backends.values())

    def get(self, key: str) -> TalkingHeadBackend:
        try:
            return self._backends[key]
        except KeyError:
            raise EngineError(f"Motor desconhecido: {key}") from None

    def build_request(
        self,
        backend: TalkingHeadBackend,
        image: Path,
        audio: Path,
        work_dir: Path,
        options: AnimationOptions,
        audio_duration: float | None = None,
        progress: ProgressCallback | None = None,
    ) -> tuple[AnimationRequest, list[str]]:
        """Converte as escolhas da interface no pedido que o motor consegue atender."""
        caps = backend.capabilities
        notes = []

        if options.framing not in FRAMINGS:
            raise EngineError(f"Enquadramento inválido: {options.framing}")
        framing = options.framing
        if framing == FRAMING_CROP and not caps.crop:
            notes.append(f"{backend.label} não tem modo recorte; usando a foto inteira.")
            framing = FRAMING_FULL

        size = options.size
        if size not in caps.sizes:
            notes.append(f"{backend.label} gera {caps.sizes[0]} px; {size} px não se aplica.")
            size = caps.sizes[0]

        if options.motion not in MOTIONS:
            raise EngineError(f"Movimento inválido: {options.motion}")
        motion = options.motion
        if not caps.motion:
            notes.append(f"O movimento do {backend.label} vem do próprio modelo (presets não se aplicam).")
            motion = MOTION_NATURAL
        if options.enhancer and not caps.enhancer:
            notes.append(f"'Melhorar rosto' não se aplica ao {backend.label}.")

        request = AnimationRequest(
            image=image,
            audio=audio,
            work_dir=work_dir,
            size=size,
            framing=framing,
            motion=motion,
            enhancer=options.enhancer and caps.enhancer,
            audio_duration=audio_duration,
            progress=progress,
        )
        return request, notes

    def animate(
        self,
        key: str,
        photo: str | Path,
        audio: str | Path,
        work_dir: str | Path,
        options: AnimationOptions = AnimationOptions(),
        audio_duration: float | None = None,
        progress: ProgressCallback | None = None,
    ) -> EngineRun:
        backend = self.get(key)
        status = backend.status()
        if not status.ready:
            raise EngineError(
                f"O motor {backend.label} ({backend.name}) não está pronto:\n- "
                + "\n- ".join(status.problems)
            )

        work_dir = Path(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        try:
            image = prepare_image(photo, work_dir / "foto.png", self.max_side)
        except MediaError as exc:
            raise EngineError(str(exc)) from exc

        request, notes = self.build_request(
            backend, image.path, Path(audio), work_dir, options, audio_duration, progress
        )

        start = time.monotonic()
        result = backend.animate(request)
        seconds = time.monotonic() - start

        try:
            info = probe_video(result.video)
        except MediaError as exc:
            raise EngineError(f"O {backend.name} gerou um vídeo ilegível: {exc}") from exc

        check_notes, warnings = check_output(request, image.size, info)
        return EngineRun(
            backend=backend,
            request=request,
            image=image,
            video=info,
            seconds=seconds,
            log=result.log,
            notes=image.notes + notes + result.notes + check_notes,
            warnings=warnings,
            details=result.details,
        )


def check_output(
    request: AnimationRequest, image_size: tuple[int, int], info: VideoInfo
) -> tuple[list[str], list[str]]:
    """Confere o MP4 do motor contra o contrato. Devolve (observações, avisos)."""
    notes, warnings = [], []
    if request.framing == FRAMING_FULL and info.size != image_size:
        iw, ih = image_size
        if abs(info.width / info.height - iw / ih) <= ASPECT_TOLERANCE * (iw / ih):
            notes.append(
                f"Motor entregou {info.resolution} (mesma proporção da foto); "
                f"a exportação Original volta para {iw}x{ih}."
            )
        else:
            warnings.append(
                f"O motor entregou {info.resolution}, com proporção diferente da foto ({iw}x{ih}). "
                "A exportação encaixa o vídeo inteiro, sem cortar."
            )
    if (
        request.audio_duration
        and info.duration
        and abs(info.duration - request.audio_duration) > DURATION_TOLERANCE
    ):
        warnings.append(
            f"O vídeo do motor tem {info.duration:.2f}s e o áudio {request.audio_duration:.2f}s."
        )
    return notes, warnings

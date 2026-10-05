"""Contrato comum dos motores de avatar falante.

Cada motor (SadTalkerNext, EchoMimic, ...) recebe uma foto já preparada
(PNG RGB com dimensões pares) e um WAV, e devolve um MP4 com o avatar.

Contrato de resolução, verificado pelo TalkingHeadEngine:
- enquadramento ``full``: o MP4 tem a proporção (normalmente a resolução exata)
  da foto preparada;
- enquadramento ``crop``: o MP4 mostra só o rosto.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

FRAMING_FULL = "full"
FRAMING_CROP = "crop"
FRAMINGS = (FRAMING_FULL, FRAMING_CROP)

# Quanto a cabeça e o rosto se mexem (motores com capabilities.motion).
MOTION_STILL = "estavel"
MOTION_NATURAL = "natural"
MOTION_EXPRESSIVE = "expressivo"
MOTIONS = (MOTION_STILL, MOTION_NATURAL, MOTION_EXPRESSIVE)

# progress(fração 0..1, mensagem)
ProgressCallback = Callable[[float, str], None]


class EngineError(RuntimeError):
    """Falha de configuração ou de execução de um motor."""


@dataclass(frozen=True)
class EngineCapabilities:
    sizes: tuple[int, ...] = (512,)
    motion: bool = True
    enhancer: bool = True
    crop: bool = True


@dataclass(frozen=True)
class AnimationOptions:
    """O que o usuário escolhe na interface."""

    size: int = 512
    framing: str = FRAMING_FULL
    motion: str = MOTION_NATURAL
    enhancer: bool = False


@dataclass(frozen=True)
class AnimationRequest:
    image: Path
    audio: Path
    work_dir: Path
    size: int = 512
    framing: str = FRAMING_FULL
    motion: str = MOTION_NATURAL
    enhancer: bool = False
    audio_duration: float | None = None
    progress: ProgressCallback | None = field(default=None, compare=False, repr=False)

    def report(self, fraction: float, message: str) -> None:
        if self.progress is not None:
            self.progress(min(1.0, max(0.0, fraction)), message)


@dataclass
class AnimationResult:
    video: Path
    log: Path | None = None
    notes: list[str] = field(default_factory=list)
    # Detalhes da execução para o relatório (dispositivo, s/quadro, ...).
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EngineStatus:
    ready: bool
    problems: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()


class TalkingHeadBackend(ABC):
    """Um motor concreto. Para adicionar um novo, implemente esta classe e
    registre-o em ``heygenhome.engines.create_engine``."""

    key: str
    label: str
    name: str
    description: str
    capabilities: EngineCapabilities = EngineCapabilities()

    @abstractmethod
    def status(self) -> EngineStatus:
        """Diz se o motor está instalado/configurado, sem executá-lo."""

    @abstractmethod
    def animate(self, request: AnimationRequest) -> AnimationResult:
        """Gera o vídeo. Deve levantar EngineError com uma mensagem útil se falhar."""

    def hardware(self) -> str | None:
        """Uma linha sobre onde o motor vai rodar (GPU/CPU), ou None se ainda não se sabe."""
        return None


def venv_python(root: Path, venv: str = "venv") -> Path:
    if os.name == "nt":
        return root / venv / "Scripts" / "python.exe"
    return root / venv / "bin" / "python"


def env_int(name: str, default: int) -> int:
    value = os.getenv(name, "").strip()
    try:
        return int(value) if value else default
    except ValueError:
        raise EngineError(f"{name} deve ser um número inteiro (atual: {value!r}).") from None


def env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name, "").strip().lower()
    if not value:
        return default
    return value in ("1", "true", "sim", "yes", "on")


def env_choice(name: str, default: str, choices: tuple[str, ...]) -> str:
    value = os.getenv(name, "").strip().lower() or default
    if value not in choices and not value.startswith("cuda:"):
        raise EngineError(f"{name} deve ser um de {', '.join(choices)} (atual: {value!r}).")
    return value

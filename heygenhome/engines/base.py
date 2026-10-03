"""Contrato comum dos motores de avatar falante.

Cada motor (SadTalkerNext, EchoMimic, ...) recebe uma foto já preparada
(PNG RGB com dimensões pares) e um WAV, e devolve um MP4 com o avatar.

Contrato de resolução, verificado pelo TalkingHeadEngine:
- enquadramento ``full``: o MP4 tem exatamente a resolução da foto preparada;
- enquadramento ``crop``: o MP4 mostra só o rosto e tem ``size`` px de largura.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

FRAMING_FULL = "full"
FRAMING_CROP = "crop"
FRAMINGS = (FRAMING_FULL, FRAMING_CROP)


class EngineError(RuntimeError):
    """Falha de configuração ou de execução de um motor."""


@dataclass(frozen=True)
class EngineCapabilities:
    sizes: tuple[int, ...] = (512,)
    still: bool = True
    enhancer: bool = True
    crop: bool = True


@dataclass(frozen=True)
class AnimationOptions:
    """O que o usuário escolhe na interface. Os padrões são os recomendados para CPU."""

    size: int = 512
    framing: str = FRAMING_FULL
    still: bool = True
    enhancer: bool = False


@dataclass(frozen=True)
class AnimationRequest:
    image: Path
    audio: Path
    work_dir: Path
    size: int = 512
    framing: str = FRAMING_FULL
    still: bool = True
    enhancer: bool = False
    audio_duration: float | None = None


@dataclass
class AnimationResult:
    video: Path
    log: Path | None = None
    notes: list[str] = field(default_factory=list)


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

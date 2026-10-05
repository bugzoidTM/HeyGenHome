"""Motores de avatar falante atrás de uma interface única (TalkingHeadEngine).

Para adicionar um motor novo: crie uma subclasse de TalkingHeadBackend e
inclua-a na lista de ``create_engine``. Kokoro, interface e exportação não mudam.
"""

from .base import (
    FRAMING_CROP,
    FRAMING_FULL,
    MOTION_EXPRESSIVE,
    MOTION_NATURAL,
    MOTION_STILL,
    MOTIONS,
    AnimationOptions,
    AnimationRequest,
    AnimationResult,
    EngineCapabilities,
    EngineError,
    EngineStatus,
    TalkingHeadBackend,
    env_int,
)
from .echomimic import EchoMimicBackend
from .engine import EngineRun, TalkingHeadEngine
from .sadtalker_next import SadTalkerNextBackend

# Fotos maiores que isso (maior lado, em px) são reduzidas proporcionalmente,
# nunca cortadas. Evita vídeos 4K/12 MP que estouram RAM e tempo em CPU.
DEFAULT_MAX_SIDE = 1920


def create_engine(max_side: int | None = None) -> TalkingHeadEngine:
    """Motores disponíveis, na ordem em que aparecem na interface."""
    if max_side is None:
        max_side = env_int("HEYGEN_MAX_SIDE", DEFAULT_MAX_SIDE)
    return TalkingHeadEngine(
        [
            SadTalkerNextBackend(),  # Standard
            EchoMimicBackend(),      # Experimental HD
        ],
        max_side=max_side,
    )


__all__ = [
    "FRAMING_CROP",
    "FRAMING_FULL",
    "MOTION_EXPRESSIVE",
    "MOTION_NATURAL",
    "MOTION_STILL",
    "MOTIONS",
    "AnimationOptions",
    "AnimationRequest",
    "AnimationResult",
    "EchoMimicBackend",
    "EngineCapabilities",
    "EngineError",
    "EngineRun",
    "EngineStatus",
    "SadTalkerNextBackend",
    "TalkingHeadBackend",
    "TalkingHeadEngine",
    "create_engine",
]

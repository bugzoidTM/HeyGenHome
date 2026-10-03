"""Motor "Standard": SadTalker oficial com os padrões otimizados para CPU.

- 512 px por padrão (checkpoint SadTalker_V0.0.2_512.safetensors);
- ``--preprocess full --still`` por padrão: anima o rosto e o recoloca na
  foto inteira, na resolução original;
- ``crop`` só quando escolhido explicitamente;
- GFPGAN (``--enhancer gfpgan``) opcional;
- ``--cpu`` (configurável por SADTALKER_DEVICE).
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from ._process import run_logged
from .base import (
    FRAMING_FULL,
    AnimationRequest,
    AnimationResult,
    EngineCapabilities,
    EngineError,
    EngineStatus,
    TalkingHeadBackend,
    venv_python,
)

DEFAULT_DIR = r"C:\AI\SadTalker"
DOWNLOAD_HINT = (
    "Baixe com o scripts/download_models.sh do SadTalker ou manualmente de "
    "https://github.com/OpenTalker/SadTalker/releases/tag/v0.0.2-rc "
    "para a pasta checkpoints."
)
RESULT_RE = re.compile(r"The generated video is named:\s*(.+?\.mp4)\s*$", re.MULTILINE)
NO_FACE_MARKERS = ("Can't get the coeffs", "No face is detected")


class SadTalkerNextBackend(TalkingHeadBackend):
    key = "sadtalker_next"
    label = "Standard"
    name = "SadTalkerNext"
    description = "SadTalker otimizado para CPU: estável e mais rápido."
    capabilities = EngineCapabilities(sizes=(512, 256), still=True, enhancer=True, crop=True)

    def __init__(
        self,
        root: str | Path | None = None,
        python: str | Path | None = None,
        device: str | None = None,
    ):
        self.root = Path(root or os.getenv("SADTALKER_DIR") or DEFAULT_DIR)
        self.python = Path(python or os.getenv("SADTALKER_PYTHON") or venv_python(self.root))
        self.device = (device or os.getenv("SADTALKER_DEVICE") or "cpu").strip().lower()

    @property
    def inference(self) -> Path:
        return self.root / "inference.py"

    @property
    def checkpoints(self) -> Path:
        return self.root / "checkpoints"

    def required_checkpoints(self, size: int, framing: str) -> list[str]:
        # O facerender "still" (usado no full) e o normal usam mapping diferentes.
        mapping = "mapping_00109-model.pth.tar" if framing == FRAMING_FULL else "mapping_00229-model.pth.tar"
        return [f"SadTalker_V0.0.2_{size}.safetensors", mapping]

    def missing_checkpoints(self, size: int, framing: str) -> list[str]:
        return [n for n in self.required_checkpoints(size, framing) if not (self.checkpoints / n).is_file()]

    def status(self) -> EngineStatus:
        problems = []
        if not self.inference.is_file():
            problems.append(f"SadTalker não encontrado: {self.inference} (configure SADTALKER_DIR).")
        if not self.python.is_file():
            problems.append(f"Python do SadTalker não encontrado: {self.python} (configure SADTALKER_PYTHON).")

        warnings = []
        if not problems:
            missing = self.missing_checkpoints(512, FRAMING_FULL)
            if missing:
                warnings.append(f"Faltam checkpoints do padrão 512/full: {', '.join(missing)}. {DOWNLOAD_HINT}")
            if not (self.root / "gfpgan" / "weights" / "GFPGANv1.4.pth").is_file():
                warnings.append("GFPGAN ainda não baixado: o primeiro uso de 'Melhorar rosto' vai baixar ~350 MB.")
        return EngineStatus(ready=not problems, problems=tuple(problems), warnings=tuple(warnings))

    def command(self, request: AnimationRequest, result_dir: Path) -> list[str]:
        cmd = [
            str(self.python), str(self.inference),
            "--driven_audio", str(request.audio),
            "--source_image", str(request.image),
            "--result_dir", str(result_dir),
            "--checkpoint_dir", str(self.checkpoints),
            "--size", str(request.size),
            "--preprocess", request.framing,
        ]
        if request.still:
            cmd.append("--still")
        if request.enhancer:
            cmd += ["--enhancer", "gfpgan"]
        if self.device == "cpu":
            cmd.append("--cpu")
        return cmd

    def animate(self, request: AnimationRequest) -> AnimationResult:
        missing = self.missing_checkpoints(request.size, request.framing)
        if missing:
            raise EngineError(
                f"Checkpoint(s) do SadTalker para {request.size}px/{request.framing} não encontrado(s) "
                f"em {self.checkpoints}: {', '.join(missing)}. {DOWNLOAD_HINT}"
            )

        result_dir = request.work_dir / "sadtalker"
        log_path = request.work_dir / "sadtalker.log"
        output = run_logged(
            self.command(request, result_dir), cwd=self.root, log_path=log_path, label="SadTalker"
        )
        return AnimationResult(video=self._find_video(output, result_dir, log_path), log=log_path)

    def _find_video(self, output: str, result_dir: Path, log_path: Path) -> Path:
        matches = RESULT_RE.findall(output)
        if matches:
            video = Path(matches[-1].strip())
            if not video.is_absolute():
                video = self.root / video
            if video.is_file():
                return video

        videos = sorted(result_dir.glob("*.mp4"), key=lambda p: p.stat().st_mtime)
        if videos:
            return videos[-1]

        if any(marker in output for marker in NO_FACE_MARKERS):
            raise EngineError(
                "O SadTalker não detectou um rosto na foto. Use uma foto frontal, "
                f"bem iluminada e com o rosto visível. Log: {log_path}"
            )
        raise EngineError(f"O SadTalker terminou, mas nenhum MP4 foi gerado. Log: {log_path}")

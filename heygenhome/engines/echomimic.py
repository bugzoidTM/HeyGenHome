"""Motor "Experimental HD": EchoMimic (v1), modelo de difusão.

O EchoMimic só anima um recorte quadrado do rosto (512x512). Para manter o
padrão "foto inteira" sem cortar nada, o rosto animado é recolocado na foto
original, na resolução original (``media.paste_back``). O recorte só aparece
no vídeo final quando o usuário escolhe ``crop``.

Em CPU usa por padrão os pesos acelerados (6 passos). Mesmo assim é muito mais
lento que o Standard: medido ~14 min por segundo de vídeo em 4 núcleos, com
pico de ~13 GB de RAM.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

from ..media import MediaError, paste_back
from ._process import EngineProcessError, run_logged
from .base import (
    FRAMING_CROP,
    AnimationRequest,
    AnimationResult,
    EngineCapabilities,
    EngineError,
    EngineStatus,
    TalkingHeadBackend,
    env_choice,
    env_flag,
    env_int,
    venv_python,
)
from .gpu import GpuProbe

DEFAULT_DIR = r"C:\AI\EchoMimic"
RUNNER = Path(__file__).with_name("echomimic_runner.py")
MIN_VRAM_GB = 12  # fp16, modo acelerado
NO_FACE_MARKER = "HG_ERROR no_face"


class EchoMimicBackend(TalkingHeadBackend):
    key = "echomimic"
    label = "Experimental HD"
    name = "EchoMimic"
    description = "EchoMimic (difusão): expressões mais naturais; precisa de GPU NVIDIA de 12 GB para ser prático."
    capabilities = EngineCapabilities(sizes=(512,), motion=False, enhancer=False, crop=True)

    def __init__(
        self,
        root: str | Path | None = None,
        python: str | Path | None = None,
        device: str | None = None,
        accelerated: bool | None = None,
    ):
        self.root = Path(root or os.getenv("ECHOMIMIC_DIR") or DEFAULT_DIR)
        self.python = Path(python or os.getenv("ECHOMIMIC_PYTHON") or venv_python(self.root))
        self.device = (device or env_choice("ECHOMIMIC_DEVICE", "auto", ("auto", "cpu", "cuda"))).lower()
        self.accelerated = env_flag("ECHOMIMIC_ACCELERATED", True) if accelerated is None else accelerated
        self.steps = env_int("ECHOMIMIC_STEPS", 0)
        self.fps = env_int("ECHOMIMIC_FPS", 24)
        self.seed = env_int("ECHOMIMIC_SEED", 420)
        # O mínimo de VRAM só vale no modo auto; cuda forçado usa a GPU mesmo assim (como o runner).
        self.gpu = GpuProbe(self.python, self.root, min_vram_gb=MIN_VRAM_GB if self.device == "auto" else 0)

    @property
    def weights_dir(self) -> Path:
        return self.root / "pretrained_weights"

    def expected_weights(self) -> list[Path]:
        suffix = "_acc" if self.accelerated else ""
        w = self.weights_dir
        return [
            w / f"denoising_unet{suffix}.pth",
            w / f"motion_module{suffix}.pth",
            w / "reference_unet.pth",
            w / "face_locator.pth",
            w / "sd-vae-ft-mse",
            w / "sd-image-variations-diffusers",
            w / "audio_processor" / "whisper_tiny.pt",
        ]

    def status(self) -> EngineStatus:
        problems = []
        if not (self.root / "src" / "pipelines").is_dir():
            problems.append(f"EchoMimic não encontrado em {self.root} (configure ECHOMIMIC_DIR).")
        if not self.python.is_file():
            problems.append(f"Python do EchoMimic não encontrado: {self.python} (configure ECHOMIMIC_PYTHON).")

        warnings = []
        if not problems:
            missing = [str(p.relative_to(self.root)) for p in self.expected_weights() if not p.exists()]
            if missing:
                warnings.append("Pesos não encontrados: " + ", ".join(missing) + ".")
        report = self.gpu.result() if self.device != "cpu" else None
        if self.device == "cpu" or (report is not None and not report.uses_gpu):
            warnings.append("Em CPU: ~14 min por segundo de vídeo (4 núcleos) e ~13 GB de RAM.")
        return EngineStatus(ready=not problems, problems=tuple(problems), warnings=tuple(warnings))

    def hardware(self) -> str | None:
        if self.device == "cpu":
            return "CPU (ECHOMIMIC_DEVICE=cpu)."
        report = self.gpu.result()
        return None if report is None else report.summary

    def command(self, request: AnimationRequest, face_video: Path, meta: Path) -> list[str]:
        cmd = [
            str(self.python), str(RUNNER),
            "--image", str(request.image),
            "--audio", str(request.audio),
            "--output", str(face_video),
            "--meta", str(meta),
            "--size", str(request.size),
            "--device", self.device,
            "--min_vram_gb", str(MIN_VRAM_GB),
            "--fps", str(self.fps),
            "--seed", str(self.seed),
        ]
        if request.audio_duration:
            cmd += ["--frames", str(math.ceil(request.audio_duration * self.fps))]
        if self.accelerated:
            cmd.append("--accelerated")
        if self.device != "cpu":
            cmd.append("--fp16")  # como no EchoMimic oficial; o runner só aplica em GPU
        if self.steps:
            cmd += ["--steps", str(self.steps)]
        return cmd

    def animate(self, request: AnimationRequest) -> AnimationResult:
        work = request.work_dir / "echomimic"
        work.mkdir(parents=True, exist_ok=True)
        face_video = work / "rosto.mp4"
        meta_path = work / "rosto.json"
        log_path = request.work_dir / "echomimic.log"

        def on_line(line: str) -> None:
            tag, _, payload = line.partition(" ")
            if tag == "HG_PROGRESS":
                try:
                    data = json.loads(payload)
                except ValueError:
                    return
                request.report(float(data.get("fraction", 0)), data.get("message", "EchoMimic…"))

        try:
            run_logged(
                self.command(request, face_video, meta_path),
                cwd=self.root, log_path=log_path, label="EchoMimic", on_line=on_line,
            )
        except EngineProcessError as exc:
            if NO_FACE_MARKER in exc.output:
                raise EngineError(
                    "O EchoMimic não detectou um rosto na foto. Use uma foto frontal, "
                    f"bem iluminada e com o rosto visível. Log: {log_path}"
                ) from exc
            raise
        if not face_video.is_file() or not meta_path.is_file():
            raise EngineError(f"O EchoMimic terminou sem gerar o vídeo do rosto. Log: {log_path}")

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if request.framing == FRAMING_CROP:
            return AnimationResult(video=face_video, log=log_path, details=meta)

        try:
            full = paste_back(request.image, face_video, tuple(meta["crop_rect"]), work / "foto_inteira.mp4")
        except (KeyError, MediaError) as exc:
            raise EngineError(f"Não foi possível recolocar o rosto na foto: {exc}") from exc
        return AnimationResult(
            video=full,
            log=log_path,
            notes=[f"Rosto animado em {request.size}x{request.size} e recolocado na foto inteira."],
            details=meta,
        )

"""Motor "Standard": SadTalkerNext, o SadTalker oficial com as melhorias do HeyGenHome.

Usa os modelos e o código do repositório oficial (https://github.com/OpenTalker/SadTalker)
por meio do ``sadtalker_runner.py``, que roda com o Python do SadTalker sem alterar
o clone. Em relação ao ``inference.py`` oficial:

- presets de movimento (cabeça, lábios e piscadas) em vez do ``--still`` congelado;
- GPU NVIDIA automática quando disponível, com a CPU compondo e codificando em
  paralelo; senão, CPU (com lote 1 e a foto codificada uma única vez);
- colagem com máscara suave e uma única codificação H.264 na resolução da foto;
- GFPGAN opcional aplicado só no rosto, sem dobrar a resolução.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from ._process import EngineProcessError, run_logged
from .base import (
    MOTIONS,
    AnimationRequest,
    AnimationResult,
    EngineCapabilities,
    EngineError,
    EngineStatus,
    TalkingHeadBackend,
    env_choice,
    env_int,
    venv_python,
)
from .gpu import GpuProbe

DEFAULT_DIR = r"C:\AI\SadTalker"
RUNNER = Path(__file__).with_name("sadtalker_runner.py")
DOWNLOAD_HINT = (
    "Baixe de https://github.com/OpenTalker/SadTalker/releases/tag/v0.0.2-rc "
    "para a pasta checkpoints do SadTalker."
)
# Mesmos valores do sadtalker_runner.py (longe de 1-4: no Windows, abort() sai com 3).
EXIT_NO_FACE = 13
EXIT_DEVICE = 14
NO_FACE_MARKER = "HG_ERROR no_face"


def find_ffmpeg() -> str:
    from ..media import find_tool

    return find_tool("ffmpeg")


class SadTalkerNextBackend(TalkingHeadBackend):
    key = "sadtalker_next"
    label = "Standard"
    name = "SadTalkerNext"
    description = "SadTalker melhorado: movimento natural de cabeça, lábios e piscadas."
    capabilities = EngineCapabilities(sizes=(512, 256), motion=True, enhancer=True, crop=True)

    def __init__(
        self,
        root: str | Path | None = None,
        python: str | Path | None = None,
        device: str | None = None,
        runner: str | Path | None = None,
    ):
        self.root = Path(root or os.getenv("SADTALKER_DIR") or DEFAULT_DIR)
        self.python = Path(python or os.getenv("SADTALKER_PYTHON") or venv_python(self.root))
        self.device = (device or env_choice("SADTALKER_DEVICE", "auto", ("auto", "cpu", "cuda"))).lower()
        self.fp16 = env_choice("SADTALKER_FP16", "auto", ("auto", "on", "off"))
        self.batch = env_int("SADTALKER_BATCH", 0)
        self.seed = env_int("HEYGEN_SEED", 42)
        self.pose_style = env_int("SADTALKER_POSE_STYLE", 0)
        self.threads = env_int("HEYGEN_CPU_THREADS", 0)
        self.runner = Path(runner or RUNNER)
        self.gpu = GpuProbe(self.python, self.root)

    @property
    def checkpoints(self) -> Path:
        return self.root / "checkpoints"

    def required_checkpoints(self, size: int) -> list[str]:
        # O runner usa sempre o mapeamento do modo foto inteira (73 coeficientes).
        return [f"SadTalker_V0.0.2_{size}.safetensors", "mapping_00109-model.pth.tar"]

    def missing_checkpoints(self, size: int) -> list[str]:
        return [n for n in self.required_checkpoints(size) if not (self.checkpoints / n).is_file()]

    def status(self) -> EngineStatus:
        problems = []
        if not (self.root / "src" / "facerender").is_dir():
            problems.append(f"SadTalker não encontrado em {self.root} (configure SADTALKER_DIR).")
        if not self.python.is_file():
            problems.append(f"Python do SadTalker não encontrado: {self.python} (configure SADTALKER_PYTHON).")

        warnings = []
        if not problems:
            missing = self.missing_checkpoints(512)
            if missing:
                warnings.append(f"Faltam checkpoints do padrão 512: {', '.join(missing)}. {DOWNLOAD_HINT}")
            if not (self.root / "gfpgan" / "weights" / "GFPGANv1.4.pth").is_file():
                warnings.append("GFPGAN ainda não baixado: o primeiro uso de 'Melhorar rosto' vai baixar ~350 MB.")
        return EngineStatus(ready=not problems, problems=tuple(problems), warnings=tuple(warnings))

    def hardware(self) -> str | None:
        if self.device == "cpu":
            return "CPU (SADTALKER_DEVICE=cpu)."
        report = self.gpu.result()
        return None if report is None else report.summary

    def command(self, request: AnimationRequest, output: Path, meta: Path, device: str) -> list[str]:
        if request.motion not in MOTIONS:
            raise EngineError(f"Movimento inválido: {request.motion}")
        cmd = [
            str(self.python), str(self.runner),
            "--image", str(request.image),
            "--audio", str(request.audio),
            "--output", str(output),
            "--meta", str(meta),
            "--checkpoint_dir", str(self.checkpoints),
            "--size", str(request.size),
            "--framing", request.framing,
            "--motion", request.motion,
            "--device", device,
            "--fp16", self.fp16,
            "--seed", str(self.seed),
            "--pose_style", str(self.pose_style),
            "--ffmpeg", find_ffmpeg(),
        ]
        if self.batch:
            cmd += ["--batch", str(self.batch)]
        if self.threads:
            cmd += ["--threads", str(self.threads)]
        if request.enhancer:
            cmd.append("--enhancer")
        return cmd

    def animate(self, request: AnimationRequest) -> AnimationResult:
        missing = self.missing_checkpoints(request.size)
        if missing:
            raise EngineError(
                f"Checkpoint(s) do SadTalker para {request.size}px não encontrado(s) "
                f"em {self.checkpoints}: {', '.join(missing)}. {DOWNLOAD_HINT}"
            )

        work = request.work_dir / "sadtalker"
        work.mkdir(parents=True, exist_ok=True)
        output = work / "avatar.mp4"
        meta_path = work / "avatar.json"
        log_path = request.work_dir / "sadtalker.log"
        notes = []

        try:
            self._run(request, output, meta_path, log_path, self.device)
        except EngineProcessError as exc:
            if exc.returncode == EXIT_NO_FACE and NO_FACE_MARKER in exc.output:
                raise EngineError(
                    "O SadTalker não detectou um rosto na foto. Use uma foto frontal, "
                    f"bem iluminada e com o rosto visível. Log: {log_path}"
                ) from exc
            if exc.returncode != EXIT_DEVICE or self.device == "cpu":
                raise
            # Erro de GPU que não é falta de memória: refaz tudo em CPU.
            notes.append("A GPU falhou durante a geração; o vídeo foi refeito em CPU (veja o log).")
            log_path.replace(log_path.with_name("sadtalker-gpu-falhou.log"))
            self._run(request, output, meta_path, log_path, "cpu")

        if not output.is_file() or not meta_path.is_file():
            raise EngineError(f"O SadTalker terminou, mas nenhum vídeo foi gerado. Log: {log_path}")
        details = json.loads(meta_path.read_text(encoding="utf-8"))
        if details.get("fallback"):
            notes.append(f"A GPU parou no meio ({details['fallback']}); o restante foi renderizado em CPU.")
        return AnimationResult(video=output, log=log_path, notes=notes, details=details)

    def _run(self, request: AnimationRequest, output: Path, meta: Path, log_path: Path, device: str) -> None:
        stages = {"load": (0.02, "Carregando o SadTalker…"), "face": (0.06, "Analisando o rosto…"),
                  "audio": (0.08, "Calculando lábios, cabeça e piscadas…")}
        where = {"device": "CPU"}

        def on_line(line: str) -> None:
            tag, _, payload = line.partition(" ")
            if tag not in ("HG_PROGRESS", "HG_DEVICE"):
                return
            try:
                data = json.loads(payload)
            except ValueError:
                return
            if tag == "HG_DEVICE":
                device = str(data.get("device", "cpu"))
                where["device"] = (data.get("name") or device) if device.startswith("cuda") else "CPU"
                return
            stage = data.get("stage")
            if stage in stages:
                request.report(*stages[stage])
            elif stage == "render" and data.get("total"):
                done, total = int(data["done"]), int(data["total"])
                eta = (total - done) * float(data.get("sec_per_frame") or 0)
                request.report(
                    0.1 + 0.9 * done / total,
                    f"Renderizando quadro {done}/{total} em {where['device']} — {_eta(eta)} restantes",
                )

        run_logged(
            self.command(request, output, meta, device),
            cwd=self.root, log_path=log_path, label="SadTalker", on_line=on_line,
        )


def _eta(seconds: float) -> str:
    minutes, sec = divmod(int(round(seconds)), 60)
    return f"~{minutes} min {sec:02d} s" if minutes else f"~{sec} s"


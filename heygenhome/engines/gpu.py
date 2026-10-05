"""Detecção de GPU para os motores (cada motor tem seu próprio Python/PyTorch).

1. ``nvidia-smi`` (milissegundos, sem PyTorch): existe GPU NVIDIA?
2. Teste com o Python do motor: o PyTorch dele tem CUDA e roda as operações do
   renderizador (conv3d + grid_sample 5D) nessa GPU? Isso pega casos como
   "PyTorch só CPU" ou "placa nova demais para o PyTorch instalado".

O resultado vira uma linha para a interface e não decide nada sozinho: quem
escolhe o dispositivo é o runner de cada motor (``--device auto``).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
from dataclasses import dataclass, field
from pathlib import Path

from ._process import engine_env

PROBE_TIMEOUT = 240  # primeira carga do CUDA no Windows pode levar ~1 min
# Placas de "12 GB" reportam ~11,99 GiB; a mesma tolerância vale no echomimic_runner.
VRAM_TOLERANCE_GB = 0.5
PROBE_SCRIPT = r"""
import json
info = {}
try:
    import torch
    import torch.nn.functional as F
    info["torch"] = torch.__version__
    info["cuda_build"] = torch.version.cuda
    info["available"] = bool(torch.cuda.is_available())
    devices = []
    if info["available"]:
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            d = {"index": i, "name": p.name, "vram_gb": round(p.total_memory / 2 ** 30, 1),
                 "total_bytes": int(p.total_memory), "capability": "%d.%d" % (p.major, p.minor)}
            try:
                dev = torch.device("cuda", i)
                x = torch.randn(1, 4, 4, 16, 16, device=dev)
                w = torch.randn(4, 4, 3, 3, 3, device=dev)
                g = torch.rand(1, 4, 16, 16, 3, device=dev) * 2 - 1
                y = F.grid_sample(F.conv3d(x, w, padding=1), g, align_corners=False)
                torch.cuda.synchronize(dev)
                d["ok"] = bool(torch.isfinite(y).all())
            except Exception as exc:
                d["ok"] = False
                d["error"] = str(exc)[:300]
            devices.append(d)
    info["devices"] = devices
except Exception as exc:
    info["error"] = str(exc)[:300]
print("HG_PROBE " + json.dumps(info))
"""


def _no_window() -> int:
    return getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0


def nvidia_gpus() -> list[dict]:
    """GPUs NVIDIA segundo o driver; lista vazia se não houver."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return []
    try:
        out = subprocess.run(
            [exe, "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15, creationflags=_no_window(),
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    return parse_nvidia_smi(out)


def parse_nvidia_smi(text: str) -> list[dict]:
    gpus = []
    for line in text.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3 and parts[0]:
            try:
                vram = round(float(parts[1]) / 1024, 1)
            except ValueError:
                vram = None
            gpus.append({"name": parts[0], "vram_gb": vram, "driver": parts[2]})
    return gpus


def parse_probe(output: str) -> dict:
    for line in reversed(output.splitlines()):
        if line.startswith("HG_PROBE "):
            try:
                return json.loads(line[len("HG_PROBE "):])
            except ValueError:
                break
    return {"error": "o teste de GPU não respondeu"}


@dataclass
class GpuReport:
    uses_gpu: bool
    summary: str
    nvidia: list[dict] = field(default_factory=list)
    probe: dict = field(default_factory=dict)


def describe(nvidia: list[dict], probe: dict, min_vram_gb: float = 0.0) -> GpuReport:
    """Texto para a interface a partir do driver e do teste com o PyTorch do motor."""
    ok = [d for d in probe.get("devices", []) if d.get("ok")]
    if ok:
        best = max(ok, key=_vram_gib)  # mesma regra dos runners: a de mais VRAM
        label = f"GPU {best['name']} ({best['vram_gb']} GB)"
        if min_vram_gb and _vram_gib(best) + VRAM_TOLERANCE_GB < min_vram_gb:
            return GpuReport(False, f"{label} tem pouca memória para este motor "
                                    f"(recomendado {min_vram_gb:g} GB): usando CPU.", nvidia, probe)
        return GpuReport(True, f"{label} será usada, com a CPU ajudando na composição do vídeo.",
                         nvidia, probe)
    if nvidia:
        name = nvidia[0]["name"]
        failed = [d for d in probe.get("devices", []) if not d.get("ok")]
        if failed:
            return GpuReport(False, f"GPU {name} encontrada, mas o PyTorch deste motor não roda nela "
                                    f"({failed[0].get('error', 'erro desconhecido')[:120]}). Usando CPU; "
                                    "veja o README, seção GPU.", nvidia, probe)
        if probe.get("torch") and not probe.get("cuda_build"):
            return GpuReport(False, f"GPU {name} encontrada, mas o PyTorch deste motor ({probe['torch']}) "
                                    "é só para CPU. Usando CPU; veja o README, seção GPU.", nvidia, probe)
        if probe.get("cuda_build") and probe.get("available") is False:
            return GpuReport(False, f"GPU {name} encontrada, mas o PyTorch deste motor (CUDA {probe['cuda_build']}) "
                                    "não consegue usá-la; atualize o driver NVIDIA. Usando CPU.", nvidia, probe)
        return GpuReport(False, f"GPU {name} encontrada, mas não pôde ser testada "
                                f"({probe.get('error', 'sem detalhes')[:120]}). Usando CPU.", nvidia, probe)
    return GpuReport(False, "Sem GPU NVIDIA: usando só a CPU.", nvidia, probe)


def _vram_gib(device: dict) -> float:
    if device.get("total_bytes"):
        return device["total_bytes"] / 2 ** 30
    return float(device.get("vram_gb") or 0)


class GpuProbe:
    """Roda o teste uma vez, em segundo plano, e guarda o resultado."""

    def __init__(self, python: Path, cwd: Path, min_vram_gb: float = 0.0):
        self.python = Path(python)
        self.cwd = Path(cwd)
        self.min_vram_gb = min_vram_gb
        self._result: GpuReport | None = None
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def start(self) -> None:
        with self._lock:
            if self._thread is None:
                self._thread = threading.Thread(target=self._run, daemon=True)
                self._thread.start()

    def result(self, wait: float = 0.0) -> GpuReport | None:
        self.start()
        if self._thread is not None and wait:
            self._thread.join(wait)
        return self._result

    def _run(self) -> None:
        nvidia = nvidia_gpus()
        probe: dict = {}
        if nvidia and self.python.is_file():
            try:
                out = subprocess.run(
                    [str(self.python), "-c", PROBE_SCRIPT],
                    cwd=str(self.cwd) if self.cwd.is_dir() else None, env=engine_env(),
                    capture_output=True, text=True, encoding="utf-8", errors="replace",
                    timeout=PROBE_TIMEOUT, creationflags=_no_window(),
                ).stdout
                probe = parse_probe(out)
            except (OSError, subprocess.SubprocessError) as exc:
                probe = {"error": str(exc)[:200]}
        self._result = describe(nvidia, probe, self.min_vram_gb)

"""Execução dos motores em processos separados (cada um tem seu próprio Python)."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from .base import EngineError

TAIL_LINES = 15
TAIL_LINE_CHARS = 300
# Código de saída quando o sistema mata o processo (SIGKILL), em geral por falta de RAM.
KILLED_CODES = (-9, 137, 247)


def engine_env() -> dict[str, str]:
    env = os.environ.copy()
    # Evita UnicodeEncodeError do tqdm/print no console do Windows (cp1252).
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    threads = os.getenv("HEYGEN_CPU_THREADS", "").strip()
    if threads:
        env["OMP_NUM_THREADS"] = threads
        env["MKL_NUM_THREADS"] = threads
    return env


def log_tail(output: str) -> str:
    """Últimas linhas do log, cada uma encurtada (barras do tqdm viram linhas)."""
    lines = [line for line in output.replace("\r", "\n").splitlines() if line.strip()]
    tail = []
    for line in lines[-TAIL_LINES:]:
        tail.append(line if len(line) <= TAIL_LINE_CHARS else line[:TAIL_LINE_CHARS] + " […]")
    return "\n".join(tail)


def run_logged(cmd: list[str], *, cwd: Path, log_path: Path, label: str) -> str:
    """Executa ``cmd`` gravando toda a saída em ``log_path`` e devolve o texto do log."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8", errors="replace") as log:
        log.write("$ " + subprocess.list2cmdline(cmd) + "\n\n")
        log.flush()
        try:
            proc = subprocess.run(
                cmd, cwd=str(cwd), env=engine_env(), stdout=log, stderr=subprocess.STDOUT
            )
        except OSError as exc:
            raise EngineError(f"Não foi possível iniciar o {label}: {exc}") from exc

    output = log_path.read_text(encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        hint = ""
        if proc.returncode in KILLED_CODES or "MemoryError" in output:
            hint = (
                "\nO processo foi encerrado pelo sistema, provavelmente por falta de memória RAM. "
                "Feche outros programas ou use um motor/resolução mais leve.\n"
            )
        raise EngineError(
            f"{label} falhou (código {proc.returncode}).{hint}\nFinal do log:\n"
            f"{log_tail(output)}\n\nLog completo: {log_path}"
        )
    return output

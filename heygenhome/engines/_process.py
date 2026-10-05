"""Execução dos motores em processos separados (cada um tem seu próprio Python)."""

from __future__ import annotations

import codecs
import os
import subprocess
import time
from pathlib import Path
from typing import Callable

from .base import EngineError

TAIL_LINES = 15
TAIL_LINE_CHARS = 300
POLL_SECONDS = 0.5
# Código de saída quando o sistema mata o processo (SIGKILL), em geral por falta de RAM.
KILLED_CODES = (-9, 137, 247)


class EngineProcessError(EngineError):
    """O processo do motor terminou com erro; guarda o código de saída."""

    def __init__(self, message: str, returncode: int, output: str):
        super().__init__(message)
        self.returncode = returncode
        self.output = output


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


def _follow(proc, log_path: Path, start: int, on_line: Callable[[str], None] | None) -> None:
    """Acompanha o log enquanto o processo roda, entregando linhas completas a ``on_line``."""
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")  # não parte caracteres
    pending = ""
    with open(log_path, "rb") as reader:
        reader.seek(start)  # só a saída do processo, não o cabeçalho
        while True:
            finished = proc.poll() is not None
            chunk = decoder.decode(reader.read(), final=finished)
            if on_line is not None:
                pending += chunk.replace("\r", "\n")
                *lines, pending = pending.split("\n")
                if finished and pending.strip():
                    lines.append(pending)
                for line in lines:
                    if line.strip():
                        try:
                            on_line(line)
                        except Exception:  # um erro ao mostrar progresso não pode derrubar a geração
                            pass
            if finished:
                return
            time.sleep(POLL_SECONDS)


def run_logged(
    cmd: list[str],
    *,
    cwd: Path,
    log_path: Path,
    label: str,
    on_line: Callable[[str], None] | None = None,
) -> str:
    """Executa ``cmd`` gravando toda a saída em ``log_path`` e devolve o texto do log.

    ``on_line`` recebe cada linha nova enquanto o processo roda (para progresso).
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "wb") as log:
        log.write(("$ " + subprocess.list2cmdline(cmd) + "\n\n").encode("utf-8", "replace"))
        log.flush()
        header_end = log.tell()
        try:
            proc = subprocess.Popen(
                cmd, cwd=str(cwd), env=engine_env(), stdout=log, stderr=subprocess.STDOUT
            )
        except OSError as exc:
            raise EngineError(f"Não foi possível iniciar o {label}: {exc}") from exc

        try:
            _follow(proc, log_path, header_end, on_line)
        except BaseException:
            # Interrompido (ou erro inesperado): não deixa o motor rodando sozinho.
            proc.kill()
            proc.wait()
            raise

    output = log_path.read_text(encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        hint = ""
        if proc.returncode in KILLED_CODES or "MemoryError" in output:
            hint = (
                "\nO processo foi encerrado pelo sistema, provavelmente por falta de memória RAM. "
                "Feche outros programas ou use um motor/resolução mais leve.\n"
            )
        raise EngineProcessError(
            f"{label} falhou (código {proc.returncode}).{hint}\nFinal do log:\n"
            f"{log_tail(output)}\n\nLog completo: {log_path}",
            proc.returncode,
            output,
        )
    return output

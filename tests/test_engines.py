import json
import subprocess
import sys

import numpy as np
import pytest

from heygenhome.engines import (
    FRAMING_CROP,
    FRAMING_FULL,
    MOTION_EXPRESSIVE,
    MOTION_NATURAL,
    MOTION_STILL,
    AnimationOptions,
    AnimationRequest,
    EchoMimicBackend,
    EngineError,
    SadTalkerNextBackend,
    TalkingHeadEngine,
    create_engine,
)
from heygenhome.engines import echomimic as echomimic_module
from heygenhome.engines import gpu
from heygenhome.engines import sadtalker_runner as runner

from .conftest import make_photo, requires_ffmpeg


def sadtalker(root, device="auto"):
    return SadTalkerNextBackend(root=root, python=sys.executable, device=device,
                                runner=root / "fake_runner.py")


def request_for(tmp_path, **kwargs):
    return AnimationRequest(image=tmp_path / "f.png", audio=tmp_path / "v.wav", work_dir=tmp_path, **kwargs)


def argv_of(job):
    return (job / "sadtalker" / "argv.txt").read_text().strip().splitlines()


# --------------------------------------------------------------------------- registro e opções

def test_registry_order_and_labels():
    labels = [(b.label, b.name) for b in create_engine().backends()]
    assert labels == [("Standard", "SadTalkerNext"), ("Experimental HD", "EchoMimic")]


def test_default_options_are_512_full_natural_without_enhancer():
    assert AnimationOptions() == AnimationOptions(
        size=512, framing=FRAMING_FULL, motion=MOTION_NATURAL, enhancer=False
    )


def test_invalid_motion_is_rejected(tmp_path, fake_sadtalker):
    backend = sadtalker(fake_sadtalker)
    with pytest.raises(EngineError, match="Movimento inválido"):
        TalkingHeadEngine([backend]).build_request(
            backend, tmp_path / "f.png", tmp_path / "v.wav", tmp_path, AnimationOptions(motion="dança")
        )


# --------------------------------------------------------------------------- SadTalkerNext

def test_sadtalker_command_defaults(tmp_path, fake_sadtalker, monkeypatch):
    for var in ("SADTALKER_DEVICE", "SADTALKER_FP16", "SADTALKER_BATCH", "HEYGEN_SEED"):
        monkeypatch.delenv(var, raising=False)
    backend = SadTalkerNextBackend(root=fake_sadtalker, python=sys.executable)
    cmd = backend.command(request_for(tmp_path), tmp_path / "o.mp4", tmp_path / "o.json", backend.device)
    args = " ".join(cmd)
    assert cmd[1].endswith("sadtalker_runner.py")
    for expected in ("--size 512", "--framing full", "--motion natural", "--device auto", "--fp16 auto",
                     "--seed 42"):
        assert expected in args
    assert "--enhancer" not in cmd and "--batch" not in cmd


def test_sadtalker_command_crop_motion_and_enhancer_only_when_chosen(tmp_path, fake_sadtalker):
    req = request_for(tmp_path, framing=FRAMING_CROP, motion=MOTION_STILL, enhancer=True, size=256)
    cmd = sadtalker(fake_sadtalker, device="cpu").command(req, tmp_path / "o.mp4", tmp_path / "o.json", "cpu")
    args = " ".join(cmd)
    assert "--framing crop" in args and "--size 256" in args and "--motion estavel" in args
    assert "--enhancer" in cmd and "--device cpu" in args


def test_sadtalker_env_overrides(tmp_path, fake_sadtalker, monkeypatch):
    monkeypatch.setenv("SADTALKER_DEVICE", "cpu")
    monkeypatch.setenv("SADTALKER_BATCH", "3")
    monkeypatch.setenv("HEYGEN_CPU_THREADS", "6")
    backend = SadTalkerNextBackend(root=fake_sadtalker, python=sys.executable)
    args = " ".join(backend.command(request_for(tmp_path), tmp_path / "o", tmp_path / "m", backend.device))
    assert "--device cpu" in args and "--batch 3" in args and "--threads 6" in args
    assert backend.hardware() == "CPU (SADTALKER_DEVICE=cpu)."


def test_sadtalker_invalid_device_env(fake_sadtalker, monkeypatch):
    monkeypatch.setenv("SADTALKER_DEVICE", "tpu")
    with pytest.raises(EngineError, match="SADTALKER_DEVICE"):
        SadTalkerNextBackend(root=fake_sadtalker, python=sys.executable)


def test_sadtalker_missing_512_checkpoint_is_clear_error(tmp_path, fake_sadtalker, photo, wav):
    (fake_sadtalker / "checkpoints" / "SadTalker_V0.0.2_512.safetensors").unlink()
    engine = TalkingHeadEngine([sadtalker(fake_sadtalker)])
    with pytest.raises(EngineError, match="SadTalker_V0.0.2_512.safetensors"):
        engine.animate("sadtalker_next", photo, wav, tmp_path / "job")


def test_engine_not_installed(tmp_path, photo, wav):
    engine = TalkingHeadEngine([SadTalkerNextBackend(root=tmp_path / "nada", python=tmp_path / "py")])
    with pytest.raises(EngineError, match="não está pronto"):
        engine.animate("sadtalker_next", photo, wav, tmp_path / "job")


@requires_ffmpeg
def test_sadtalker_full_preserves_photo_resolution_and_reports_progress(tmp_path, fake_sadtalker, wav):
    photo = make_photo(tmp_path / "retrato.png", size=(721, 1001))
    engine = TalkingHeadEngine([sadtalker(fake_sadtalker)])
    events = []
    run = engine.animate("sadtalker_next", photo, wav, tmp_path / "job", audio_duration=1.6,
                         progress=lambda f, m: events.append((f, m)))
    assert run.image.size == (722, 1002)
    assert run.video.size == (722, 1002)
    assert run.warnings == []
    assert run.details["device_name"] == "Fake GPU"
    assert "--motion natural" in argv_of(tmp_path / "job")[0]
    fractions = [f for f, _ in events]
    assert fractions == sorted(fractions) and fractions[-1] == pytest.approx(1.0)
    assert any("Renderizando quadro" in m and "Fake GPU" in m for _, m in events)


@requires_ffmpeg
def test_sadtalker_crop_only_when_requested(tmp_path, fake_sadtalker, photo, wav):
    engine = TalkingHeadEngine([sadtalker(fake_sadtalker)])
    run = engine.animate(
        "sadtalker_next", photo, wav, tmp_path / "job", AnimationOptions(framing=FRAMING_CROP)
    )
    assert run.video.size == (512, 512)


@requires_ffmpeg
def test_sadtalker_gpu_failure_is_retried_on_cpu(tmp_path, fake_sadtalker, photo, wav, monkeypatch):
    monkeypatch.setenv("FAKE_GPU_FAILS", "1")
    engine = TalkingHeadEngine([sadtalker(fake_sadtalker)])
    run = engine.animate("sadtalker_next", photo, wav, tmp_path / "job")
    calls = argv_of(tmp_path / "job")
    assert len(calls) == 2 and "--device auto" in calls[0] and "--device cpu" in calls[1]
    assert run.details["device"] == "cpu"
    assert any("refeito em CPU" in n for n in run.notes)
    assert (tmp_path / "job" / "sadtalker-gpu-falhou.log").is_file()


@requires_ffmpeg
def test_sadtalker_no_face_message(tmp_path, fake_sadtalker, photo, wav, monkeypatch):
    monkeypatch.setenv("FAKE_NO_FACE", "1")
    engine = TalkingHeadEngine([sadtalker(fake_sadtalker)])
    with pytest.raises(EngineError, match="não detectou um rosto"):
        engine.animate("sadtalker_next", photo, wav, tmp_path / "job")


# --------------------------------------------------------------------------- EchoMimic

def test_echomimic_adapts_unsupported_options(tmp_path):
    backend = EchoMimicBackend(root=tmp_path, python=sys.executable)
    engine = TalkingHeadEngine([backend])
    req, notes = engine.build_request(
        backend, tmp_path / "f.png", tmp_path / "v.wav", tmp_path,
        AnimationOptions(size=256, motion=MOTION_EXPRESSIVE, enhancer=True),
    )
    assert (req.size, req.enhancer, req.framing) == (512, False, FRAMING_FULL)
    assert len(notes) == 3


def test_echomimic_command_is_auto_device_and_accelerated_by_default(tmp_path, monkeypatch):
    for var in ("ECHOMIMIC_DEVICE", "ECHOMIMIC_ACCELERATED", "ECHOMIMIC_STEPS", "ECHOMIMIC_FPS"):
        monkeypatch.delenv(var, raising=False)
    backend = EchoMimicBackend(root=tmp_path, python=sys.executable)
    cmd = backend.command(request_for(tmp_path, audio_duration=2.01), tmp_path / "r.mp4", tmp_path / "r.json")
    args = " ".join(cmd)
    assert cmd[1].endswith("echomimic_runner.py")
    assert "--device auto" in args and "--accelerated" in cmd and "--fp16" in cmd
    assert "--min_vram_gb 12" in args
    assert "--frames 49" in args  # ceil(2.01 * 24)
    cpu = EchoMimicBackend(root=tmp_path, python=sys.executable, device="cpu")
    assert "--fp16" not in cpu.command(request_for(tmp_path), tmp_path / "r.mp4", tmp_path / "r.json")


@requires_ffmpeg
@pytest.mark.parametrize("framing,expected", [(FRAMING_FULL, (600, 800)), (FRAMING_CROP, (512, 512))])
def test_echomimic_full_pastes_face_back_into_photo(tmp_path, monkeypatch, photo, wav, framing, expected):
    root = tmp_path / "EchoMimic"
    (root / "src" / "pipelines").mkdir(parents=True)
    events = []

    def fake_runner(cmd, *, cwd, log_path, label, on_line=None):
        out = cmd[cmd.index("--output") + 1]
        meta = cmd[cmd.index("--meta") + 1]
        on_line('HG_PROGRESS {"fraction": 0.5, "message": "Difusão: passo 3/6 em cpu"}')
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi",
             "-i", "testsrc2=size=512x512:rate=24:duration=1.6", "-c:v", "libx264", out],
            check=True,
        )
        with open(meta, "w") as fh:
            json.dump({"crop_rect": [50, 100, 450, 500], "image_size": [600, 800], "device": "cpu"}, fh)
        log_path.write_text("ok")
        return "ok"

    monkeypatch.setattr(echomimic_module, "run_logged", fake_runner)
    engine = TalkingHeadEngine([EchoMimicBackend(root=root, python=sys.executable, device="cpu")])
    run = engine.animate(
        "echomimic", photo, wav, tmp_path / "job", AnimationOptions(framing=framing), audio_duration=1.6,
        progress=lambda f, m: events.append((f, m)),
    )
    assert run.video.size == expected
    assert run.warnings == []
    assert run.details["device"] == "cpu"
    assert events == [(0.5, "Difusão: passo 3/6 em cpu")]


# --------------------------------------------------------------------------- execução e GPU

def test_engine_failure_shows_short_log_tail(tmp_path):
    from heygenhome.engines._process import EngineProcessError, run_logged

    script = "print('x' * 5000); print('ERRO: algo deu errado'); raise SystemExit(1)"
    with pytest.raises(EngineProcessError) as err:
        run_logged([sys.executable, "-c", script], cwd=tmp_path, log_path=tmp_path / "m.log", label="Motor")
    message = str(err.value)
    assert err.value.returncode == 1
    assert "ERRO: algo deu errado" in message
    assert "x" * 400 not in message  # linhas gigantes são encurtadas
    assert "memória" not in message


def test_run_logged_streams_lines(tmp_path):
    from heygenhome.engines._process import run_logged

    script = "import time\nfor i in range(3):\n    print('linha', i, flush=True)\n    time.sleep(0.3)"
    lines = []
    run_logged([sys.executable, "-c", script], cwd=tmp_path, log_path=tmp_path / "m.log",
               label="Motor", on_line=lines.append)
    assert lines == ["linha 0", "linha 1", "linha 2"]


@pytest.mark.skipif(sys.platform == "win32", reason="SIGKILL não existe no Windows")
def test_engine_killed_by_system_mentions_memory(tmp_path):
    from heygenhome.engines._process import run_logged

    script = "import os, signal; os.kill(os.getpid(), signal.SIGKILL)"
    with pytest.raises(EngineError, match="falta de memória"):
        run_logged([sys.executable, "-c", script], cwd=tmp_path, log_path=tmp_path / "m.log", label="Motor")


def test_parse_nvidia_smi():
    assert gpu.parse_nvidia_smi("NVIDIA GeForce RTX 3060, 12288, 552.22\n") == [
        {"name": "NVIDIA GeForce RTX 3060", "vram_gb": 12.0, "driver": "552.22"}
    ]
    assert gpu.parse_nvidia_smi("") == []


def test_gpu_describe_cases():
    rtx = [{"name": "NVIDIA GeForce RTX 3060", "vram_gb": 12.0, "driver": "552"}]
    ok = {"torch": "2.1.2+cu118", "cuda_build": "11.8",
          "devices": [{"name": "NVIDIA GeForce RTX 3060", "vram_gb": 12.0, "ok": True}]}
    assert gpu.describe(rtx, ok).uses_gpu
    assert "será usada" in gpu.describe(rtx, ok).summary
    cpu_torch = {"torch": "1.12.1+cpu", "cuda_build": None, "devices": []}
    report = gpu.describe(rtx, cpu_torch)
    assert not report.uses_gpu and "só para CPU" in report.summary
    broken = {"torch": "2.1.2+cu118", "cuda_build": "11.8",
              "devices": [{"name": "RTX 5090", "ok": False, "error": "no kernel image is available"}]}
    assert "no kernel image" in gpu.describe(rtx, broken).summary
    assert not gpu.describe([], {}).uses_gpu
    small = gpu.describe(rtx, ok, min_vram_gb=16)
    assert not small.uses_gpu and "pouca memória" in small.summary


def test_parse_probe():
    out = 'lixo\nHG_PROBE {"torch": "2.1.2", "available": true, "devices": []}\n'
    assert gpu.parse_probe(out)["torch"] == "2.1.2"
    assert "error" in gpu.parse_probe("nada")


# --------------------------------------------------------------------------- funções do runner

def test_runner_presets_are_ordered_by_intensity():
    still, natural, expressive = (runner.MOTION_PRESETS[k] for k in ("estavel", "natural", "expressivo"))
    assert not any(still["pose_rms"]) and not any(still["idle_rms"])
    assert all(a < b for a, b in zip(natural["pose_rms"], expressive["pose_rms"]))
    assert still["mouth"] < natural["mouth"] < expressive["mouth"]


def test_head_motion_respects_limits_and_starts_and_ends_at_photo_pose():
    rng = np.random.default_rng(0)
    raw = np.cumsum(rng.normal(0, 2, size=(200, 3)), axis=0)  # movimento grande e aleatório
    preset = runner.MOTION_PRESETS["natural"]
    d = runner.head_motion(raw, preset, np.random.default_rng(1))
    assert d.shape == (200, 3)
    assert np.all(np.abs(d) <= np.asarray(preset["pose_limit"]) + 1e-9)
    assert np.allclose(d[0], 0) and np.allclose(d[-1], 0)
    assert np.abs(np.diff(d, axis=0)).max() < 1.0  # sem saltos entre quadros


def test_head_motion_amplitude_follows_the_preset_with_capped_gain():
    t = np.linspace(0, 12, 300)
    preset = dict(runner.MOTION_PRESETS["natural"], idle_rms=(0.0, 0.0, 0.0))

    def yaw_rms(amplitude):
        raw = np.zeros((300, 3))
        raw[:, 1] = amplitude * np.sin(t)
        d = runner.head_motion(raw, preset, np.random.default_rng(0))
        return np.sqrt((d[:, 1] ** 2).mean())

    target = preset["pose_rms"][1]
    assert yaw_rms(5.0) == pytest.approx(target, rel=0.2)  # movimento grande: reduzido ao alvo
    assert yaw_rms(1.0) == pytest.approx(target, rel=0.2)  # pequeno: ampliado até o alvo
    tiny = yaw_rms(0.1)  # quase parado: ampliado no máximo MAX_POSE_GAIN vezes
    assert tiny <= 0.1 * runner.MAX_POSE_GAIN


def test_still_preset_has_no_head_motion():
    raw = np.random.default_rng(0).normal(0, 5, size=(80, 3))
    assert not runner.head_motion(raw, runner.MOTION_PRESETS["estavel"], np.random.default_rng(0)).any()


def test_blink_schedule_is_realistic():
    r = runner.blink_schedule(25 * 30, np.random.default_rng(3))
    starts = np.flatnonzero((r[1:] > 0) & (r[:-1] == 0)) + 1
    assert 6 <= len(starts) <= 20  # ~3 s entre piscadas em 30 s
    assert r.max() <= 1.25 + 1e-9 and r[-8:].sum() == 0
    assert runner.blink_schedule(10, np.random.default_rng(0)).sum() == 0  # áudio curtíssimo


def test_face_box_is_square_around_the_face():
    lm = np.zeros((68, 2))
    lm[36:42] = [400, 500]  # olho esquerdo
    lm[42:48] = [600, 500]  # olho direito
    lm[48] = [440, 700]
    lm[54] = [560, 700]
    x0, y0, side = runner.face_box(lm, 0.1)
    assert side % 2 == 0 and side > 400
    cx, cy = x0 + side / 2, y0 + side / 2
    assert abs(cx - 500) <= 1 and 500 < cy < 700


def test_head_mask_is_soft_and_zero_at_the_border():
    m = runner.head_mask(200)
    assert m.shape == (200, 200) and m.dtype == np.float32
    assert m[100, 100] == pytest.approx(1.0)
    assert m[0].max() == 0 and m[-1].max() == 0 and m[:, 0].max() == 0 and m[:, -1].max() == 0
    assert 0 < m[100, 25] < 1  # transição suave na lateral


def test_color_match_recovers_a_global_shift():
    rng = np.random.default_rng(0)
    orig = rng.uniform(50, 200, size=(100, 100, 3)).astype(np.float32)
    gen = orig * 0.9 + 10
    alpha = runner.head_mask(100)
    gain, bias = runner.color_match(gen, orig, alpha, np.ones((100, 100), bool))
    assert np.allclose(gen * gain + bias, orig, atol=0.5)


def test_auto_batch():
    gib = 2 ** 30
    assert runner.auto_batch(3 * gib, 512, False) == 1
    assert runner.auto_batch(11 * gib, 512, False) == 4
    assert runner.auto_batch(6 * gib, 512, True) > runner.auto_batch(6 * gib, 512, False)


def test_runner_is_python38_compatible():
    import ast
    from pathlib import Path

    for name in ("sadtalker_runner.py", "echomimic_runner.py"):
        source = (Path(runner.__file__).parent / name).read_text(encoding="utf-8")
        ast.parse(source, feature_version=(3, 8))


# --------------------------------------------------------------------------- correções da revisão

@pytest.mark.parametrize("device", ["auto", "cpu", "cuda", "cuda:1"])
def test_backend_commands_are_accepted_by_the_real_runners(tmp_path, fake_sadtalker, device, monkeypatch):
    from heygenhome.engines import echomimic_runner

    monkeypatch.setenv("SADTALKER_BATCH", "2")
    monkeypatch.setenv("HEYGEN_CPU_THREADS", "4")
    backend = SadTalkerNextBackend(root=fake_sadtalker, python=sys.executable, device=device)
    for motion in (MOTION_STILL, MOTION_NATURAL, MOTION_EXPRESSIVE):
        for framing in (FRAMING_FULL, FRAMING_CROP):
            for enhancer in (False, True):
                req = request_for(tmp_path, motion=motion, framing=framing, enhancer=enhancer, size=256)
                args = runner.parse_args(backend.command(req, tmp_path / "o.mp4", tmp_path / "o.json", device)[2:])
                assert (args.motion, args.framing, args.enhancer, args.device) == (motion, framing, enhancer, device)
    echo = EchoMimicBackend(root=tmp_path, python=sys.executable, device=device)
    cmd = echo.command(request_for(tmp_path, audio_duration=1.0), tmp_path / "r.mp4", tmp_path / "r.json")
    assert echomimic_runner.parse_args(cmd[2:]).device == device


def test_render_policy_never_loops_and_respects_rejected_fp16():
    # fp16 rejeitado na validação: falta de VRAM vai do lote direto para a CPU.
    p = runner.RenderPolicy(batch=4, fp16=False, fp16_allowed=False)
    assert [p.on_oom() for _ in range(3)] == ["retry", "retry", "cpu"]
    assert (p.batch, p.fp16) == (1, False)

    # fp16 permitido: lote -> fp16 (uma vez) -> CPU.
    p = runner.RenderPolicy(batch=2, fp16=False, fp16_allowed=True)
    assert p.on_oom() == "retry" and p.batch == 1
    assert p.on_oom() == "retry" and p.fp16
    assert p.on_oom() == "cpu"

    # fp16 dá NaN e fp32 não cabe: antes alternava para sempre; agora termina em CPU.
    p = runner.RenderPolicy(batch=1, fp16=False, fp16_allowed=True)
    steps = []
    for _ in range(10):
        action = p.on_oom() if not p.fp16 else p.on_nan()
        steps.append(action)
        if action == "cpu":
            break
    assert steps == ["retry", "retry", "cpu"] and not p.fp16_allowed

    # Erro não-OOM: com fp16 tenta de novo em fp32; em fp32 desiste.
    p = runner.RenderPolicy(batch=1, fp16=True, fp16_allowed=True)
    assert p.on_error() == "retry" and not p.fp16
    assert p.on_error() == "raise"


def test_is_oom_variants():
    assert runner.is_oom(RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB"))
    assert runner.is_oom(RuntimeError("CUDA error: CUBLAS_STATUS_ALLOC_FAILED when calling cublasCreate"))
    assert runner.is_oom(RuntimeError("Unable to find a valid cuDNN algorithm to run convolution"))
    assert not runner.is_oom(RuntimeError("CUDA error: an illegal memory access was encountered"))


def test_gpu_describe_accepts_12gb_cards_reported_as_11_99_gib():
    rtx4070 = [{"name": "NVIDIA GeForce RTX 4070", "vram_gb": 12.0, "driver": "560"}]
    probe = {"torch": "2.2.2+cu121", "cuda_build": "12.1", "devices": [
        {"name": "NVIDIA GeForce RTX 4070", "vram_gb": 12.0, "total_bytes": 12878610432, "ok": True}]}
    assert gpu.describe(rtx4070, probe, min_vram_gb=12).uses_gpu
    small = {"torch": "2.2.2+cu121", "cuda_build": "12.1", "devices": [
        {"name": "RTX 3060 Ti", "vram_gb": 8.0, "total_bytes": 8589934592, "ok": True}]}
    assert not gpu.describe(rtx4070, small, min_vram_gb=12).uses_gpu


def test_gpu_describe_cuda_build_without_driver():
    report = gpu.describe([{"name": "RTX 3060", "vram_gb": 12.0, "driver": "456"}],
                          {"torch": "2.1.2+cu118", "cuda_build": "11.8", "available": False, "devices": []})
    assert not report.uses_gpu and "driver" in report.summary


@requires_ffmpeg
def test_native_abort_is_not_reported_as_missing_face(tmp_path, fake_sadtalker, photo, wav, monkeypatch):
    monkeypatch.setenv("FAKE_ABORT", "1")
    engine = TalkingHeadEngine([sadtalker(fake_sadtalker, device="cpu")])
    with pytest.raises(EngineError) as err:
        engine.animate("sadtalker_next", photo, wav, tmp_path / "job")
    assert "rosto" not in str(err.value) and "libiomp5md" in str(err.value)


def test_run_logged_keeps_utf8_characters_split_across_reads(tmp_path):
    from heygenhome.engines._process import run_logged

    script = (
        "import sys, time\n"
        "data = 'Difusão: passo 1/6\\n'.encode('utf-8')\n"
        "cut = data.index('ã'.encode('utf-8')) + 1\n"
        "sys.stdout.buffer.write(data[:cut]); sys.stdout.flush(); time.sleep(1.2)\n"
        "sys.stdout.buffer.write(data[cut:]); sys.stdout.flush()\n"
    )
    lines = []
    run_logged([sys.executable, "-c", script], cwd=tmp_path, log_path=tmp_path / "m.log",
               label="Motor", on_line=lines.append)
    assert lines == ["Difusão: passo 1/6"]


def test_progress_callback_errors_do_not_abort_the_engine(tmp_path):
    from heygenhome.engines._process import run_logged

    def broken(line):
        raise ValueError("bug no callback")

    out = run_logged([sys.executable, "-c", "print('a'); print('b')"], cwd=tmp_path,
                     log_path=tmp_path / "m.log", label="Motor", on_line=broken)
    assert "a" in out and "b" in out


def test_run_classifies_failures_by_device(monkeypatch, capsys):
    def boom(argv=None):
        raise RuntimeError("GET was unable to find an engine to execute this computation")

    monkeypatch.setattr(runner, "main", boom)
    monkeypatch.setitem(runner.STATE, "device", "cuda:0")
    assert runner.run() == runner.EXIT_DEVICE  # na GPU: o app refaz em CPU
    assert "HG_DEVICE_FAILED" in capsys.readouterr().out

    monkeypatch.setitem(runner.STATE, "device", "cpu")
    with pytest.raises(RuntimeError):  # em CPU, refazer não ajudaria
        runner.run()

    def ffmpeg_died(argv=None):
        raise runner.CompositeError("ffmpeg falhou")

    monkeypatch.setattr(runner, "main", ffmpeg_died)
    monkeypatch.setitem(runner.STATE, "device", "cuda:0")
    with pytest.raises(runner.CompositeError):  # ffmpeg não é culpa da GPU
        runner.run()


@pytest.mark.skipif(sys.platform == "win32", reason="usa um ffmpeg falso em script")
def test_compositor_surfaces_the_real_ffmpeg_error(tmp_path):
    pytest.importorskip("cv2")
    fake = tmp_path / "ffmpeg"
    fake.write_text(f"#!{sys.executable}\nimport sys\nsys.stderr.write(\"Unknown encoder 'libx264'\\n\")\nsys.exit(1)\n")
    fake.chmod(0o755)
    comp = runner.Compositor(str(fake), str(tmp_path / "o.mp4"), str(tmp_path / "ffmpeg.log"),
                             None, None, 64, "crop", None)
    comp.start()
    frames = np.zeros((1, 64, 64, 3), np.uint8)
    with pytest.raises(runner.CompositeError, match="Unknown encoder"):
        for _ in range(200):
            comp.put(frames)
        comp.finish()

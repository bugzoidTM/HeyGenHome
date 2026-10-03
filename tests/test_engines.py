import json
import subprocess
import sys

import pytest

from heygenhome.engines import (
    FRAMING_CROP,
    FRAMING_FULL,
    AnimationOptions,
    AnimationRequest,
    EchoMimicBackend,
    EngineError,
    SadTalkerNextBackend,
    TalkingHeadEngine,
    create_engine,
)
from heygenhome.engines import echomimic as echomimic_module

from .conftest import make_photo, requires_ffmpeg


def sadtalker(root):
    return SadTalkerNextBackend(root=root, python=sys.executable, device="cpu")


def request_for(tmp_path, **kwargs):
    return AnimationRequest(image=tmp_path / "f.png", audio=tmp_path / "v.wav", work_dir=tmp_path, **kwargs)


def test_registry_order_and_labels():
    labels = [(b.label, b.name) for b in create_engine().backends()]
    assert labels == [("Standard", "SadTalkerNext"), ("Experimental HD", "EchoMimic")]


def test_default_options_are_512_full_still_without_enhancer():
    assert AnimationOptions() == AnimationOptions(size=512, framing=FRAMING_FULL, still=True, enhancer=False)


def test_sadtalker_command_defaults(tmp_path, fake_sadtalker):
    cmd = sadtalker(fake_sadtalker).command(request_for(tmp_path), tmp_path / "out")
    args = " ".join(cmd)
    assert "--size 512" in args
    assert "--preprocess full" in args
    assert "--still" in cmd and "--cpu" in cmd
    assert "--enhancer" not in cmd


def test_sadtalker_command_crop_and_enhancer_only_when_chosen(tmp_path, fake_sadtalker):
    req = request_for(tmp_path, framing=FRAMING_CROP, still=False, enhancer=True, size=256)
    cmd = sadtalker(fake_sadtalker).command(req, tmp_path / "out")
    args = " ".join(cmd)
    assert "--preprocess crop" in args and "--size 256" in args and "--enhancer gfpgan" in args
    assert "--still" not in cmd


def test_sadtalker_gpu_device_drops_cpu_flag(tmp_path, fake_sadtalker):
    backend = SadTalkerNextBackend(root=fake_sadtalker, python=sys.executable, device="cuda")
    assert "--cpu" not in backend.command(request_for(tmp_path), tmp_path)


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
def test_sadtalker_full_preserves_photo_resolution(tmp_path, fake_sadtalker, wav):
    photo = make_photo(tmp_path / "retrato.png", size=(721, 1001))
    engine = TalkingHeadEngine([sadtalker(fake_sadtalker)])
    run = engine.animate("sadtalker_next", photo, wav, tmp_path / "job", audio_duration=1.6)
    assert run.image.size == (722, 1002)
    assert run.video.size == (722, 1002)
    assert run.warnings == []
    argv = (tmp_path / "job" / "sadtalker" / "argv.txt").read_text()
    assert "--preprocess full" in argv and "--still" in argv and "--size 512" in argv


@requires_ffmpeg
def test_sadtalker_enhancer_doubling_is_noted_not_warned(tmp_path, fake_sadtalker, photo, wav):
    engine = TalkingHeadEngine([sadtalker(fake_sadtalker)])
    run = engine.animate("sadtalker_next", photo, wav, tmp_path / "job", AnimationOptions(enhancer=True))
    assert run.video.size == (1200, 1600)
    assert run.warnings == []
    assert any("mesma proporção" in n for n in run.notes)


@requires_ffmpeg
def test_sadtalker_crop_only_when_requested(tmp_path, fake_sadtalker, photo, wav):
    engine = TalkingHeadEngine([sadtalker(fake_sadtalker)])
    run = engine.animate(
        "sadtalker_next", photo, wav, tmp_path / "job", AnimationOptions(framing=FRAMING_CROP)
    )
    assert run.video.size == (512, 512)


def test_echomimic_adapts_unsupported_options(tmp_path):
    backend = EchoMimicBackend(root=tmp_path, python=sys.executable)
    engine = TalkingHeadEngine([backend])
    req, notes = engine.build_request(
        backend, tmp_path / "f.png", tmp_path / "v.wav", tmp_path,
        AnimationOptions(size=256, still=True, enhancer=True),
    )
    assert (req.size, req.still, req.enhancer, req.framing) == (512, False, False, FRAMING_FULL)
    assert len(notes) == 3


def test_echomimic_command_is_cpu_and_accelerated_by_default(tmp_path, monkeypatch):
    for var in ("ECHOMIMIC_DEVICE", "ECHOMIMIC_ACCELERATED", "ECHOMIMIC_STEPS", "ECHOMIMIC_FPS"):
        monkeypatch.delenv(var, raising=False)
    backend = EchoMimicBackend(root=tmp_path, python=sys.executable)
    cmd = backend.command(request_for(tmp_path, audio_duration=2.01), tmp_path / "r.mp4", tmp_path / "r.json")
    args = " ".join(cmd)
    assert cmd[1].endswith("echomimic_runner.py")
    assert "--device cpu" in args and "--accelerated" in cmd
    assert "--frames 49" in args  # ceil(2.01 * 24)
    assert "--fp16" not in cmd
    gpu = EchoMimicBackend(root=tmp_path, python=sys.executable, device="cuda")
    assert "--fp16" in gpu.command(request_for(tmp_path), tmp_path / "r.mp4", tmp_path / "r.json")


@requires_ffmpeg
@pytest.mark.parametrize("framing,expected", [(FRAMING_FULL, (600, 800)), (FRAMING_CROP, (512, 512))])
def test_echomimic_full_pastes_face_back_into_photo(tmp_path, monkeypatch, photo, wav, framing, expected):
    root = tmp_path / "EchoMimic"
    (root / "src" / "pipelines").mkdir(parents=True)

    def fake_runner(cmd, *, cwd, log_path, label):
        out = cmd[cmd.index("--output") + 1]
        meta = cmd[cmd.index("--meta") + 1]
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi",
             "-i", "testsrc2=size=512x512:rate=24:duration=1.6", "-c:v", "libx264", out],
            check=True,
        )
        with open(meta, "w") as fh:
            json.dump({"crop_rect": [50, 100, 450, 500], "image_size": [600, 800]}, fh)
        log_path.write_text("ok")
        return "ok"

    monkeypatch.setattr(echomimic_module, "run_logged", fake_runner)
    engine = TalkingHeadEngine([EchoMimicBackend(root=root, python=sys.executable)])
    run = engine.animate(
        "echomimic", photo, wav, tmp_path / "job", AnimationOptions(framing=framing), audio_duration=1.6
    )
    assert run.video.size == expected
    assert run.warnings == []


def test_engine_failure_shows_short_log_tail(tmp_path):
    from heygenhome.engines._process import run_logged

    script = "print('x' * 5000); print('ERRO: algo deu errado'); raise SystemExit(1)"
    with pytest.raises(EngineError) as err:
        run_logged([sys.executable, "-c", script], cwd=tmp_path, log_path=tmp_path / "m.log", label="Motor")
    message = str(err.value)
    assert "ERRO: algo deu errado" in message
    assert "x" * 400 not in message  # linhas gigantes são encurtadas
    assert "memória" not in message


@pytest.mark.skipif(sys.platform == "win32", reason="SIGKILL não existe no Windows")
def test_engine_killed_by_system_mentions_memory(tmp_path):
    from heygenhome.engines._process import run_logged

    script = "import os, signal; os.kill(os.getpid(), signal.SIGKILL)"
    with pytest.raises(EngineError, match="falta de memória"):
        run_logged([sys.executable, "-c", script], cwd=tmp_path, log_path=tmp_path / "m.log", label="Motor")

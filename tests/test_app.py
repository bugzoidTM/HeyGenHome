"""Fluxo completo da interface: voz (simulada) -> motor (simulado) -> exportação."""

import sys

import numpy as np
import pytest
import soundfile as sf

pytest.importorskip("gradio")

import app  # noqa: E402
from heygenhome.engines import FRAMING_CROP, FRAMING_FULL, SadTalkerNextBackend, TalkingHeadEngine  # noqa: E402
from heygenhome.export import BACKGROUND_BLUR, FORMAT_ORIGINAL, FORMAT_VERTICAL  # noqa: E402
from heygenhome.media import probe_video  # noqa: E402

from .conftest import make_photo, requires_ffmpeg  # noqa: E402


def fake_synthesize(text, voice_label, speed, out_path):
    sr = 24000
    t = np.arange(int(sr * 1.2)) / sr
    sf.write(out_path, (0.2 * np.sin(2 * np.pi * 300 * t)).astype(np.float32), sr)
    return 1.2


@pytest.fixture
def app_env(tmp_path, monkeypatch, fake_sadtalker):
    engine = TalkingHeadEngine([SadTalkerNextBackend(root=fake_sadtalker, python=sys.executable)])
    monkeypatch.setattr(app, "ENGINE", engine)
    monkeypatch.setattr(app, "OUTPUT_DIR", tmp_path / "outputs")
    monkeypatch.setattr(app, "synthesize", fake_synthesize)
    return tmp_path


def no_progress(*args, **kwargs):
    pass


@requires_ffmpeg
@pytest.mark.parametrize(
    "framing,fmt,expected",
    [
        (FRAMING_FULL, FORMAT_ORIGINAL, (640, 480)),
        (FRAMING_FULL, FORMAT_VERTICAL, (1080, 1920)),
        (FRAMING_CROP, FORMAT_ORIGINAL, (512, 512)),
    ],
)
def test_generate_end_to_end(app_env, framing, fmt, expected):
    photo = make_photo(app_env / "foto.jpg", size=(640, 480))
    wav, mp4, report = app.generate(
        str(photo), "Olá!", app.DEFAULT_VOICE, 1.0, "sadtalker_next",
        framing, True, 512, False, fmt, BACKGROUND_BLUR, progress=no_progress,
    )
    info = probe_video(mp4)
    assert info.size == expected
    assert (info.codec, info.audio_codec) == ("h264", "aac")
    assert "✅ resolução validada" in report
    assert "Standard (SadTalkerNext)" in report

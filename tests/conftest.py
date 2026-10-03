import shutil
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

requires_ffmpeg = pytest.mark.skipif(
    not (shutil.which("ffmpeg") and shutil.which("ffprobe")),
    reason="FFmpeg/ffprobe não estão no PATH",
)


def make_photo(path: Path, size=(600, 800), mode="RGB", exif_orientation=None) -> Path:
    w, h = size
    x = np.linspace(0, 255, w, dtype=np.uint8)
    y = np.linspace(0, 255, h, dtype=np.uint8)
    rgb = np.stack(np.broadcast_arrays(x[None, :], y[:, None], np.full((h, w), 128, np.uint8)), -1)
    im = Image.fromarray(rgb.astype(np.uint8), "RGB").convert(mode)
    kwargs = {}
    if exif_orientation:
        exif = Image.Exif()
        exif[274] = exif_orientation
        kwargs["exif"] = exif
    im.save(path, **kwargs)
    return path


@pytest.fixture
def photo(tmp_path):
    return make_photo(tmp_path / "foto.jpg")


@pytest.fixture
def wav(tmp_path):
    sr, seconds = 24000, 1.6
    t = np.arange(int(sr * seconds)) / sr
    path = tmp_path / "voz.wav"
    sf.write(path, (0.2 * np.sin(2 * np.pi * 220 * t)).astype(np.float32), sr)
    return path


FAKE_SADTALKER = textwrap.dedent(
    """
    # Imita a CLI do SadTalker: mesmos argumentos, mesmo formato de saída (MPEG-4 Part 2).
    import argparse, os, subprocess, sys, time
    from PIL import Image

    p = argparse.ArgumentParser()
    for name in ("--driven_audio", "--source_image", "--result_dir", "--checkpoint_dir",
                 "--preprocess", "--enhancer"):
        p.add_argument(name)
    p.add_argument("--size", type=int)
    p.add_argument("--still", action="store_true")
    p.add_argument("--cpu", action="store_true")
    a = p.parse_args()

    w, h = Image.open(a.source_image).size
    vw, vh = (w, h) if a.preprocess == "full" else (a.size, a.size)
    if a.enhancer:  # o GFPGAN do SadTalker usa upscale=2
        vw, vh = vw * 2, vh * 2
    os.makedirs(a.result_dir, exist_ok=True)
    with open(os.path.join(a.result_dir, "argv.txt"), "w") as fh:
        fh.write(" ".join(sys.argv[1:]))
    out = os.path.join(a.result_dir, time.strftime("%Y_%m_%d_%H.%M.%S") + ".mp4")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-loop", "1", "-framerate", "25",
                    "-i", a.source_image, "-i", a.driven_audio, "-vf", f"scale={vw}:{vh}",
                    "-c:v", "mpeg4", "-c:a", "aac", "-shortest", out], check=True)
    print("The generated video is named:", out)
    """
)


@pytest.fixture
def fake_sadtalker(tmp_path):
    root = tmp_path / "SadTalker"
    (root / "checkpoints").mkdir(parents=True)
    (root / "inference.py").write_text(FAKE_SADTALKER, encoding="utf-8")
    for name in (
        "SadTalker_V0.0.2_256.safetensors",
        "SadTalker_V0.0.2_512.safetensors",
        "mapping_00109-model.pth.tar",
        "mapping_00229-model.pth.tar",
    ):
        (root / "checkpoints" / name).write_bytes(b"")
    return root

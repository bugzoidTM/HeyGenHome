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


FAKE_RUNNER = textwrap.dedent(
    """
    # Imita a CLI do sadtalker_runner.py: mesmos argumentos, mesmas linhas de progresso,
    # mesmo contrato de saída (vídeo H.264 sem áudio + meta JSON).
    import argparse, json, math, os, subprocess, sys
    import soundfile as sf
    from PIL import Image

    p = argparse.ArgumentParser()
    for name in ("--image", "--audio", "--output", "--meta", "--checkpoint_dir", "--framing",
                 "--motion", "--device", "--fp16", "--ffmpeg"):
        p.add_argument(name)
    for name in ("--size", "--seed", "--pose_style", "--batch", "--threads"):
        p.add_argument(name, type=int, default=0)
    p.add_argument("--enhancer", action="store_true")
    a = p.parse_args()

    with open(os.path.join(os.path.dirname(a.output), "argv.txt"), "a") as fh:
        fh.write(" ".join(sys.argv[1:]) + "\\n")
    if os.environ.get("FAKE_NO_FACE"):
        print("HG_ERROR no_face")
        print("ERRO: nenhum rosto detectado na foto.")
        sys.exit(13)
    if os.environ.get("FAKE_ABORT"):  # abort() nativo no Windows sai com 3
        print("OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.")
        sys.exit(3)
    if os.environ.get("FAKE_GPU_FAILS") and a.device != "cpu":
        print("HG_DEVICE_FAILED " + json.dumps({"error": "CUDA error: illegal memory access"}))
        sys.exit(14)

    on_gpu = a.device != "cpu"
    print("HG_DEVICE " + json.dumps({"device": "cuda:0" if on_gpu else "cpu", "name": "Fake GPU" if on_gpu else None}))
    for stage in ("load", "face", "audio"):
        print("HG_PROGRESS " + json.dumps({"stage": stage, "done": 0, "total": 1}))
    info = sf.info(a.audio)
    frames = max(2, math.ceil(info.frames / info.samplerate * 25))
    for done in (frames // 2, frames):
        print("HG_PROGRESS " + json.dumps({"stage": "render", "done": done, "total": frames, "sec_per_frame": 0.01}))
    w, h = Image.open(a.image).size
    vw, vh = (w, h) if a.framing == "full" else (a.size, a.size)
    subprocess.run([a.ffmpeg, "-y", "-loglevel", "error", "-loop", "1", "-framerate", "25", "-i", a.image,
                    "-frames:v", str(frames), "-vf", "scale=%d:%d" % (vw, vh), "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", a.output], check=True)
    with open(a.meta, "w") as fh:
        json.dump({"frames": frames, "device": "cuda:0" if on_gpu else "cpu", "device_name": "Fake GPU" if on_gpu else "CPU",
                   "fp16": on_gpu, "batch": 2 if on_gpu else 1, "threads": 4, "sec_per_frame": 0.01,
                   "motion": a.motion}, fh)
    print("SadTalkerNext OK: " + a.output)
    """
)


@pytest.fixture
def fake_sadtalker(tmp_path):
    """Pasta com a estrutura que o motor confere + o runner falso."""
    root = tmp_path / "SadTalker"
    (root / "checkpoints").mkdir(parents=True)
    (root / "src" / "facerender").mkdir(parents=True)
    (root / "fake_runner.py").write_text(FAKE_RUNNER, encoding="utf-8")
    for name in (
        "SadTalker_V0.0.2_256.safetensors",
        "SadTalker_V0.0.2_512.safetensors",
        "mapping_00109-model.pth.tar",
    ):
        (root / "checkpoints" / name).write_bytes(b"")
    return root

import subprocess

import pytest
from PIL import Image

from heygenhome.export import (
    BACKGROUND_BLACK,
    BACKGROUND_BLUR,
    FORMAT_ORIGINAL,
    FORMAT_VERTICAL,
    VERTICAL_SIZE,
    ExportError,
    export_video,
    validate_export,
)
from heygenhome.media import paste_back, prepare_image, probe_video

from .conftest import make_photo, requires_ffmpeg


def make_video(path, size, seconds=1.5, codec="mpeg4"):
    w, h = size
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi",
         "-i", f"testsrc2=size={w}x{h}:rate=25:duration={seconds}", "-c:v", codec, str(path)],
        check=True,
    )
    return path


def test_prepare_image_keeps_resolution_and_converts_to_png(tmp_path):
    src = make_photo(tmp_path / "foto.webp", size=(640, 960))
    prepared = prepare_image(src, tmp_path / "out.png")
    assert prepared.size == prepared.original_size == (640, 960)
    assert prepared.notes == []
    with Image.open(prepared.path) as im:
        assert (im.format, im.mode, im.size) == ("PNG", "RGB", (640, 960))


def test_prepare_image_pads_odd_sizes_instead_of_cropping(tmp_path):
    prepared = prepare_image(make_photo(tmp_path / "f.png", size=(1001, 1499)), tmp_path / "o.png")
    assert prepared.original_size == (1001, 1499)
    assert prepared.size == (1002, 1500)
    with Image.open(prepared.path) as im:
        # a borda nova repete a última coluna/linha da foto
        assert im.getpixel((1001, 10)) == im.getpixel((1000, 10))
        assert im.getpixel((10, 1499)) == im.getpixel((10, 1498))


def test_prepare_image_applies_exif_rotation(tmp_path):
    src = make_photo(tmp_path / "celular.jpg", size=(800, 600), exif_orientation=6)
    assert prepare_image(src, tmp_path / "o.png").size == (600, 800)


def test_prepare_image_fills_transparency_with_white(tmp_path):
    src = tmp_path / "alpha.png"
    Image.new("RGBA", (40, 40), (0, 0, 0, 0)).save(src)
    prepared = prepare_image(src, tmp_path / "o.png")
    with Image.open(prepared.path) as im:
        assert im.getpixel((5, 5)) == (255, 255, 255)


def test_prepare_image_max_side_scales_proportionally(tmp_path):
    src = make_photo(tmp_path / "grande.jpg", size=(4000, 3000))
    prepared = prepare_image(src, tmp_path / "o.png", max_side=1920)
    assert prepared.size == (1920, 1440)
    assert any("HEYGEN_MAX_SIDE" in n for n in prepared.notes)


@requires_ffmpeg
@pytest.mark.parametrize(
    "fmt,source,expected",
    [
        (FORMAT_ORIGINAL, (600, 800), (600, 800)),
        (FORMAT_ORIGINAL, (512, 530), (512, 530)),
        (FORMAT_VERTICAL, (600, 800), VERTICAL_SIZE),
        (FORMAT_VERTICAL, (512, 512), VERTICAL_SIZE),
        (FORMAT_VERTICAL, (1920, 1080), VERTICAL_SIZE),
    ],
)
@pytest.mark.parametrize("background", [BACKGROUND_BLUR, BACKGROUND_BLACK])
def test_export_formats_are_h264_with_exact_resolution(tmp_path, wav, fmt, source, expected, background):
    video = make_video(tmp_path / "motor.mp4", source)
    result = export_video(video, tmp_path / "final.mp4", fmt=fmt, audio=wav, background=background)
    assert result.info.size == result.target == expected
    assert (result.info.codec, result.info.pix_fmt, result.info.audio_codec) == ("h264", "yuv420p", "aac")


@requires_ffmpeg
def test_original_uses_photo_size_when_engine_returns_other_size(tmp_path, wav):
    # ex.: SadTalker com GFPGAN devolve o dobro da resolução da foto
    video = make_video(tmp_path / "motor.mp4", (1200, 1600))
    result = export_video(video, tmp_path / "f.mp4", audio=wav, original_size=(600, 800))
    assert result.info.size == (600, 800)


@requires_ffmpeg
def test_vertical_export_does_not_crop(tmp_path, wav):
    # vídeo branco em 1:1 -> 9:16 com barras pretas: a faixa central continua toda branca
    src = tmp_path / "branco.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i",
         "color=white:size=512x512:rate=25:duration=1", "-c:v", "mpeg4", str(src)],
        check=True,
    )
    out = export_video(src, tmp_path / "v.mp4", fmt=FORMAT_VERTICAL, audio=wav,
                       background=BACKGROUND_BLACK).path
    frame = tmp_path / "frame.png"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(out), "-frames:v", "1", str(frame)],
                   check=True)
    with Image.open(frame) as im:
        gray = im.convert("L")
        # 512 -> 1080 de largura: conteúdo ocupa y de 420 a 1500
        assert gray.getpixel((5, 960)) > 240 and gray.getpixel((1074, 960)) > 240
        assert gray.getpixel((540, 430)) > 240 and gray.getpixel((540, 1490)) > 240
        assert gray.getpixel((540, 100)) < 20 and gray.getpixel((540, 1820)) < 20


@requires_ffmpeg
def test_validate_export_rejects_wrong_resolution(tmp_path):
    info = probe_video(make_video(tmp_path / "x.mp4", (640, 480), codec="libx264"))
    with pytest.raises(ExportError, match="resolução 640x480"):
        validate_export(info, (1080, 1920), expect_audio=False)


@requires_ffmpeg
def test_paste_back_returns_photo_resolution(tmp_path):
    base = prepare_image(make_photo(tmp_path / "f.png", size=(900, 1200)), tmp_path / "b.png")
    face = make_video(tmp_path / "rosto.mp4", (512, 512), seconds=1.2)
    out = paste_back(base.path, face, (200, 150, 700, 650), tmp_path / "full.mp4")
    info = probe_video(out)
    assert info.size == (900, 1200)
    assert abs(info.duration - 1.2) < 0.1

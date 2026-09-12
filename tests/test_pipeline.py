from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw

from covermorph import __version__
from covermorph.pipeline import PipelineOptions, _outpaint_or_fallback, process_image_file
from covermorph.presets import PRESETS


class BaseFakeAI:
    def lama_available(self) -> bool:
        return False

    def rembg_available(self) -> bool:
        return False

    def sdxl_available(self) -> bool:
        return False

    def upscale(self, img: Image.Image, scale: int = 2, model: str = "realesrgan-x4plus"):
        raise RuntimeError("Real-ESRGAN executable unavailable")


class FailingSdxlAI(BaseFakeAI):
    def sdxl_available(self) -> bool:
        return True

    def outpaint(self, *args, **kwargs):
        raise RuntimeError("model download failed")


class SuccessfulSdxlAI(BaseFakeAI):
    def rembg_available(self) -> bool:
        return True

    def sdxl_available(self) -> bool:
        return True

    def person_mask(self, img: Image.Image):
        return Image.new("L", img.size, 255), "fake person mask"

    def outpaint(self, canvas: Image.Image, *args, **kwargs):
        return Image.new("RGB", canvas.size, (255, 0, 0)), "Fake SDXL"


def make_input(path: Path, size: tuple[int, int] = (360, 360)) -> Path:
    img = Image.new("RGB", size, (150, 120, 90))
    draw = ImageDraw.Draw(img)
    draw.ellipse((120, 70, 240, 250), fill=(220, 185, 150))
    draw.rectangle((80, 260, 280, 320), fill=(245, 245, 245))
    draw.text((125, 280), "TITLE", fill=(10, 10, 10))
    img.save(path, "JPEG")
    return path


def test_sdxl_error_falls_back_per_output(tmp_path: Path) -> None:
    source = make_input(tmp_path / "cover.jpg")
    options = PipelineOptions(
        output_dir=tmp_path / "out",
        auto_remove_text=False,
        prefer_esrgan=False,
        out_square=False,
        out_thumb=True,
        out_shorts=True,
        use_sdxl=True,
    )

    result = process_image_file(source, options, FailingSdxlAI(), tmp_path)

    assert result.status == "partial"
    assert result.success_outputs == 2
    assert result.metadata["sdxl_fallback"] is True
    assert result.metadata["thumbnail_16x9_engine"] == "Fallback blur canvas"
    assert result.metadata["shorts_9x16_engine"] == "Fallback blur canvas"
    assert Path(result.metadata["output_files"]["thumbnail_16x9"]).is_file()
    assert Path(result.metadata["output_files"]["shorts_9x16"]).is_file()


def test_realesrgan_missing_falls_back_to_local_sharpen(tmp_path: Path) -> None:
    source = make_input(tmp_path / "cover.jpg")
    options = PipelineOptions(
        output_dir=tmp_path / "out",
        auto_remove_text=False,
        prefer_esrgan=True,
        out_square=True,
        out_thumb=False,
        out_shorts=False,
    )

    result = process_image_file(source, options, BaseFakeAI(), tmp_path)

    assert result.status == "partial"
    assert result.success_outputs == 1
    assert result.metadata["upscale_engine"] == "Local sharpen (Real-ESRGAN fallback)"
    assert any(error["category"] == "realesrgan_failed" for error in result.errors)


def test_corrupt_image_is_reported_and_json_is_written(tmp_path: Path) -> None:
    source = tmp_path / "broken.jpg"
    source.write_bytes(b"not an image")
    options = PipelineOptions(output_dir=tmp_path / "out", auto_remove_text=False)

    result = process_image_file(source, options, BaseFakeAI(), tmp_path)

    assert result.status == "failed"
    assert result.errors[0]["category"] == "corrupt_image"
    job_json = tmp_path / "out" / "broken" / "covermorph_job.json"
    assert job_json.is_file()
    data = json.loads(job_json.read_text(encoding="utf-8"))
    assert data["status"] == "failed"


def test_job_json_contains_v051_metadata(tmp_path: Path) -> None:
    source = make_input(tmp_path / "cover.jpg")
    options = PipelineOptions(
        output_dir=tmp_path / "out",
        auto_remove_text=False,
        prefer_esrgan=False,
        out_square=True,
        out_thumb=True,
        out_shorts=True,
    )

    result = process_image_file(source, options, BaseFakeAI(), tmp_path)
    job_json = tmp_path / "out" / "cover" / "covermorph_job.json"
    data = json.loads(job_json.read_text(encoding="utf-8"))

    assert result.status == "success"
    assert data["program_version"] == __version__ == "0.5.1"
    assert data["original_filename"] == "cover.jpg"
    assert data["ocr_languages"] == ["en"]
    assert data["text_removal_engine"] == "No text mask"
    assert data["upscale_engine"] == "Skipped"
    assert data["thumbnail_16x9_engine"] == "Blur Canvas"
    assert data["shorts_9x16_engine"] == "Blur Canvas"
    assert data["status"] == "success"
    assert set(data["output_files"]) == {"square_1x1", "thumbnail_16x9", "shorts_9x16"}


def test_thumbnail_failure_does_not_stop_shorts(tmp_path: Path, monkeypatch) -> None:
    source = make_input(tmp_path / "cover.jpg")
    options = PipelineOptions(
        output_dir=tmp_path / "out",
        auto_remove_text=False,
        prefer_esrgan=False,
        out_square=False,
        out_thumb=True,
        out_shorts=True,
    )

    def flaky_outpaint(*args, **kwargs):
        kind = args[5]
        size = args[6]
        if kind == "thumbnail":
            raise RuntimeError("thumbnail failure")
        return Image.new("RGB", size, (30, 40, 50)), "Fake shorts", [], "Disabled"

    monkeypatch.setattr("covermorph.pipeline._outpaint_or_fallback", flaky_outpaint)

    result = process_image_file(source, options, BaseFakeAI(), tmp_path)

    assert result.status == "partial"
    assert result.success_outputs == 1
    assert result.failed_outputs == 1
    assert result.metadata["shorts_9x16_engine"] == "Fake shorts"
    assert Path(result.metadata["output_files"]["shorts_9x16"]).is_file()


def test_sdxl_success_restores_protected_pixels(tmp_path: Path) -> None:
    img = Image.new("RGB", (120, 120), (0, 180, 70))
    preset = PRESETS["OldPopLounge"]
    options = PipelineOptions(
        output_dir=tmp_path / "out",
        use_sdxl=True,
        protect_person=True,
    )

    out, engine, errors, person_engine = _outpaint_or_fallback(
        tmp_path,
        SuccessfulSdxlAI(),
        img,
        preset,
        options,
        "thumbnail",
        (192, 108),
        "right",
    )

    assert out.size == (192, 108)
    assert engine == "Fake SDXL + fake person mask"
    assert person_engine == "fake person mask"
    assert errors == []
    assert out.getextrema() != ((255, 255), (0, 0), (0, 0))

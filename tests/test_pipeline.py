from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from covermorph import __version__
from covermorph.pipeline import (
    ImageJob,
    PipelineOptions,
    _outpaint_or_fallback,
    count_selected_outputs_for_jobs,
    output_count_by_kind,
    process_image_file,
    process_image_jobs,
    progress_ratio,
)
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


def make_input(path: Path, size: tuple[int, int] = (360, 360), color=(150, 120, 90)) -> Path:
    img = Image.new("RGB", size, color)
    draw = ImageDraw.Draw(img)
    draw.ellipse((120, 70, 240, 250), fill=(220, 185, 150))
    draw.rectangle((80, 260, 280, 320), fill=(245, 245, 245))
    draw.text((125, 280), "TITLE", fill=(10, 10, 10))
    img.save(path, "JPEG")
    return path


def local_options(tmp_path: Path, **overrides) -> PipelineOptions:
    defaults = {
        "output_dir": tmp_path / "out",
        "auto_remove_text": False,
        "prefer_esrgan": False,
        "extension_mode": "natural",
        "use_sdxl": False,
    }
    defaults.update(overrides)
    return PipelineOptions(**defaults)


def output_names(result) -> set[str]:
    return {Path(path).name for path in result.metadata.get("output_files", {}).values()}


def assert_no_blank_frame(path: Path) -> None:
    with Image.open(path) as opened:
        arr = np.asarray(opened.convert("RGB"), dtype=np.int16)
    strips = [
        arr[:12, :, :],
        arr[-12:, :, :],
        arr[:, :12, :],
        arr[:, -12:, :],
    ]
    for strip in strips:
        mean = strip.reshape(-1, 3).mean(axis=0)
        assert not np.all(mean < 5)
        assert not np.all(mean > 250)


def test_sdxl_error_fails_per_output_without_saving_a_local_replacement(tmp_path: Path) -> None:
    source = make_input(tmp_path / "cover.jpg")
    options = PipelineOptions(
        output_dir=tmp_path / "out",
        auto_remove_text=False,
        prefer_esrgan=False,
        out_square=False,
        out_thumb=True,
        out_shorts=True,
        extension_mode="ai_natural",
        use_sdxl=True,
    )

    result = process_image_file(source, options, FailingSdxlAI(), tmp_path)

    assert result.status == "failed"
    assert result.success_outputs == 0
    assert result.failed_outputs == 2
    assert result.metadata["sdxl_fallback"] is False
    assert "not saved" in result.metadata["thumbnail_16x9_engine"]
    assert "not saved" in result.metadata["shorts_9x16_engine"]
    assert all(error["category"] == "sdxl_failed" for error in result.metadata["sdxl_failures"])
    assert not result.metadata["output_files"]


def test_sdxl_failure_does_not_try_an_implicit_natural_or_blur_fallback(tmp_path: Path, monkeypatch) -> None:
    source = make_input(tmp_path / "cover.jpg")
    options = PipelineOptions(
        output_dir=tmp_path / "out",
        auto_remove_text=False,
        prefer_esrgan=False,
        out_square=False,
        out_thumb=True,
        out_shorts=False,
        extension_mode="ai_natural",
        use_sdxl=True,
    )

    def fail_natural(*args, **kwargs):
        raise AssertionError("implicit natural fallback must not run")

    monkeypatch.setattr("covermorph.pipeline.natural_background_extend", fail_natural)

    result = process_image_file(source, options, FailingSdxlAI(), tmp_path)

    assert result.status == "failed"
    assert result.success_outputs == 0
    assert result.failed_outputs == 1
    assert not result.metadata["output_files"]


def test_sdxl_unavailable_marks_output_failed_and_does_not_save(tmp_path: Path) -> None:
    source = make_input(tmp_path / "unavailable.jpg")
    options = PipelineOptions(
        output_dir=tmp_path / "out",
        auto_remove_text=False,
        prefer_esrgan=False,
        out_square=False,
        out_thumb=False,
        out_shorts=True,
        extension_mode="ai_natural",
        use_sdxl=True,
    )

    result = process_image_file(source, options, BaseFakeAI(), tmp_path)

    assert result.status == "failed"
    assert result.success_outputs == 0
    assert result.metadata["outputs"]["shorts_9x16"]["status"] == "failed"
    assert result.metadata["shorts_9x16_engine"].endswith("not saved)")
    assert result.metadata["sdxl_failures"][0]["category"] == "sdxl_unavailable"


def test_realesrgan_missing_falls_back_to_local_sharpen(tmp_path: Path) -> None:
    source = make_input(tmp_path / "cover.jpg")
    options = local_options(
        tmp_path,
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
    options = local_options(tmp_path)

    result = process_image_file(source, options, BaseFakeAI(), tmp_path)

    assert result.status == "failed"
    assert result.failed_outputs == 3
    assert result.errors[0]["category"] == "corrupt_image"
    job_json = tmp_path / "out" / "broken" / "covermorph_job.json"
    assert job_json.is_file()
    data = json.loads(job_json.read_text(encoding="utf-8"))
    assert data["status"] == "failed"


def test_job_json_contains_v053_metadata(tmp_path: Path) -> None:
    source = make_input(tmp_path / "cover.jpg")
    options = local_options(
        tmp_path,
        out_square=True,
        out_thumb=True,
        out_shorts=True,
    )

    result = process_image_file(source, options, BaseFakeAI(), tmp_path)
    job_json = tmp_path / "out" / "cover" / "covermorph_job.json"
    data = json.loads(job_json.read_text(encoding="utf-8"))

    assert result.status == "success"
    assert data["program_version"] == __version__ == "0.5.4"
    assert data["original_filename"] == "cover.jpg"
    assert data["ocr_languages"] == ["en"]
    assert data["text_removal_engine"] == "No text mask"
    assert data["upscale_engine"] == "Skipped"
    assert data["thumbnail_16x9_engine"] == "Natural edge extension"
    assert data["shorts_9x16_engine"] == "Natural edge extension"
    assert data["extension_mode"] == "natural"
    assert data["core_protection"] is True
    assert data["status"] == "success"
    assert set(data["output_files"]) == {"square_1x1", "thumbnail_16x9", "shorts_9x16"}


def test_thumbnail_only_generates_only_thumbnail(tmp_path: Path) -> None:
    source = make_input(tmp_path / "cover.jpg")
    options = local_options(
        tmp_path,
        out_square=False,
        out_thumb=True,
        out_shorts=False,
    )

    result = process_image_file(source, options, BaseFakeAI(), tmp_path)

    assert result.success_outputs == 1
    assert set(result.metadata["output_files"]) == {"thumbnail_16x9"}
    assert "_thumbnail_16x9_1920x1080" in next(iter(output_names(result)))


def test_shorts_only_generates_only_shorts(tmp_path: Path) -> None:
    source = make_input(tmp_path / "cover.jpg")
    options = local_options(
        tmp_path,
        out_square=False,
        out_thumb=False,
        out_shorts=True,
    )

    result = process_image_file(source, options, BaseFakeAI(), tmp_path)

    assert result.success_outputs == 1
    assert set(result.metadata["output_files"]) == {"shorts_9x16"}


def test_thumbnail_and_shorts_generate_two_files_only(tmp_path: Path) -> None:
    source = make_input(tmp_path / "cover.jpg")
    options = local_options(
        tmp_path,
        out_square=False,
        out_thumb=True,
        out_shorts=True,
    )

    result = process_image_file(source, options, BaseFakeAI(), tmp_path)

    assert result.success_outputs == 2
    assert set(result.metadata["output_files"]) == {"thumbnail_16x9", "shorts_9x16"}


def test_all_three_output_formats_generate_three_files(tmp_path: Path) -> None:
    source = make_input(tmp_path / "cover.jpg")
    options = local_options(
        tmp_path,
        out_square=True,
        out_thumb=True,
        out_shorts=True,
    )

    result = process_image_file(source, options, BaseFakeAI(), tmp_path)

    assert result.success_outputs == 3
    assert set(result.metadata["output_files"]) == {"square_1x1", "thumbnail_16x9", "shorts_9x16"}


def test_thumbnail_resolution_1280x720(tmp_path: Path) -> None:
    source = make_input(tmp_path / "cover.jpg")
    options = local_options(
        tmp_path,
        out_square=False,
        out_thumb=True,
        out_shorts=False,
        thumbnail_resolution="1280x720",
    )

    result = process_image_file(source, options, BaseFakeAI(), tmp_path)
    path = Path(result.metadata["output_files"]["thumbnail_16x9"])

    assert path.name.endswith("_thumbnail_16x9_1280x720.jpg")
    with Image.open(path) as reopened:
        assert reopened.size == (1280, 720)


def test_duplicate_new_number_policy(tmp_path: Path) -> None:
    source = make_input(tmp_path / "cover.jpg")
    output_dir = tmp_path / "out"
    existing = output_dir / "cover" / "cover_thumbnail_16x9_1920x1080.jpg"
    existing.parent.mkdir(parents=True)
    existing.write_bytes(b"existing")
    options = local_options(
        tmp_path,
        output_dir=output_dir,
        out_square=False,
        out_thumb=True,
        out_shorts=False,
        duplicate_policy="new_number",
    )

    result = process_image_file(source, options, BaseFakeAI(), tmp_path)
    path = Path(result.metadata["output_files"]["thumbnail_16x9"])

    assert path.name == "cover_thumbnail_16x9_1920x1080_02.jpg"
    assert existing.read_bytes() == b"existing"


def test_thumbnail_failure_does_not_stop_shorts(tmp_path: Path, monkeypatch) -> None:
    source = make_input(tmp_path / "cover.jpg")
    options = local_options(
        tmp_path,
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
        extension_mode="ai_natural",
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


def test_single_image_job_selection_count(tmp_path: Path) -> None:
    source = make_input(tmp_path / "cover.jpg")
    jobs = [ImageJob(source=source, out_square=False, out_thumb=True, out_shorts=False)]

    assert count_selected_outputs_for_jobs(jobs) == 1
    assert output_count_by_kind(jobs) == {"square_1x1": 0, "thumbnail_16x9": 1, "shorts_9x16": 0}


def test_five_image_selection_counts(tmp_path: Path) -> None:
    sources = [make_input(tmp_path / f"cover{i}.jpg") for i in range(5)]
    jobs = [ImageJob(source=source, out_square=False, out_thumb=True, out_shorts=True) for source in sources]

    assert len(jobs) == 5
    assert count_selected_outputs_for_jobs(jobs) == 10
    assert output_count_by_kind(jobs) == {"square_1x1": 0, "thumbnail_16x9": 5, "shorts_9x16": 5}


def test_five_images_all_thumbnail(tmp_path: Path) -> None:
    sources = [make_input(tmp_path / f"thumb{i}.jpg") for i in range(5)]
    jobs = [ImageJob(source=source, extension_mode="natural", out_square=False, out_thumb=True, out_shorts=False) for source in sources]

    results = process_image_jobs(jobs, local_options(tmp_path), tmp_path, BaseFakeAI())

    assert sum(result.success_outputs for result in results) == 5
    assert all(set(result.metadata["output_files"]) == {"thumbnail_16x9"} for result in results)


def test_five_images_all_shorts(tmp_path: Path) -> None:
    sources = [make_input(tmp_path / f"shorts{i}.jpg") for i in range(5)]
    jobs = [ImageJob(source=source, extension_mode="natural", out_square=False, out_thumb=False, out_shorts=True) for source in sources]

    results = process_image_jobs(jobs, local_options(tmp_path), tmp_path, BaseFakeAI())

    assert sum(result.success_outputs for result in results) == 5
    assert all(set(result.metadata["output_files"]) == {"shorts_9x16"} for result in results)


def test_five_images_thumbnail_and_shorts(tmp_path: Path) -> None:
    sources = [make_input(tmp_path / f"both{i}.jpg") for i in range(5)]
    jobs = [ImageJob(source=source, extension_mode="natural", out_square=False, out_thumb=True, out_shorts=True) for source in sources]

    results = process_image_jobs(jobs, local_options(tmp_path), tmp_path, BaseFakeAI())

    assert sum(result.success_outputs for result in results) == 10
    assert all(set(result.metadata["output_files"]) == {"thumbnail_16x9", "shorts_9x16"} for result in results)


def test_per_image_different_output_formats(tmp_path: Path) -> None:
    sources = [make_input(tmp_path / f"mix{i}.jpg") for i in range(5)]
    jobs = [
        ImageJob(source=sources[0], extension_mode="natural", out_square=False, out_thumb=True, out_shorts=True),
        ImageJob(source=sources[1], extension_mode="natural", out_square=False, out_thumb=False, out_shorts=True),
        ImageJob(source=sources[2], extension_mode="natural", out_square=False, out_thumb=True, out_shorts=False),
        ImageJob(source=sources[3], extension_mode="natural", out_square=True, out_thumb=False, out_shorts=False),
        ImageJob(source=sources[4], extension_mode="natural", out_square=True, out_thumb=True, out_shorts=True),
    ]

    results = process_image_jobs(jobs, local_options(tmp_path), tmp_path, BaseFakeAI())

    assert [set(result.metadata["output_files"]) for result in results] == [
        {"thumbnail_16x9", "shorts_9x16"},
        {"shorts_9x16"},
        {"thumbnail_16x9"},
        {"square_1x1"},
        {"square_1x1", "thumbnail_16x9", "shorts_9x16"},
    ]


def test_per_image_manual_masks_are_isolated(tmp_path: Path) -> None:
    source1 = make_input(tmp_path / "mask1.jpg")
    source2 = make_input(tmp_path / "mask2.jpg")
    jobs = [
        ImageJob(source=source1, out_square=True, out_thumb=False, out_shorts=False, manual_boxes=((1, 1, 20, 20),)),
        ImageJob(
            source=source2,
            out_square=True,
            out_thumb=False,
            out_shorts=False,
            manual_boxes=((2, 2, 20, 20), (30, 30, 60, 60)),
        ),
    ]

    process_image_jobs(jobs, local_options(tmp_path), tmp_path, BaseFakeAI())

    data1 = json.loads((tmp_path / "out" / "mask1" / "covermorph_job.json").read_text(encoding="utf-8"))
    data2 = json.loads((tmp_path / "out" / "mask2" / "covermorph_job.json").read_text(encoding="utf-8"))
    assert data1["text_boxes_total"] == 1
    assert data2["text_boxes_total"] == 2


def test_per_image_ocr_and_manual_masks_are_combined(tmp_path: Path) -> None:
    source = make_input(tmp_path / "ocr.jpg")
    job = ImageJob(
        source=source,
        out_square=True,
        out_thumb=False,
        out_shorts=False,
        manual_boxes=((1, 1, 20, 20),),
        ocr_boxes=((30, 30, 60, 60),),
    )

    process_image_jobs([job], local_options(tmp_path), tmp_path, BaseFakeAI())

    data = json.loads((tmp_path / "out" / "ocr" / "covermorph_job.json").read_text(encoding="utf-8"))
    assert data["text_boxes_total"] == 2


def test_per_image_presets_are_isolated(tmp_path: Path) -> None:
    source1 = make_input(tmp_path / "preset1.jpg")
    source2 = make_input(tmp_path / "preset2.jpg")
    jobs = [
        ImageJob(source=source1, out_square=False, out_thumb=True, out_shorts=False, preset_name="OldPopLounge"),
        ImageJob(source=source2, out_square=False, out_thumb=True, out_shorts=False, preset_name="Generic Playlist"),
    ]

    process_image_jobs(jobs, local_options(tmp_path), tmp_path, BaseFakeAI())

    data1 = json.loads((tmp_path / "out" / "preset1" / "covermorph_job.json").read_text(encoding="utf-8"))
    data2 = json.loads((tmp_path / "out" / "preset2" / "covermorph_job.json").read_text(encoding="utf-8"))
    assert data1["preset"]["name"] == "OldPopLounge"
    assert data2["preset"]["name"] == "Generic Playlist"


def test_one_image_failure_does_not_stop_remaining_images(tmp_path: Path) -> None:
    broken = tmp_path / "broken.jpg"
    broken.write_bytes(b"not an image")
    good = make_input(tmp_path / "good.jpg")
    jobs = [
        ImageJob(source=broken, out_square=False, out_thumb=True, out_shorts=False, extension_mode="natural"),
        ImageJob(source=good, out_square=False, out_thumb=True, out_shorts=False, extension_mode="natural"),
    ]

    results = process_image_jobs(jobs, local_options(tmp_path), tmp_path, BaseFakeAI())

    assert [result.status for result in results] == ["failed", "success"]
    assert sum(result.success_outputs for result in results) == 1
    assert sum(result.failed_outputs for result in results) == 1


def test_planned_output_count_and_progress_ratio(tmp_path: Path) -> None:
    sources = [make_input(tmp_path / f"plan{i}.jpg") for i in range(5)]
    jobs = [
        ImageJob(source=sources[0], out_square=True, out_thumb=False, out_shorts=False),
        ImageJob(source=sources[1], out_square=False, out_thumb=True, out_shorts=True),
        ImageJob(source=sources[2], out_square=False, out_thumb=True, out_shorts=True),
        ImageJob(source=sources[3], out_square=False, out_thumb=True, out_shorts=True),
        ImageJob(source=sources[4], out_square=False, out_thumb=True, out_shorts=True),
    ]

    assert count_selected_outputs_for_jobs(jobs) == 9
    assert progress_ratio(3, 10) == 0.3
    assert progress_ratio(99, 10) == 1.0
    assert progress_ratio(1, 0) == 0.0


def test_progress_counts_failed_outputs(tmp_path: Path) -> None:
    broken = tmp_path / "broken.jpg"
    broken.write_bytes(b"not an image")
    good = make_input(tmp_path / "good.jpg")
    jobs = [
        ImageJob(source=broken, out_square=False, out_thumb=True, out_shorts=True),
        ImageJob(source=good, out_square=False, out_thumb=True, out_shorts=False),
    ]
    events: list[dict] = []

    process_image_jobs(
        jobs,
        local_options(tmp_path),
        tmp_path,
        BaseFakeAI(),
        progress_callback=events.append,
    )

    done_events = [event for event in events if event.get("type") == "batch_done"]
    assert done_events[-1]["completed_outputs"] == 3
    assert done_events[-1]["total_outputs"] == 3


def test_multi_image_duplicate_file_names_get_new_numbers(tmp_path: Path) -> None:
    source = make_input(tmp_path / "same.jpg")
    jobs = [
        ImageJob(source=source, extension_mode="natural", out_square=False, out_thumb=True, out_shorts=False),
        ImageJob(source=source, extension_mode="natural", out_square=False, out_thumb=True, out_shorts=False),
    ]

    results = process_image_jobs(
        jobs,
        local_options(tmp_path, duplicate_policy="new_number"),
        tmp_path,
        BaseFakeAI(),
    )

    names = [Path(result.metadata["output_files"]["thumbnail_16x9"]).name for result in results]
    assert names == ["same_thumbnail_16x9_1920x1080.jpg", "same_thumbnail_16x9_1920x1080_02.jpg"]


def test_korean_and_space_file_name_outputs(tmp_path: Path) -> None:
    source = make_input(tmp_path / "한글 파일 1.jpg")
    job = ImageJob(source=source, out_square=False, out_thumb=True, out_shorts=False, extension_mode="natural")

    result = process_image_jobs([job], local_options(tmp_path), tmp_path, BaseFakeAI())[0]

    assert result.status == "success"
    assert (tmp_path / "out" / "한글 파일 1" / "covermorph_job.json").is_file()
    assert Path(result.metadata["output_files"]["thumbnail_16x9"]).is_file()


def test_full_frame_thumbnail_has_no_blank_margins(tmp_path: Path) -> None:
    source = make_input(tmp_path / "wide.jpg")
    options = local_options(tmp_path, out_square=False, out_thumb=True, out_shorts=False)

    result = process_image_file(source, options, BaseFakeAI(), tmp_path)
    path = Path(result.metadata["output_files"]["thumbnail_16x9"])

    with Image.open(path) as reopened:
        assert reopened.size == (1920, 1080)
    assert_no_blank_frame(path)


def test_full_frame_shorts_has_no_blank_margins(tmp_path: Path) -> None:
    source = make_input(tmp_path / "tall.jpg")
    options = local_options(tmp_path, out_square=False, out_thumb=False, out_shorts=True)

    result = process_image_file(source, options, BaseFakeAI(), tmp_path)
    path = Path(result.metadata["output_files"]["shorts_9x16"])

    with Image.open(path) as reopened:
        assert reopened.size == (1080, 1920)
    assert_no_blank_frame(path)

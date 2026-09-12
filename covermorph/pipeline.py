from __future__ import annotations

import json
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from threading import Event
from typing import Any

from PIL import Image, ImageOps, UnidentifiedImageError

from . import __version__
from .ai_plugins import AIBackends
from .logger import write_exception, write_log
from .presets import PRESETS, ChannelPreset, preset_to_dict
from .processor import (
    Rect,
    build_full_frame_outpaint_canvas,
    detect_text_boxes_multilang,
    inpaint_text_opencv,
    make_square,
    mask_pil_from_boxes,
    mild_enhance,
    natural_background_extend,  # noqa: F401 - kept as a testable fallback seam
    render_full_frame_format,
    restore_protected_pixels,
    save_jpg,
)
from .settings import resolve_duplicate_path, thumbnail_size

DEFAULT_OUTPAINT_PROMPT = (
    "natural photographic continuation of the existing background, "
    "seamless lighting, realistic detail, no text"
)
NEGATIVE_OUTPAINT_PROMPT = (
    "text, letters, logo, watermark, duplicate person, extra limbs, distorted face, oversaturated"
)

ProgressCallback = Callable[[dict[str, Any]], None]


class PipelineCancelled(RuntimeError):
    pass


@dataclass(slots=True)
class PipelineOptions:
    output_dir: Path
    preset_name: str = "OldPopLounge"
    ocr_languages: tuple[str, ...] = ("en",)
    manual_boxes: tuple[Rect, ...] = ()
    auto_remove_text: bool = True
    prefer_lama: bool = True
    prefer_esrgan: bool = True
    enhance: bool = True
    out_square: bool = True
    out_thumb: bool = True
    out_shorts: bool = True
    thumbnail_resolution: str = "1920x1080"
    duplicate_policy: str = "new_number"
    extension_mode: str = "ai_natural"
    protect_core: bool = True
    subject_offset_x: float = 0.0
    subject_offset_y: float = 0.0
    subject_scale: float = 1.0
    use_sdxl: bool = True
    protect_person: bool = True
    outpaint_prompt: str = DEFAULT_OUTPAINT_PROMPT


@dataclass(slots=True)
class ImageJob:
    source: Path
    out_square: bool = True
    out_thumb: bool = True
    out_shorts: bool = True
    manual_boxes: tuple[Rect, ...] = ()
    ocr_boxes: tuple[Rect, ...] = ()
    preset_name: str = "OldPopLounge"
    ocr_languages: tuple[str, ...] = ("en",)
    extension_mode: str = "ai_natural"
    subject_offset_x: float = 0.0
    subject_offset_y: float = 0.0
    subject_scale: float = 1.0
    outpaint_prompt: str = DEFAULT_OUTPAINT_PROMPT
    status: str = "대기"
    error: str = ""


@dataclass(slots=True)
class FileResult:
    source: Path
    item_dir: Path
    status: str
    success_outputs: int
    failed_outputs: int
    skipped_outputs: int = 0
    failed_file_names: list[str] = field(default_factory=list)
    errors: list[dict[str, str]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


def _emit(callback: ProgressCallback | None, **payload: Any) -> None:
    if callback is not None:
        callback(payload)


def _check_cancel(cancel_event: Event | None) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise PipelineCancelled("작업이 취소되었습니다.")


def _error(category: str, message: str, detail: str | None = None) -> dict[str, str]:
    item = {"category": category, "message": message}
    if detail:
        item["detail"] = detail
    return item


def _dedupe_boxes(boxes: Sequence[Rect]) -> list[Rect]:
    seen: set[Rect] = set()
    unique: list[Rect] = []
    for box in boxes:
        normalized = tuple(int(value) for value in box)
        if normalized not in seen:
            unique.append(normalized)
            seen.add(normalized)
    return unique


def _load_image(path: Path) -> Image.Image:
    with Image.open(path) as opened:
        return ImageOps.exif_transpose(opened).convert("RGB")


def _write_job_json(root: Path, item_dir: Path, metadata: dict[str, Any]) -> None:
    try:
        item_dir.mkdir(parents=True, exist_ok=True)
        (item_dir / "covermorph_job.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except (OSError, ValueError) as exc:
        write_exception(root, "Job JSON write failed", exc)


def count_selected_outputs_for_options(options: PipelineOptions) -> int:
    return int(options.out_square) + int(options.out_thumb) + int(options.out_shorts)


def count_selected_outputs_for_jobs(jobs: Sequence[ImageJob]) -> int:
    return sum(int(job.out_square) + int(job.out_thumb) + int(job.out_shorts) for job in jobs)


def output_count_by_kind(jobs: Sequence[ImageJob]) -> dict[str, int]:
    return {
        "square_1x1": sum(1 for job in jobs if job.out_square),
        "thumbnail_16x9": sum(1 for job in jobs if job.out_thumb),
        "shorts_9x16": sum(1 for job in jobs if job.out_shorts),
    }


def progress_ratio(completed_outputs: int, total_outputs: int) -> float:
    if total_outputs <= 0:
        return 0.0
    return max(0.0, min(1.0, completed_outputs / total_outputs))


def options_for_job(base_options: PipelineOptions, job: ImageJob) -> PipelineOptions:
    text_boxes = tuple(_dedupe_boxes([*job.ocr_boxes, *job.manual_boxes]))
    return replace(
        base_options,
        preset_name=job.preset_name,
        ocr_languages=job.ocr_languages,
        manual_boxes=text_boxes,
        out_square=job.out_square,
        out_thumb=job.out_thumb,
        out_shorts=job.out_shorts,
        extension_mode=job.extension_mode,
        subject_offset_x=job.subject_offset_x,
        subject_offset_y=job.subject_offset_y,
        subject_scale=job.subject_scale,
        outpaint_prompt=job.outpaint_prompt,
    )


def _base_metadata(
    source: Path,
    options: PipelineOptions,
    preset: ChannelPreset,
) -> dict[str, Any]:
    return {
        "program_version": __version__,
        "version": __version__,
        "source_file": source.name,
        "original_filename": source.name,
        "preset": preset_to_dict(preset),
        "ocr_languages": list(options.ocr_languages),
        "detected_text_regions": 0,
        "text_boxes_total": 0,
        "text_removal_engine": "Skipped",
        "inpaint_engine": "Skipped",
        "upscale_engine": "Skipped",
        "thumbnail_16x9_engine": "Skipped",
        "shorts_9x16_engine": "Skipped",
        "thumbnail_resolution": options.thumbnail_resolution,
        "duplicate_policy": options.duplicate_policy,
        "extension_mode": options.extension_mode,
        "extension_fallbacks": [],
        "subject_offset_x": options.subject_offset_x,
        "subject_offset_y": options.subject_offset_y,
        "subject_scale": options.subject_scale,
        "core_protection": options.protect_core,
        "selected_outputs": {
            "square_1x1": options.out_square,
            "thumbnail_16x9": options.out_thumb,
            "shorts_9x16": options.out_shorts,
        },
        "sdxl_fallback": False,
        "sdxl_failures": [],
        "person_protection": options.protect_person,
        "person_protection_engine": "Disabled" if not options.protect_person else "Pending",
        "outpaint_prompt": options.outpaint_prompt,
        "output_files": {},
        "outputs": {},
        "skipped_outputs": 0,
        "processing_time_seconds": 0.0,
        "status": "pending",
        "errors": [],
    }


def _detect_and_remove_text(
    root: Path,
    ai: AIBackends,
    img: Image.Image,
    options: PipelineOptions,
    metadata: dict[str, Any],
    progress_callback: ProgressCallback | None,
) -> tuple[Image.Image, list[Rect], list[dict[str, str]]]:
    detected_boxes: list[Rect] = []
    errors: list[dict[str, str]] = []

    if options.auto_remove_text:
        _emit(progress_callback, type="file_stage", status="OCR 처리 중")
        try:
            detected_boxes = detect_text_boxes_multilang(
                img,
                options.ocr_languages,
                status_callback=lambda message: _emit(progress_callback, type="status", message=message),
            )
        except Exception as exc:
            write_exception(root, "OCR fallback", exc)
            errors.append(_error("ocr_failed", "OCR failed; manual masks can still be used.", str(exc)))

    boxes = _dedupe_boxes([*detected_boxes, *options.manual_boxes])
    metadata["detected_text_regions"] = len(detected_boxes)
    metadata["text_boxes_total"] = len(boxes)

    if not boxes:
        metadata["text_removal_engine"] = "No text mask"
        metadata["inpaint_engine"] = "No text mask"
        return img.copy(), boxes, errors

    _emit(progress_callback, type="file_stage", status="글자 제거 중")
    if options.prefer_lama and ai.lama_available():
        try:
            clean, engine = ai.inpaint(img, mask_pil_from_boxes(img.size, boxes))
            metadata["text_removal_engine"] = engine
            metadata["inpaint_engine"] = engine
            return clean, boxes, errors
        except Exception as exc:
            write_exception(root, "LaMa fallback", exc)
            errors.append(_error("lama_failed", "LaMa failed; OpenCV fallback was used.", str(exc)))

    try:
        clean = inpaint_text_opencv(img, boxes)
        metadata["text_removal_engine"] = "OpenCV NS/Telea (quality selected)"
        metadata["inpaint_engine"] = "OpenCV NS/Telea (quality selected)"
        return clean, boxes, errors
    except Exception as exc:
        write_exception(root, "OpenCV inpaint failed", exc)
        errors.append(
            _error("opencv_inpaint_failed", "Text removal failed; original pixels were kept.", str(exc))
        )
        metadata["text_removal_engine"] = "Failed; original kept"
        metadata["inpaint_engine"] = "Failed; original kept"
        return img.copy(), boxes, errors


def _upscale_or_fallback(
    root: Path,
    ai: AIBackends,
    img: Image.Image,
    prefer_esrgan: bool,
) -> tuple[Image.Image, str, list[dict[str, str]]]:
    if not prefer_esrgan:
        return img, "Skipped", []

    try:
        return (*ai.upscale(img, scale=2, model="realesrgan-x4plus"), [])
    except Exception as exc:
        write_exception(root, "Real-ESRGAN fallback", exc)
        enhanced = mild_enhance(img, 0.65)
        return (
            enhanced,
            "Local sharpen (Real-ESRGAN fallback)",
            [_error("realesrgan_failed", "Real-ESRGAN failed or is incomplete; local sharpen was used.", str(exc))],
        )


def _person_mask_or_whole_foreground(
    root: Path,
    ai: AIBackends,
    img: Image.Image,
    protect_person: bool,
) -> tuple[Image.Image | None, str, dict[str, str] | None]:
    if not protect_person:
        return None, "Disabled", None
    if not ai.rembg_available():
        return None, "Core region protect (rembg unavailable)", None
    try:
        mask, engine = ai.person_mask(img)
        return mask, engine, None
    except Exception as exc:
        write_exception(root, "Person segmentation fallback", exc)
        return (
            None,
            "Core region protect (person segmentation fallback)",
            _error(
                "person_segmentation_failed",
                "Person segmentation failed; the central core was protected instead.",
                str(exc),
            ),
        )


def _local_format(
    img: Image.Image,
    preset: ChannelPreset,
    kind: str,
    size: tuple[int, int],
    options: PipelineOptions,
    mode: str | None = None,
) -> tuple[Image.Image, str]:
    selected_mode = mode or options.extension_mode
    anchor = preset.person_anchor_16x9 if kind == "thumbnail" else preset.person_anchor_9x16
    if selected_mode == "natural":
        return (
            natural_background_extend(
                img,
                size,
                preset,
                kind,
                anchor=anchor,
                offset_x=options.subject_offset_x,
                offset_y=options.subject_offset_y,
                subject_scale=options.subject_scale,
            ),
            "Natural edge extension",
        )
    return render_full_frame_format(
        img,
        size,
        preset,
        kind,
        mode=selected_mode,
        anchor=anchor,
        offset_x=options.subject_offset_x,
        offset_y=options.subject_offset_y,
        subject_scale=options.subject_scale,
    )


def _outpaint_or_fallback(
    root: Path,
    ai: AIBackends,
    img: Image.Image,
    preset: ChannelPreset,
    options: PipelineOptions,
    kind: str,
    size: tuple[int, int],
    anchor: str,
) -> tuple[Image.Image, str, list[dict[str, str]], str]:
    errors: list[dict[str, str]] = []
    person_engine = "Disabled" if not options.protect_person else "Pending"

    if options.extension_mode != "ai_natural":
        local, engine = _local_format(img, preset, kind, size, options)
        return local, engine, errors, person_engine

    if not options.use_sdxl or not ai.sdxl_available():
        try:
            local, engine = _local_format(img, preset, kind, size, options, mode="natural")
            write_log(root, f"SDXL unavailable for {kind}; Natural extension fallback was used.")
            errors.append(
                _error(
                    "sdxl_unavailable",
                    f"SDXL unavailable for {kind}; Natural extension fallback was used.",
                )
            )
            return local, f"Fallback natural extension ({engine})", errors, person_engine
        except Exception as exc:
            write_exception(root, f"Natural extension fallback failed {kind}", exc)
            local, engine = _local_format(img, preset, kind, size, options, mode="blur")
            errors.append(
                _error(
                    "natural_extension_failed",
                    f"Natural extension failed for {kind}; Blur Canvas fallback was used.",
                    str(exc),
                )
            )
            return local, f"Fallback blur canvas ({engine})", errors, person_engine

    try:
        person_mask, person_engine, person_error = _person_mask_or_whole_foreground(
            root,
            ai,
            img,
            options.protect_person,
        )
        if person_error is not None:
            errors.append(person_error)
        canvas, mask, protect = build_full_frame_outpaint_canvas(
            img,
            size,
            preset,
            kind,
            person_mask,
            anchor=anchor,
            offset_x=options.subject_offset_x,
            offset_y=options.subject_offset_y,
            subject_scale=options.subject_scale,
            protect_core=options.protect_core,
        )
        if mask.size != size or protect.size != size:
            raise RuntimeError("Protection mask size does not match output canvas size.")
        generated, engine = ai.outpaint(
            canvas,
            mask,
            options.outpaint_prompt.strip() or DEFAULT_OUTPAINT_PROMPT,
            NEGATIVE_OUTPAINT_PROMPT,
        )
        return restore_protected_pixels(generated, protect), f"{engine} + {person_engine}", errors, person_engine
    except Exception as exc:
        write_exception(root, f"SDXL {kind} fallback", exc)
        errors.append(
            _error(
                "sdxl_failed",
                f"SDXL failed for {kind}; Natural extension fallback was used.",
                str(exc),
            )
        )
        try:
            local, engine = _local_format(img, preset, kind, size, options, mode="natural")
            return local, f"Fallback natural extension ({engine})", errors, person_engine
        except Exception as natural_exc:
            write_exception(root, f"Natural extension fallback failed {kind}", natural_exc)
            local, engine = _local_format(img, preset, kind, size, options, mode="blur")
            errors.append(
                _error(
                    "natural_extension_failed",
                    f"Natural extension failed for {kind}; Blur Canvas fallback was used.",
                    str(natural_exc),
                )
            )
            return local, f"Fallback blur canvas ({engine})", errors, person_engine


def _save_output(
    root: Path,
    img: Image.Image,
    path: Path,
    metadata: dict[str, Any],
    key: str,
    engine: str,
) -> tuple[int, int, int, list[dict[str, str]]]:
    resolved_path = resolve_duplicate_path(path, metadata.get("duplicate_policy", "new_number"))
    if resolved_path is None:
        metadata["outputs"][key] = {
            "status": "skipped",
            "path": str(path),
            "engine": engine,
            "reason": "File already exists",
        }
        return 0, 0, 1, []

    try:
        save_jpg(img, resolved_path)
        metadata["output_files"][key] = str(resolved_path)
        metadata["outputs"][key] = {"status": "success", "path": str(resolved_path), "engine": engine}
        if resolved_path != path:
            metadata["outputs"][key]["original_path"] = str(path)
        return 1, 0, 0, []
    except OSError as exc:
        write_exception(root, f"Save failed {path.name}", exc)
        metadata["outputs"][key] = {"status": "failed", "path": str(path), "engine": engine, "error": str(exc)}
        return 0, 1, 0, [_error("save_failed", f"Could not save {path.name}.", str(exc))]


def _output_done(progress_callback: ProgressCallback | None, key: str, filename: str, status: str) -> None:
    _emit(progress_callback, type="output_done", key=key, filename=filename, status=status)


def process_image_file(
    source: Path,
    options: PipelineOptions,
    ai: AIBackends,
    app_root: Path,
    progress_callback: ProgressCallback | None = None,
    cancel_event: Event | None = None,
) -> FileResult:
    start = time.perf_counter()
    preset = PRESETS[options.preset_name]
    item_dir = options.output_dir / source.stem
    metadata = _base_metadata(source, options, preset)
    errors: list[dict[str, str]] = []
    success_outputs = 0
    failed_outputs = 0
    skipped_outputs = 0
    failed_names: list[str] = []

    try:
        item_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        selected_count = max(1, count_selected_outputs_for_options(options))
        write_exception(app_root, f"Output folder failed {source.name}", exc)
        metadata["status"] = "failed"
        metadata["errors"].append(_error("output_folder_create_failed", "Output folder could not be created.", str(exc)))
        metadata["processing_time_seconds"] = round(time.perf_counter() - start, 3)
        return FileResult(
            source=source,
            item_dir=item_dir,
            status="failed",
            success_outputs=0,
            failed_outputs=selected_count,
            failed_file_names=[source.name],
            errors=metadata["errors"],
            metadata=metadata,
        )

    try:
        _check_cancel(cancel_event)
        img = _load_image(source)
    except PipelineCancelled:
        raise
    except FileNotFoundError as exc:
        errors.append(_error("read_error", "Source image was not found.", str(exc)))
        metadata["status"] = "failed"
        metadata["errors"] = errors
        metadata["processing_time_seconds"] = round(time.perf_counter() - start, 3)
        _write_job_json(app_root, item_dir, metadata)
        return FileResult(
            source,
            item_dir,
            "failed",
            0,
            max(1, count_selected_outputs_for_options(options)),
            0,
            [source.name],
            errors,
            metadata,
        )
    except UnidentifiedImageError as exc:
        errors.append(_error("corrupt_image", "Source image is damaged or unsupported.", str(exc)))
        metadata["status"] = "failed"
        metadata["errors"] = errors
        metadata["processing_time_seconds"] = round(time.perf_counter() - start, 3)
        _write_job_json(app_root, item_dir, metadata)
        return FileResult(
            source,
            item_dir,
            "failed",
            0,
            max(1, count_selected_outputs_for_options(options)),
            0,
            [source.name],
            errors,
            metadata,
        )
    except OSError as exc:
        errors.append(_error("read_error", "Source image could not be read.", str(exc)))
        metadata["status"] = "failed"
        metadata["errors"] = errors
        metadata["processing_time_seconds"] = round(time.perf_counter() - start, 3)
        _write_job_json(app_root, item_dir, metadata)
        return FileResult(
            source,
            item_dir,
            "failed",
            0,
            max(1, count_selected_outputs_for_options(options)),
            0,
            [source.name],
            errors,
            metadata,
        )

    clean, _boxes, text_errors = _detect_and_remove_text(
        app_root,
        ai,
        img,
        options,
        metadata,
        progress_callback,
    )
    errors.extend(text_errors)

    if options.enhance:
        clean = mild_enhance(clean, preset.enhance_strength)

    _check_cancel(cancel_event)
    clean, upscale_engine, upscale_errors = _upscale_or_fallback(
        app_root,
        ai,
        clean,
        options.prefer_esrgan,
    )
    metadata["upscale_engine"] = upscale_engine
    errors.extend(upscale_errors)

    if options.out_square:
        path = item_dir / f"{source.stem}_clean_1x1_1400x1400.jpg"
        try:
            _check_cancel(cancel_event)
            _emit(progress_callback, type="file_stage", status="1:1 생성 중")
            ok, fail, skipped, save_errors = _save_output(
                app_root,
                make_square(clean),
                path,
                metadata,
                "square_1x1",
                "Crop 1:1",
            )
            success_outputs += ok
            failed_outputs += fail
            skipped_outputs += skipped
            errors.extend(save_errors)
            if fail:
                failed_names.append(path.name)
            _output_done(
                progress_callback,
                "square_1x1",
                source.name,
                "success" if ok else "failed" if fail else "skipped",
            )
        except PipelineCancelled:
            raise
        except Exception as exc:
            write_exception(app_root, f"Square output failed {source.name}", exc)
            errors.append(_error("output_generation_failed", "1:1 output generation failed.", str(exc)))
            failed_outputs += 1
            failed_names.append(path.name)
            metadata["outputs"]["square_1x1"] = {
                "status": "failed",
                "path": str(path),
                "engine": "Crop 1:1",
                "error": str(exc),
            }
            _output_done(progress_callback, "square_1x1", source.name, "failed")

    if options.out_thumb:
        thumb_size = thumbnail_size(options.thumbnail_resolution)
        thumb_width, thumb_height = thumb_size
        path = item_dir / f"{source.stem}_thumbnail_16x9_{thumb_width}x{thumb_height}.jpg"
        try:
            _check_cancel(cancel_event)
            _emit(progress_callback, type="file_stage", status="16:9 생성 중")
            thumb, engine, outpaint_errors, person_engine = _outpaint_or_fallback(
                app_root,
                ai,
                clean,
                preset,
                options,
                "thumbnail",
                thumb_size,
                preset.person_anchor_16x9,
            )
            metadata["thumbnail_16x9_engine"] = engine
            metadata["person_protection_engine"] = person_engine
            errors.extend(outpaint_errors)
            ok, fail, skipped, save_errors = _save_output(app_root, thumb, path, metadata, "thumbnail_16x9", engine)
            success_outputs += ok
            failed_outputs += fail
            skipped_outputs += skipped
            errors.extend(save_errors)
            if fail:
                failed_names.append(path.name)
            _output_done(
                progress_callback,
                "thumbnail_16x9",
                source.name,
                "success" if ok else "failed" if fail else "skipped",
            )
        except PipelineCancelled:
            raise
        except Exception as exc:
            write_exception(app_root, f"Thumbnail output failed {source.name}", exc)
            errors.append(_error("output_generation_failed", "16:9 output generation failed.", str(exc)))
            failed_outputs += 1
            failed_names.append(path.name)
            metadata["thumbnail_16x9_engine"] = "Failed"
            metadata["outputs"]["thumbnail_16x9"] = {
                "status": "failed",
                "path": str(path),
                "engine": "Failed",
                "error": str(exc),
            }
            _output_done(progress_callback, "thumbnail_16x9", source.name, "failed")

    if options.out_shorts:
        path = item_dir / f"{source.stem}_shorts_9x16_1080x1920.jpg"
        try:
            _check_cancel(cancel_event)
            _emit(progress_callback, type="file_stage", status="9:16 생성 중")
            shorts, engine, outpaint_errors, person_engine = _outpaint_or_fallback(
                app_root,
                ai,
                clean,
                preset,
                options,
                "shorts",
                (1080, 1920),
                preset.person_anchor_9x16,
            )
            metadata["shorts_9x16_engine"] = engine
            if metadata["person_protection_engine"] in {"Pending", "Disabled"}:
                metadata["person_protection_engine"] = person_engine
            errors.extend(outpaint_errors)
            ok, fail, skipped, save_errors = _save_output(app_root, shorts, path, metadata, "shorts_9x16", engine)
            success_outputs += ok
            failed_outputs += fail
            skipped_outputs += skipped
            errors.extend(save_errors)
            if fail:
                failed_names.append(path.name)
            _output_done(
                progress_callback,
                "shorts_9x16",
                source.name,
                "success" if ok else "failed" if fail else "skipped",
            )
        except PipelineCancelled:
            raise
        except Exception as exc:
            write_exception(app_root, f"Shorts output failed {source.name}", exc)
            errors.append(_error("output_generation_failed", "9:16 output generation failed.", str(exc)))
            failed_outputs += 1
            failed_names.append(path.name)
            metadata["shorts_9x16_engine"] = "Failed"
            metadata["outputs"]["shorts_9x16"] = {
                "status": "failed",
                "path": str(path),
                "engine": "Failed",
                "error": str(exc),
            }
            _output_done(progress_callback, "shorts_9x16", source.name, "failed")

    sdxl_errors = [error for error in errors if error["category"].startswith("sdxl")]
    extension_errors = [error for error in errors if "extension" in error["category"] or error["category"].startswith("sdxl")]
    metadata["sdxl_fallback"] = bool(sdxl_errors)
    metadata["sdxl_failures"] = sdxl_errors
    metadata["extension_fallbacks"] = extension_errors
    metadata["errors"] = errors
    metadata["skipped_outputs"] = skipped_outputs
    metadata["processing_time_seconds"] = round(time.perf_counter() - start, 3)

    if success_outputs == 0 and skipped_outputs > 0 and failed_outputs == 0:
        status = "skipped"
    elif success_outputs == 0:
        status = "failed"
        failed_names.append(source.name)
    elif errors or failed_outputs:
        status = "partial"
    else:
        status = "success"
    metadata["status"] = status
    _write_job_json(app_root, item_dir, metadata)
    write_log(
        app_root,
        (
            f"{status.upper()} {source.name} | outputs={success_outputs} | "
            f"inpaint={metadata['inpaint_engine']} | upscale={upscale_engine}"
        ),
    )
    return FileResult(
        source=source,
        item_dir=item_dir,
        status=status,
        success_outputs=success_outputs,
        failed_outputs=failed_outputs,
        skipped_outputs=skipped_outputs,
        failed_file_names=failed_names,
        errors=errors,
        metadata=metadata,
    )


def process_files(
    files: Sequence[Path],
    options: PipelineOptions,
    app_root: Path,
    ai: AIBackends | None = None,
    progress_callback: ProgressCallback | None = None,
    cancel_event: Event | None = None,
) -> list[FileResult]:
    jobs = [
        ImageJob(
            source=source,
            out_square=options.out_square,
            out_thumb=options.out_thumb,
            out_shorts=options.out_shorts,
            manual_boxes=options.manual_boxes if index == 0 else (),
            preset_name=options.preset_name,
            ocr_languages=options.ocr_languages,
            extension_mode=options.extension_mode,
            subject_offset_x=options.subject_offset_x,
            subject_offset_y=options.subject_offset_y,
            subject_scale=options.subject_scale,
            outpaint_prompt=options.outpaint_prompt,
        )
        for index, source in enumerate(files)
    ]
    return process_image_jobs(jobs, options, app_root, ai, progress_callback, cancel_event)


def process_image_jobs(
    jobs: Sequence[ImageJob],
    base_options: PipelineOptions,
    app_root: Path,
    ai: AIBackends | None = None,
    progress_callback: ProgressCallback | None = None,
    cancel_event: Event | None = None,
) -> list[FileResult]:
    backend = ai or AIBackends(app_root)
    results: list[FileResult] = []
    total_files = len(jobs)
    total_outputs = count_selected_outputs_for_jobs(jobs)
    completed_outputs = 0
    success_files = 0
    failed_files = 0
    success_outputs = 0
    failed_outputs = 0
    skipped_outputs = 0

    for index, job in enumerate(jobs, 1):
        if cancel_event is not None and cancel_event.is_set():
            _emit(progress_callback, type="cancelled", message="작업이 취소되었습니다.")
            break
        job.status = "대기"
        completed_before_job = completed_outputs
        _emit(
            progress_callback,
            type="file_start",
            index=index,
            total=total_files,
            total_outputs=total_outputs,
            completed_outputs=completed_outputs,
            filename=job.source.name,
            success_files=success_files,
            failed_files=failed_files,
        )

        def job_progress(payload: dict[str, Any], *, job_index: int = index, job_source: Path = job.source) -> None:
            nonlocal completed_outputs
            if payload.get("type") == "file_stage":
                job.status = str(payload.get("status", "처리 중"))
                _emit(
                    progress_callback,
                    type="file_stage",
                    index=job_index,
                    filename=job_source.name,
                    status=job.status,
                    total_outputs=total_outputs,
                    completed_outputs=completed_outputs,
                )
                return
            if payload.get("type") == "output_done":
                completed_outputs += 1
                _emit(
                    progress_callback,
                    type="output_progress",
                    index=job_index,
                    filename=job_source.name,
                    key=payload.get("key"),
                    status=payload.get("status"),
                    total_outputs=total_outputs,
                    completed_outputs=completed_outputs,
                    ratio=progress_ratio(completed_outputs, total_outputs),
                )
                return
            _emit(progress_callback, **payload)

        try:
            result = process_image_file(
                job.source,
                options_for_job(base_options, job),
                backend,
                app_root,
                progress_callback=job_progress,
                cancel_event=cancel_event,
            )
        except PipelineCancelled:
            job.status = "취소됨"
            _emit(progress_callback, type="cancelled", message="작업이 취소되었습니다.")
            break
        except Exception as exc:
            write_exception(app_root, f"Unexpected file failure {job.source.name}", exc)
            job.status = "실패"
            job.error = str(exc)
            result = FileResult(
                source=job.source,
                item_dir=base_options.output_dir / job.source.stem,
                status="failed",
                success_outputs=0,
                failed_outputs=max(1, count_selected_outputs_for_options(options_for_job(base_options, job))),
                skipped_outputs=0,
                failed_file_names=[job.source.name],
                errors=[_error("unexpected_error", "Unexpected processing error.", str(exc))],
                metadata={},
            )

        result_done_outputs = result.success_outputs + result.failed_outputs + result.skipped_outputs
        emitted_done_outputs = completed_outputs - completed_before_job
        if result_done_outputs > emitted_done_outputs:
            completed_outputs += result_done_outputs - emitted_done_outputs
            _emit(
                progress_callback,
                type="output_progress",
                index=index,
                filename=job.source.name,
                key="file_result",
                status=result.status,
                total_outputs=total_outputs,
                completed_outputs=completed_outputs,
                ratio=progress_ratio(completed_outputs, total_outputs),
            )

        results.append(result)
        if result.status == "success":
            success_files += 1
            job.status = "저장 완료"
        elif result.status == "skipped":
            job.status = "건너뜀"
        elif result.success_outputs > 0:
            failed_files += 1
            job.status = "일부 완료"
            job.error = "; ".join(error["message"] for error in result.errors)
        else:
            failed_files += 1
            job.status = "실패"
            job.error = "; ".join(error["message"] for error in result.errors)
        success_outputs += result.success_outputs
        failed_outputs += result.failed_outputs
        skipped_outputs += result.skipped_outputs
        _emit(
            progress_callback,
            type="file_done",
            index=index,
            total=total_files,
            total_outputs=total_outputs,
            completed_outputs=completed_outputs,
            filename=job.source.name,
            status=result.status,
            success_files=success_files,
            failed_files=failed_files,
            success_outputs=success_outputs,
            failed_outputs=failed_outputs,
            skipped_outputs=skipped_outputs,
            failed_file_names=result.failed_file_names,
        )

    _emit(
        progress_callback,
        type="batch_done",
        total=total_files,
        processed=len(results),
        total_outputs=total_outputs,
        completed_outputs=completed_outputs,
        success_files=success_files,
        failed_files=failed_files,
        success_outputs=success_outputs,
        failed_outputs=failed_outputs,
        skipped_outputs=skipped_outputs,
    )
    return results

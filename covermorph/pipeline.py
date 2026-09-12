from __future__ import annotations

import json
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from threading import Event
from typing import Any

from PIL import Image, UnidentifiedImageError

from . import __version__
from .ai_plugins import AIBackends
from .logger import write_exception, write_log
from .presets import PRESETS, ChannelPreset, preset_to_dict
from .processor import (
    Rect,
    build_outpaint_canvas,
    detect_text_boxes_easyocr,
    inpaint_text_opencv,
    make_shorts,
    make_square,
    make_text_safe_landscape,
    mask_pil_from_boxes,
    mild_enhance,
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
    use_sdxl: bool = False
    protect_person: bool = True
    outpaint_prompt: str = DEFAULT_OUTPAINT_PROMPT


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
        normalized = tuple(int(v) for v in box)
        if normalized not in seen:
            unique.append(normalized)
            seen.add(normalized)
    return unique


def _load_image(path: Path) -> Image.Image:
    with Image.open(path) as opened:
        return opened.convert("RGB")


def _write_job_json(root: Path, item_dir: Path, metadata: dict[str, Any]) -> None:
    try:
        item_dir.mkdir(parents=True, exist_ok=True)
        (item_dir / "covermorph_job.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except (OSError, ValueError) as exc:
        write_exception(root, "Job JSON write failed", exc)


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
        try:
            detected_boxes = detect_text_boxes_easyocr(
                img,
                options.ocr_languages,
                status_callback=lambda message: _emit(
                    progress_callback,
                    type="status",
                    message=message,
                ),
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
        metadata["text_removal_engine"] = "OpenCV Telea"
        metadata["inpaint_engine"] = "OpenCV Telea"
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
        return None, "Whole foreground protect (rembg unavailable)", None
    try:
        mask, engine = ai.person_mask(img)
        return mask, engine, None
    except Exception as exc:
        write_exception(root, "Person segmentation fallback", exc)
        return (
            None,
            "Whole foreground protect (person segmentation fallback)",
            _error(
                "person_segmentation_failed",
                "Person segmentation failed; the full foreground was protected instead.",
                str(exc),
            ),
        )


def _local_format(
    img: Image.Image,
    preset: ChannelPreset,
    kind: str,
    size: tuple[int, int],
) -> tuple[Image.Image, str]:
    if kind == "thumbnail":
        return make_text_safe_landscape(img, preset, size), "Blur Canvas"
    return make_shorts(img, preset), "Blur Canvas"


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

    if not options.use_sdxl:
        local, engine = _local_format(img, preset, kind, size)
        return local, engine, errors, person_engine

    if not ai.sdxl_available():
        local, _engine = _local_format(img, preset, kind, size)
        write_log(root, f"SDXL unavailable for {kind}; Blur Canvas fallback was used.")
        error = _error(
            "sdxl_unavailable",
            f"SDXL unavailable for {kind}; Blur Canvas fallback was used.",
        )
        errors.append(error)
        return local, "Fallback blur canvas", errors, person_engine

    try:
        person_mask, person_engine, person_error = _person_mask_or_whole_foreground(
            root,
            ai,
            img,
            options.protect_person,
        )
        if person_error is not None:
            errors.append(person_error)
        canvas, mask, protect = build_outpaint_canvas(img, size, person_mask, anchor=anchor)
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
        local, _engine = _local_format(img, preset, kind, size)
        errors.append(
            _error(
                "sdxl_failed",
                f"SDXL failed for {kind}; Blur Canvas fallback was used.",
                str(exc),
            )
        )
        return local, "Fallback blur canvas", errors, person_engine


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
        write_exception(app_root, f"Output folder failed {source.name}", exc)
        metadata["status"] = "failed"
        metadata["errors"].append(_error("output_folder_create_failed", "Output folder could not be created.", str(exc)))
        metadata["processing_time_seconds"] = round(time.perf_counter() - start, 3)
        return FileResult(
            source=source,
            item_dir=item_dir,
            status="failed",
            success_outputs=0,
            failed_outputs=1,
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
            source=source,
            item_dir=item_dir,
            status="failed",
            success_outputs=0,
            failed_outputs=1,
            failed_file_names=[source.name],
            errors=errors,
            metadata=metadata,
        )
    except UnidentifiedImageError as exc:
        errors.append(_error("corrupt_image", "Source image is damaged or unsupported.", str(exc)))
        metadata["status"] = "failed"
        metadata["errors"] = errors
        metadata["processing_time_seconds"] = round(time.perf_counter() - start, 3)
        _write_job_json(app_root, item_dir, metadata)
        return FileResult(
            source=source,
            item_dir=item_dir,
            status="failed",
            success_outputs=0,
            failed_outputs=1,
            failed_file_names=[source.name],
            errors=errors,
            metadata=metadata,
        )
    except OSError as exc:
        errors.append(_error("read_error", "Source image could not be read.", str(exc)))
        metadata["status"] = "failed"
        metadata["errors"] = errors
        metadata["processing_time_seconds"] = round(time.perf_counter() - start, 3)
        _write_job_json(app_root, item_dir, metadata)
        return FileResult(
            source=source,
            item_dir=item_dir,
            status="failed",
            success_outputs=0,
            failed_outputs=1,
            failed_file_names=[source.name],
            errors=errors,
            metadata=metadata,
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

    if options.out_thumb:
        thumb_size = thumbnail_size(options.thumbnail_resolution)
        thumb_width, thumb_height = thumb_size
        path = item_dir / f"{source.stem}_thumbnail_16x9_{thumb_width}x{thumb_height}.jpg"
        try:
            _check_cancel(cancel_event)
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

    if options.out_shorts:
        path = item_dir / f"{source.stem}_shorts_9x16_1080x1920.jpg"
        try:
            _check_cancel(cancel_event)
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

    sdxl_errors = [error for error in errors if error["category"].startswith("sdxl")]
    metadata["sdxl_fallback"] = bool(sdxl_errors)
    metadata["sdxl_failures"] = sdxl_errors
    metadata["errors"] = errors
    metadata["processing_time_seconds"] = round(time.perf_counter() - start, 3)

    metadata["skipped_outputs"] = skipped_outputs

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
    backend = ai or AIBackends(app_root)
    results: list[FileResult] = []
    total = len(files)
    success_files = 0
    failed_files = 0
    success_outputs = 0
    failed_outputs = 0
    skipped_outputs = 0

    for index, source in enumerate(files, 1):
        if cancel_event is not None and cancel_event.is_set():
            _emit(progress_callback, type="cancelled", message="작업이 취소되었습니다.")
            break
        _emit(
            progress_callback,
            type="file_start",
            index=index,
            total=total,
            filename=source.name,
            success_files=success_files,
            failed_files=failed_files,
        )
        try:
            item_options = options
            if index != 1 and options.manual_boxes:
                item_options = replace(options, manual_boxes=())
            result = process_image_file(
                source,
                item_options,
                backend,
                app_root,
                progress_callback=progress_callback,
                cancel_event=cancel_event,
            )
        except PipelineCancelled:
            _emit(progress_callback, type="cancelled", message="작업이 취소되었습니다.")
            break
        except Exception as exc:
            write_exception(app_root, f"Unexpected file failure {source.name}", exc)
            result = FileResult(
                source=source,
                item_dir=options.output_dir / source.stem,
                status="failed",
                success_outputs=0,
                failed_outputs=1,
                skipped_outputs=0,
                failed_file_names=[source.name],
                errors=[_error("unexpected_error", "Unexpected processing error.", str(exc))],
                metadata={},
            )

        results.append(result)
        if result.status == "success":
            success_files += 1
        elif result.status != "skipped":
            failed_files += 1
        success_outputs += result.success_outputs
        failed_outputs += result.failed_outputs
        skipped_outputs += result.skipped_outputs
        _emit(
            progress_callback,
            type="file_done",
            index=index,
            total=total,
            filename=source.name,
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
        total=total,
        processed=len(results),
        success_files=success_files,
        failed_files=failed_files,
        success_outputs=success_outputs,
        failed_outputs=failed_outputs,
        skipped_outputs=skipped_outputs,
    )
    return results

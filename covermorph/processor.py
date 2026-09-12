from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageEnhance, ImageFilter, ImageOps

Rect = tuple[int, int, int, int]
StatusCallback = Callable[[str], None]

_EASYOCR_READERS: dict[tuple[str, ...], Any] = {}


def pil_to_cv(img: Image.Image) -> np.ndarray:
    arr = np.array(img.convert("RGB"))
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def cv_to_pil(arr: np.ndarray) -> Image.Image:
    rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def easyocr_reader(langs: Sequence[str], status_callback: StatusCallback | None = None) -> Any:
    """Return a cached EasyOCR reader for the requested language combination."""
    key = tuple(langs)
    if key not in _EASYOCR_READERS:
        if status_callback is not None:
            status_callback("EasyOCR 모델을 준비 중입니다. 첫 실행이면 다운로드가 진행될 수 있습니다.")
        import easyocr

        _EASYOCR_READERS[key] = easyocr.Reader(list(key), gpu=False, verbose=False)
    return _EASYOCR_READERS[key]


def clear_easyocr_cache() -> None:
    _EASYOCR_READERS.clear()


def detect_text_boxes_easyocr(
    img: Image.Image,
    langs: Sequence[str] = ("en",),
    min_conf: float = 0.15,
    status_callback: StatusCallback | None = None,
) -> list[Rect]:
    try:
        reader = easyocr_reader(langs, status_callback=status_callback)
    except ImportError:
        return []

    arr = np.array(img.convert("RGB"))
    results = reader.readtext(arr, detail=1, paragraph=False)
    boxes: list[Rect] = []
    for quad, _text, conf in results:
        if conf < min_conf:
            continue
        xs = [point[0] for point in quad]
        ys = [point[1] for point in quad]
        x1, x2 = int(min(xs)), int(max(xs))
        y1, y2 = int(min(ys)), int(max(ys))
        pad = max(6, int(min(img.size) * 0.008))
        boxes.append(
            (
                max(0, x1 - pad),
                max(0, y1 - pad),
                min(img.width, x2 + pad),
                min(img.height, y2 + pad),
            )
        )
    return boxes


def _boxes_overlap(left: Rect, right: Rect) -> bool:
    intersection_width = max(0, min(left[2], right[2]) - max(left[0], right[0]))
    intersection_height = max(0, min(left[3], right[3]) - max(left[1], right[1]))
    intersection = intersection_width * intersection_height
    if not intersection:
        return False
    left_area = max(1, (left[2] - left[0]) * (left[3] - left[1]))
    right_area = max(1, (right[2] - right[0]) * (right[3] - right[1]))
    return intersection / min(left_area, right_area) >= 0.10


def merge_text_boxes(box_groups: Sequence[Sequence[Rect]]) -> list[Rect]:
    """Merge overlapping OCR candidates from several language readers."""
    merged: list[Rect] = []
    for group in box_groups:
        for box in group:
            current = tuple(int(value) for value in box)
            for index, existing in enumerate(merged):
                if _boxes_overlap(existing, current):
                    merged[index] = (
                        min(existing[0], current[0]),
                        min(existing[1], current[1]),
                        max(existing[2], current[2]),
                        max(existing[3], current[3]),
                    )
                    break
            else:
                merged.append(current)
    return merged


def detect_text_boxes_multilang(
    img: Image.Image,
    langs: Sequence[str],
    status_callback: StatusCallback | None = None,
) -> list[Rect]:
    """Run the selected reader and language-specific helpers without aborting the job."""
    requested = tuple(dict.fromkeys(langs))
    groups: list[list[Rect]] = []
    combinations = [requested]
    for code in ("ja", "ko", "en", "fr"):
        if code in requested and (code,) not in combinations:
            combinations.append((code,))
    for combination in combinations:
        try:
            groups.append(detect_text_boxes_easyocr(img, combination, status_callback=status_callback))
        except (ImportError, RuntimeError, OSError, ValueError):
            continue
    return merge_text_boxes(groups)


def mask_from_boxes(size: tuple[int, int], boxes: Sequence[Rect], dilation: int = 9) -> np.ndarray:
    width, height = size
    mask = np.zeros((height, width), dtype=np.uint8)
    for x1, y1, x2, y2 in boxes:
        left = max(0, min(width, int(x1)))
        top = max(0, min(height, int(y1)))
        right = max(0, min(width, int(x2)))
        bottom = max(0, min(height, int(y2)))
        if right > left and bottom > top:
            cv2.rectangle(mask, (left, top), (right, bottom), 255, -1)
    if dilation > 0 and np.any(mask):
        kernel_size = max(3, dilation | 1)
        mask = cv2.dilate(mask, np.ones((kernel_size, kernel_size), np.uint8), iterations=1)
    return mask


def mask_pil_from_boxes(size: tuple[int, int], boxes: Sequence[Rect]) -> Image.Image:
    return Image.fromarray(adaptive_mask_from_boxes(size, boxes)).convert("L")


def adaptive_mask_from_boxes(size: tuple[int, int], boxes: Sequence[Rect]) -> np.ndarray:
    width, height = size
    mask = np.zeros((height, width), dtype=np.uint8)
    for x1, y1, x2, y2 in boxes:
        box_width = max(1, int(x2) - int(x1))
        box_height = max(1, int(y2) - int(y1))
        padding = max(3, min(48, int(max(box_width, box_height) * 0.14)))
        left = max(0, min(width, int(x1) - padding))
        top = max(0, min(height, int(y1) - padding))
        right = max(0, min(width, int(x2) + padding))
        bottom = max(0, min(height, int(y2) + padding))
        if right > left and bottom > top:
            cv2.rectangle(mask, (left, top), (right, bottom), 255, -1)
    return mask


def _inpaint_quality(candidate: np.ndarray, original: np.ndarray, mask: np.ndarray) -> float:
    boundary = cv2.subtract(
        cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=1),
        cv2.erode(mask, np.ones((5, 5), np.uint8), iterations=1),
    )
    pixels = boundary > 0
    if not np.any(pixels):
        return float("inf")
    candidate_gray = cv2.cvtColor(candidate, cv2.COLOR_BGR2GRAY)
    original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    return float(np.mean(np.abs(candidate_gray[pixels].astype(np.float32) - original_gray[pixels].astype(np.float32))))


def inpaint_text_opencv(img: Image.Image, boxes: Sequence[Rect], radius: int = 7) -> Image.Image:
    if not boxes:
        return img.copy()
    arr = pil_to_cv(img)
    mask = adaptive_mask_from_boxes(img.size, boxes)
    ns = cv2.inpaint(arr, mask, max(3, radius), cv2.INPAINT_NS)
    telea = cv2.inpaint(arr, mask, max(3, radius), cv2.INPAINT_TELEA)
    result = ns if _inpaint_quality(ns, arr, mask) <= _inpaint_quality(telea, arr, mask) else telea
    # A narrow feather hides the rectangular inpaint boundary without restoring text pixels.
    soft_mask = cv2.GaussianBlur(mask, (0, 0), max(1.0, radius / 2)) / 255.0
    blended = (result * soft_mask[..., None] + arr * (1.0 - soft_mask[..., None])).clip(0, 255).astype(np.uint8)
    return cv_to_pil(blended)


def detect_largest_face(img: Image.Image) -> Rect | None:
    if not hasattr(cv2, "CascadeClassifier") or not hasattr(cv2, "data"):
        return None
    gray = cv2.cvtColor(pil_to_cv(img), cv2.COLOR_BGR2GRAY)
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(cascade_path)
    if cascade.empty():
        return None
    faces = cascade.detectMultiScale(gray, 1.08, 5, minSize=(48, 48))
    if len(faces) == 0:
        return None
    x, y, width, height = max(faces, key=lambda face: face[2] * face[3])
    return (int(x), int(y), int(x + width), int(y + height))


def mild_enhance(img: Image.Image, strength: float = 1.0) -> Image.Image:
    arr = pil_to_cv(img)
    blur = cv2.GaussianBlur(arr, (0, 0), 1.0)
    alpha = 1.0 + 0.16 * strength
    sharp = cv2.addWeighted(arr, alpha, blur, -(alpha - 1.0), 0)
    sharp = np.clip(sharp, 0, 255).astype(np.uint8)
    out = cv_to_pil(sharp)
    return ImageEnhance.Contrast(out).enhance(1.02)


def blurred_background(
    img: Image.Image,
    size: tuple[int, int],
    blur: int = 36,
) -> Image.Image:
    bg = ImageOps.fit(img, size, method=Image.Resampling.LANCZOS)
    bg = bg.filter(ImageFilter.GaussianBlur(blur))
    return ImageEnhance.Brightness(bg).enhance(0.88)


def _target_fill_scale(
    img: Image.Image,
    size: tuple[int, int],
    kind: str,
    subject_scale: float,
) -> float:
    target_width, target_height = size
    safe_scale = max(0.75, min(1.45, subject_scale))
    if kind == "thumbnail":
        scale = target_height / img.height
    elif kind == "shorts":
        scale = target_width / img.width
    else:
        scale = max(target_width / img.width, target_height / img.height)
    return scale * safe_scale


def _foreground_position(
    fg_size: tuple[int, int],
    target_size: tuple[int, int],
    kind: str,
    anchor: str,
    safe_ratio: float,
    offset_x: float,
    offset_y: float,
) -> tuple[int, int]:
    fg_width, fg_height = fg_size
    target_width, target_height = target_size
    if kind == "thumbnail":
        if anchor == "right":
            x = target_width - fg_width
            safe_left = int(target_width * safe_ratio)
            if fg_width < target_width - safe_left:
                x = max(safe_left, x)
        elif anchor == "left":
            x = 0
        else:
            x = (target_width - fg_width) // 2
        y = (target_height - fg_height) // 2
    elif kind == "shorts":
        x = (target_width - fg_width) // 2
        y = (target_height - fg_height) // 2
    else:
        x = (target_width - fg_width) // 2
        y = (target_height - fg_height) // 2

    x += int(target_width * max(-0.35, min(0.35, offset_x)))
    y += int(target_height * max(-0.35, min(0.35, offset_y)))
    return x, y


def _cover_background(img: Image.Image, size: tuple[int, int]) -> Image.Image:
    target_width, target_height = size
    scale = min(target_width / img.width, target_height / img.height)
    fg = img.resize(
        (max(1, int(img.width * scale)), max(1, int(img.height * scale))),
        Image.Resampling.LANCZOS,
    )
    bg = Image.new("RGB", size, img.resize((1, 1), Image.Resampling.BILINEAR).getpixel((0, 0)))
    x = (target_width - fg.width) // 2
    y = (target_height - fg.height) // 2
    _paste_axis_extensions(bg, fg, x, y)
    return bg.filter(ImageFilter.GaussianBlur(max(2, int(min(size) * 0.003))))


def _paste_axis_extensions(base: Image.Image, fg: Image.Image, x: int, y: int) -> None:
    target_width, target_height = base.size
    fg_width, fg_height = fg.size
    crop_left = max(0, x)
    crop_top = max(0, y)
    crop_right = min(target_width, x + fg_width)
    crop_bottom = min(target_height, y + fg_height)

    visible_left = crop_left - x
    visible_top = crop_top - y
    visible_right = visible_left + max(0, crop_right - crop_left)
    visible_bottom = visible_top + max(0, crop_bottom - crop_top)
    if visible_right <= visible_left or visible_bottom <= visible_top:
        return

    visible = fg.crop((visible_left, visible_top, visible_right, visible_bottom))
    strip = max(8, min(96, min(visible.size) // 5))

    if crop_left > 0:
        left_strip = visible.crop((0, 0, min(strip, visible.width), visible.height))
        left_ext = ImageOps.mirror(left_strip).resize((crop_left, visible.height), Image.Resampling.BICUBIC)
        base.paste(left_ext, (0, crop_top))
    if crop_right < target_width:
        right_strip = visible.crop((max(0, visible.width - strip), 0, visible.width, visible.height))
        right_ext = ImageOps.mirror(right_strip).resize(
            (target_width - crop_right, visible.height),
            Image.Resampling.BICUBIC,
        )
        base.paste(right_ext, (crop_right, crop_top))
    if crop_top > 0:
        top_strip = visible.crop((0, 0, visible.width, min(strip, visible.height)))
        top_ext = ImageOps.flip(top_strip).resize((visible.width, crop_top), Image.Resampling.BICUBIC)
        base.paste(top_ext, (crop_left, 0))
    if crop_bottom < target_height:
        bottom_strip = visible.crop((0, max(0, visible.height - strip), visible.width, visible.height))
        bottom_ext = ImageOps.flip(bottom_strip).resize(
            (visible.width, target_height - crop_bottom),
            Image.Resampling.BICUBIC,
        )
        base.paste(bottom_ext, (crop_left, crop_bottom))


def _feather_mask(size: tuple[int, int], feather: int) -> Image.Image:
    width, height = size
    feather = max(1, min(feather, width // 4, height // 4))
    mask = Image.new("L", size, 255)
    edge = Image.new("L", size, 0)
    draw = ImageDraw.Draw(edge)
    draw.rectangle((feather, feather, width - feather, height - feather), fill=255)
    edge = edge.filter(ImageFilter.GaussianBlur(feather / 2))
    return ImageChops.multiply(mask, edge)


def core_protection_mask(size: tuple[int, int], kind: str, feather: int = 18) -> Image.Image:
    width, height = size
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    if kind == "thumbnail":
        rect = (
            int(width * 0.14),
            int(height * 0.08),
            int(width * 0.92),
            int(height * 0.92),
        )
    elif kind == "shorts":
        rect = (
            int(width * 0.08),
            int(height * 0.18),
            int(width * 0.92),
            int(height * 0.82),
        )
    else:
        rect = (
            int(width * 0.08),
            int(height * 0.08),
            int(width * 0.92),
            int(height * 0.92),
        )
    draw.rounded_rectangle(rect, radius=max(12, min(size) // 12), fill=255)
    return mask.filter(ImageFilter.GaussianBlur(feather))


def natural_background_extend(
    img: Image.Image,
    size: tuple[int, int],
    preset: Any,
    kind: str,
    anchor: str = "center",
    offset_x: float = 0.0,
    offset_y: float = 0.0,
    subject_scale: float = 1.0,
) -> Image.Image:
    target_width, target_height = size
    scale = _target_fill_scale(img, size, kind, subject_scale)
    fg_width = max(1, int(img.width * scale))
    fg_height = max(1, int(img.height * scale))
    fg = img.resize((fg_width, fg_height), Image.Resampling.LANCZOS)
    x, y = _foreground_position(
        fg.size,
        size,
        kind,
        anchor,
        getattr(preset, "text_safe_ratio_16x9", 0.38),
        offset_x,
        offset_y,
    )

    if fg_width >= target_width and fg_height >= target_height:
        return make_smart_crop(
            img,
            size,
            kind=kind,
            anchor=anchor,
            offset_x=offset_x,
            offset_y=offset_y,
            subject_scale=subject_scale,
        )

    base = _cover_background(img, size)
    _paste_axis_extensions(base, fg, x, y)

    crop_left = max(0, -x)
    crop_top = max(0, -y)
    crop_right = min(fg_width, target_width - x)
    crop_bottom = min(fg_height, target_height - y)
    if crop_right > crop_left and crop_bottom > crop_top:
        visible = fg.crop((crop_left, crop_top, crop_right, crop_bottom))
        paste_x = max(0, x)
        paste_y = max(0, y)
        feather = max(10, int(min(size) * 0.018))
        base.paste(visible, (paste_x, paste_y), _feather_mask(visible.size, feather))

    return base.convert("RGB")


def make_smart_crop(
    img: Image.Image,
    size: tuple[int, int],
    kind: str = "thumbnail",
    anchor: str = "center",
    offset_x: float = 0.0,
    offset_y: float = 0.0,
    subject_scale: float = 1.0,
) -> Image.Image:
    center_x = 0.5 + max(-0.35, min(0.35, offset_x)) * 0.5
    center_y = 0.5 + max(-0.35, min(0.35, offset_y)) * 0.5
    if kind == "thumbnail" and anchor == "right":
        center_x = min(0.76, center_x + 0.18)
    scale = max(1.0, min(1.8, subject_scale))
    work_size = (max(1, int(img.width / scale)), max(1, int(img.height / scale)))
    left = max(0, min(img.width - work_size[0], int(img.width * center_x - work_size[0] / 2)))
    top = max(0, min(img.height - work_size[1], int(img.height * center_y - work_size[1] / 2)))
    crop = img.crop((left, top, left + work_size[0], top + work_size[1]))
    return ImageOps.fit(crop, size, method=Image.Resampling.LANCZOS, centering=(center_x, center_y))


def make_fit_original(img: Image.Image, size: tuple[int, int]) -> Image.Image:
    bg = Image.new("RGB", size, (18, 18, 18))
    fg = ImageOps.contain(img, size, method=Image.Resampling.LANCZOS)
    x = (size[0] - fg.width) // 2
    y = (size[1] - fg.height) // 2
    bg.paste(fg, (x, y))
    return bg


def render_full_frame_format(
    img: Image.Image,
    size: tuple[int, int],
    preset: Any,
    kind: str,
    mode: str = "ai_natural",
    anchor: str = "center",
    offset_x: float = 0.0,
    offset_y: float = 0.0,
    subject_scale: float = 1.0,
) -> tuple[Image.Image, str]:
    """Render one output format using the same geometry as the preview and fallback."""
    normalized = img.convert("RGB")
    if mode == "smart_crop":
        return make_smart_crop(
            normalized,
            size,
            kind=kind,
            anchor=anchor,
            offset_x=offset_x,
            offset_y=offset_y,
            subject_scale=subject_scale,
        ), "Smart crop"
    if mode == "fit":
        return make_fit_original(normalized, size), "Original fit"
    if mode == "blur":
        if kind == "thumbnail":
            return make_text_safe_landscape(normalized, preset, size), "Blur Canvas"
        return make_shorts(normalized, preset), "Blur Canvas"
    if mode == "ai_natural":
        raise RuntimeError("AI 자연 배경 확장 결과가 없습니다. 먼저 변환을 실행하거나 로컬 방식을 선택하세요.")
    return natural_background_extend(
        normalized,
        size,
        preset,
        kind,
        anchor=anchor,
        offset_x=offset_x,
        offset_y=offset_y,
        subject_scale=subject_scale,
    ), "Natural edge extension"


def build_full_frame_outpaint_canvas(
    img: Image.Image,
    size: tuple[int, int],
    preset: Any,
    kind: str,
    person_mask: Image.Image | None = None,
    anchor: str = "center",
    offset_x: float = 0.0,
    offset_y: float = 0.0,
    subject_scale: float = 1.0,
    protect_core: bool = True,
) -> tuple[Image.Image, Image.Image, Image.Image]:
    target_width, target_height = size
    scale = _target_fill_scale(img, size, kind, subject_scale)
    fg_width = max(1, int(img.width * scale))
    fg_height = max(1, int(img.height * scale))
    fg = img.resize((fg_width, fg_height), Image.Resampling.LANCZOS)
    x, y = _foreground_position(
        fg.size,
        size,
        kind,
        anchor,
        getattr(preset, "text_safe_ratio_16x9", 0.38),
        offset_x,
        offset_y,
    )

    base = natural_background_extend(
        img,
        size,
        preset,
        kind,
        anchor=anchor,
        offset_x=offset_x,
        offset_y=offset_y,
        subject_scale=subject_scale,
    )
    gen_mask = Image.new("L", size, 255)
    crop_left = max(0, -x)
    crop_top = max(0, -y)
    crop_right = min(fg_width, target_width - x)
    crop_bottom = min(fg_height, target_height - y)
    paste_x = max(0, x)
    paste_y = max(0, y)
    visible_width = max(0, crop_right - crop_left)
    visible_height = max(0, crop_bottom - crop_top)

    protect = Image.new("RGBA", size, (0, 0, 0, 0))
    if visible_width and visible_height:
        core_mask = (
            core_protection_mask((visible_width, visible_height), kind)
            if protect_core
            else Image.new("L", (visible_width, visible_height), 0)
        )
        if person_mask is not None:
            resized_person = person_mask.convert("L").resize((fg_width, fg_height), Image.Resampling.LANCZOS)
            visible_person = resized_person.crop((crop_left, crop_top, crop_right, crop_bottom))
            core_mask = ImageChops.lighter(core_mask, visible_person)

        visible_fg = fg.crop((crop_left, crop_top, crop_right, crop_bottom)).convert("RGBA")
        protect.paste(visible_fg, (paste_x, paste_y), core_mask)

        safe = max(10, int(min(size) * 0.018))
        protected_rect = (
            paste_x + safe,
            paste_y + safe,
            paste_x + visible_width - safe,
            paste_y + visible_height - safe,
        )
        if protected_rect[2] > protected_rect[0] and protected_rect[3] > protected_rect[1]:
            gen_mask.paste(0, protected_rect)
        gen_mask = gen_mask.filter(ImageFilter.GaussianBlur(max(5, safe // 2)))
    return base, gen_mask, protect


def build_outpaint_canvas(
    img: Image.Image,
    size: tuple[int, int],
    person_mask: Image.Image | None = None,
    anchor: str = "center",
    margin: float = 0.04,
) -> tuple[Image.Image, Image.Image, Image.Image]:
    """Create an AI canvas, generation mask, and protected-pixel overlay."""
    target_width, target_height = size
    scale = min(
        (target_width * (1 - 2 * margin)) / img.width,
        (target_height * (1 - 2 * margin)) / img.height,
    )
    fg_width = max(1, int(img.width * scale))
    fg_height = max(1, int(img.height * scale))
    fg = img.resize((fg_width, fg_height), Image.Resampling.LANCZOS)

    if anchor == "right" and target_width > target_height:
        x = target_width - fg_width - int(target_width * margin)
    elif anchor == "left" and target_width > target_height:
        x = int(target_width * margin)
    else:
        x = (target_width - fg_width) // 2
    y = (target_height - fg_height) // 2

    base = blurred_background(img, size, blur=max(20, int(min(size) * 0.025)))
    base.paste(fg, (x, y))

    gen_mask = Image.new("L", size, 255)
    inset = max(8, int(min(size) * 0.012))
    gen_mask.paste(0, (x + inset, y + inset, x + fg_width - inset, y + fg_height - inset))
    gen_mask = gen_mask.filter(ImageFilter.GaussianBlur(max(4, inset // 2)))

    protect = Image.new("RGBA", size, (0, 0, 0, 0))
    if person_mask is not None:
        mask = person_mask.convert("L").resize((fg_width, fg_height), Image.Resampling.LANCZOS)
    else:
        mask = Image.new("L", (fg_width, fg_height), 255)
    protect.paste(fg.convert("RGBA"), (x, y), mask)
    return base, gen_mask, protect


def restore_protected_pixels(generated: Image.Image, protect: Image.Image) -> Image.Image:
    if protect.size != generated.size:
        protect = protect.resize(generated.size, Image.Resampling.NEAREST)
    out = generated.convert("RGBA")
    out.alpha_composite(protect.convert("RGBA"))
    return out.convert("RGB")


def crop_subject_region(img: Image.Image) -> Image.Image:
    face = detect_largest_face(img)
    if not face:
        return img.copy()
    x1, y1, x2, y2 = face
    face_width = x2 - x1
    face_height = y2 - y1
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    left = max(0, int(center_x - face_width * 3.1))
    right = min(img.width, int(center_x + face_width * 3.1))
    top = max(0, int(center_y - face_height * 2.35))
    bottom = min(img.height, int(center_y + face_height * 4.8))
    if right - left < img.width * 0.45:
        return img.copy()
    return img.crop((left, top, right, bottom))


def paste_subject_on_canvas(
    img: Image.Image,
    target_size: tuple[int, int],
    anchor: str = "right",
    subject_ratio: float = 0.42,
    safe_ratio: float = 0.35,
) -> Image.Image:
    target_width, target_height = target_size
    bg = blurred_background(
        img,
        (target_width, target_height),
        blur=max(24, int(min(target_width, target_height) * 0.028)),
    )
    subject = crop_subject_region(img)

    if target_width > target_height:
        max_width = int(target_width * subject_ratio)
        max_height = int(target_height * 0.96)
    else:
        max_width = int(target_width * 0.94)
        max_height = int(target_height * subject_ratio)

    scale = min(max_width / subject.width, max_height / subject.height)
    fg_width = max(1, int(subject.width * scale))
    fg_height = max(1, int(subject.height * scale))
    fg = subject.resize((fg_width, fg_height), Image.Resampling.LANCZOS)

    if target_width > target_height:
        if anchor == "right":
            x = max(int(target_width * safe_ratio), target_width - int(target_width * 0.035) - fg_width)
        elif anchor == "left":
            x = int(target_width * 0.035)
        else:
            x = (target_width - fg_width) // 2
        y = (target_height - fg_height) // 2
    else:
        x = (target_width - fg_width) // 2
        y = max(int(target_height * 0.05), (target_height - fg_height) // 2)

    mask = Image.new("L", (fg_width, fg_height), 255).filter(
        ImageFilter.GaussianBlur(max(6, int(min(fg_width, fg_height) * 0.018)))
    )
    bg.paste(fg, (x, y), mask)
    return bg


def make_text_safe_landscape(
    img: Image.Image,
    preset: Any,
    size: tuple[int, int] = (1920, 1080),
) -> Image.Image:
    out = paste_subject_on_canvas(
        img,
        size,
        anchor=preset.person_anchor_16x9,
        subject_ratio=preset.person_ratio_16x9,
        safe_ratio=preset.text_safe_ratio_16x9,
    )
    safe_width = int(out.width * preset.text_safe_ratio_16x9)
    zone = out.crop((0, 0, safe_width, out.height)).filter(ImageFilter.GaussianBlur(3))
    zone = ImageEnhance.Contrast(zone).enhance(0.88)
    out.paste(zone, (0, 0))
    return out


def make_shorts(img: Image.Image, preset: Any) -> Image.Image:
    return paste_subject_on_canvas(
        img,
        (1080, 1920),
        anchor=preset.person_anchor_9x16,
        subject_ratio=preset.person_ratio_9x16,
        safe_ratio=0.0,
    )


def make_square(img: Image.Image) -> Image.Image:
    return ImageOps.fit(img, (1400, 1400), method=Image.Resampling.LANCZOS)


def save_jpg(img: Image.Image, path: Path, quality: int = 98) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img.convert("RGB").save(
        path,
        "JPEG",
        quality=max(98, quality),
        subsampling=0,
        progressive=True,
        optimize=True,
    )

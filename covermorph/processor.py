from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

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
    min_conf: float = 0.22,
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
    return Image.fromarray(mask_from_boxes(size, boxes)).convert("L")


def inpaint_text_opencv(img: Image.Image, boxes: Sequence[Rect], radius: int = 7) -> Image.Image:
    if not boxes:
        return img.copy()
    arr = pil_to_cv(img)
    mask = mask_from_boxes(img.size, boxes, dilation=9)
    result = cv2.inpaint(arr, mask, radius, cv2.INPAINT_TELEA)
    return cv_to_pil(result)


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


def make_text_safe_landscape(img: Image.Image, preset: Any) -> Image.Image:
    out = paste_subject_on_canvas(
        img,
        (1920, 1080),
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


def save_jpg(img: Image.Image, path: Path, quality: int = 96) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img.convert("RGB").save(path, "JPEG", quality=quality, subsampling=0, optimize=True)

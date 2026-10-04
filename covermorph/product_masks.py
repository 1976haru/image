"""Product masks for 원본 상품 보존: automatic mask, simple corrections, confidence, persistence.

A mask is an 8-bit "L" image the size of the reference photo (255 = product). The automatic mask comes from
``product_checks.auto_mask`` (alpha channel, or GrabCut seeded from the backdrop color). Corrections are plain
image operations (invert, expand/contract, feather, brush strokes); nothing generative touches the product.
Corrected masks are stored per reference file in ``%LOCALAPPDATA%\\CoverMorphStudio\\masks`` and copied into
each job folder, so a job always records the exact mask it used.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

LOW_CONTRAST_DELTA_E = 6.0      # boundary pixels whose inside/outside colors differ less than this are "unsure"
UNCERTAIN_FRACTION = 0.12       # more unsure boundary than this -> ask the user to check the mask


@dataclass(slots=True)
class MaskReport:
    coverage: float              # product share of the image
    unsure_edges: float          # share of the outline with almost no color contrast
    holes: int                   # see-through openings (handles, leaf splits)
    uncertain: bool
    message: str


def mask_key(reference: Path) -> str:
    stat = Path(reference).stat()
    return hashlib.sha1(f"{Path(reference).resolve()}|{stat.st_size}|{int(stat.st_mtime)}".encode("utf-8")).hexdigest()[:16]


def stored_mask_path(reference: Path) -> Path:
    from .app_paths import data_dir
    folder = data_dir() / "masks"
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f"{mask_key(reference)}.png"


def save_mask(reference: Path, mask: Image.Image) -> Path:
    path = stored_mask_path(reference)
    mask.convert("L").save(path)
    return path


def load_mask(path: str | Path | None, size: tuple[int, int]) -> Image.Image | None:
    if not path or not Path(path).exists():
        return None
    with Image.open(path) as opened:
        mask = opened.convert("L")
    return mask if mask.size == size else mask.resize(size, Image.Resampling.NEAREST)


def refine(mask: Image.Image, *, invert: bool = False, grow: int = 0, feather: float = 0.0) -> Image.Image:
    """grow > 0 expands, grow < 0 contracts (pixels); feather softens only the edge."""
    array = np.array(mask.convert("L"))
    if invert:
        array = 255 - array
    if grow:
        kernel = np.ones((abs(grow) * 2 + 1, abs(grow) * 2 + 1), np.uint8)
        array = cv2.dilate(array, kernel) if grow > 0 else cv2.erode(array, kernel)
    out = Image.fromarray(array, "L")
    return out.filter(ImageFilter.GaussianBlur(feather)) if feather > 0 else out


def paint(mask: Image.Image, strokes: list[tuple[float, float, float, bool]]) -> Image.Image:
    """Brush strokes in image pixels: (x, y, radius, add)."""
    out = mask.convert("L").copy()
    draw = ImageDraw.Draw(out)
    for x, y, radius, add in strokes:
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=255 if add else 0)
    return out


def assess(image: Image.Image, mask: Image.Image) -> MaskReport:
    """How sure is the outline? Low inside/outside color contrast along the edge means "check this mask"."""
    rgb = np.array(image.convert("RGB"))
    binary = (np.array(mask.convert("L")) > 127).astype(np.uint8)
    coverage = float(binary.mean())
    if coverage < 0.005 or coverage > 0.97:
        return MaskReport(coverage, 1.0, 0, True, "상품 영역을 거의 찾지 못했습니다. 마스크를 직접 칠해 주세요.")
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32) * np.array([100 / 255, 1, 1], np.float32)
    inner = cv2.erode(binary, np.ones((7, 7), np.uint8))
    outer = cv2.dilate(binary, np.ones((7, 7), np.uint8))
    edge = (binary - cv2.erode(binary, np.ones((3, 3), np.uint8))) > 0
    # local mean colors just inside / just outside the outline
    k = (15, 15)
    inside = cv2.blur(lab * inner[..., None], k) / np.maximum(cv2.blur(inner.astype(np.float32), k), 1e-3)[..., None]
    ring = (outer - binary).astype(np.uint8)
    outside = cv2.blur(lab * ring[..., None], k) / np.maximum(cv2.blur(ring.astype(np.float32), k), 1e-3)[..., None]
    delta = np.linalg.norm(inside - outside, axis=2)[edge]
    unsure = float((delta < LOW_CONTRAST_DELTA_E).mean()) if delta.size else 1.0
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    holes = 0 if hierarchy is None else sum(1 for c, h in zip(contours, hierarchy[0])
                                            if h[3] >= 0 and cv2.contourArea(c) > 0.003 * binary.sum())
    uncertain = unsure > UNCERTAIN_FRACTION
    message = (f"윤곽의 {unsure:.0%}가 배경과 색이 거의 같습니다. 마스크 미리보기에서 빠진 부분이 없는지 확인하세요."
               if uncertain else "자동 마스크가 배경과 잘 구분됩니다.")
    return MaskReport(round(coverage, 4), round(unsure, 3), holes, uncertain, message)


def cutout(image: Image.Image, mask: Image.Image) -> Image.Image:
    """RGBA product cut out with the given mask, cropped to the product. RGB pixels are untouched."""
    rgba = image.convert("RGB").copy()
    rgba.putalpha(mask.convert("L").resize(image.size))
    box = mask.convert("L").point(lambda v: 255 if v > 16 else 0).getbbox()
    return rgba.crop(box) if box else rgba


def preview(image: Image.Image, mask: Image.Image, size: tuple[int, int] = (520, 520)) -> Image.Image:
    """Product over a checkerboard with the removed area tinted red, for the mask window."""
    base = image.convert("RGB")
    tint = Image.new("RGB", base.size, (220, 40, 60))
    shown = Image.composite(base, Image.blend(base, tint, 0.55), mask.convert("L"))
    shown.thumbnail(size, Image.Resampling.LANCZOS)
    return shown

"""상품 형태 보존: product cutout, exact-pixel composite, and distortion checks on generated candidates.

No diffusion model can guarantee an exact product, so preservation mode offers two kinds of candidates:
  * composite — the reference product's own pixels pasted onto a generated background (exact by construction;
    lighting integration is approximate: contact shadow only, no relighting)
  * generated — FLUX.2-klein with the PRODUCT reference (natural lighting, may distort), checked here.
Automatic checks on generated candidates are limited to what proved reliable: Lab color drift of the
product-colored region and an OCR text/logo comparison (only with locally installed EasyOCR weights).
A silhouette/hole-count check was tried on 8 FLUX.2 mug results (1 vs duplicated handle) and could not
separate them, so it is not used: every regenerated product is labelled "check the shape" instead and is
never ranked above an exact composite.
"""
from __future__ import annotations

import json

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

COLOR_DRIFT_CEILING = 14.0       # ΔE (CIE76) between median product colors
MIN_PRODUCT_AREA = 0.004         # product-colored region must cover this fraction of the candidate
REGENERATED_WARNING = "AI가 다시 그린 상품 — 형태·부품 수·로고를 참조와 직접 비교하세요"


# ------------------------------------------------------------------ cutout
def extract_product(reference: Image.Image, mask: Image.Image | None = None) -> Image.Image:
    """RGBA cutout cropped to the product, from a given (user-corrected) mask or the automatic one."""
    from .product_masks import cutout
    return cutout(reference, mask if mask is not None else auto_mask(reference))


def auto_mask(reference: Image.Image) -> Image.Image:
    """Full-size "L" mask (255 = product). Alpha channel when present, else GrabCut seeded from the backdrop."""
    if reference.mode in ("RGBA", "LA") or "transparency" in reference.info:
        alpha = reference.convert("RGBA").getchannel("A")
        if (np.array(alpha) < 250).mean() > 0.02:
            return alpha
    rgb = np.array(reference.convert("RGB"))
    height, width = rgb.shape[:2]
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    band = max(4, int(min(height, width) * 0.03))
    border = np.concatenate([lab[:band].reshape(-1, 3), lab[-band:].reshape(-1, 3), lab[:, :band].reshape(-1, 3),
                             lab[:, -band:].reshape(-1, 3)])
    background = np.median(border, axis=0)
    chroma = np.linalg.norm(lab[..., 1:] - background[1:], axis=2)
    lighter = lab[..., 0] - background[0]
    # Shadows on a studio backdrop keep its chroma and only get darker: they are background, not product.
    mask = np.full((height, width), cv2.GC_PR_BGD, np.uint8)
    probable = (chroma > 8) | (lighter > 25) | (lighter < -70)
    sure = (chroma > 20) | (lighter > 45)
    mask[probable] = cv2.GC_PR_FGD
    mask[cv2.erode(sure.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0] = cv2.GC_FGD
    mask[:band], mask[-band:], mask[:, :band], mask[:, -band:] = (cv2.GC_BGD,) * 4
    if (mask == cv2.GC_FGD).sum() + (mask == cv2.GC_PR_FGD).sum() < 50:
        raise ValueError("상품을 배경과 구분하지 못했습니다. 배경이 투명한 PNG나 단색 배경 사진을 사용하세요.")
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    cv2.grabCut(rgb, mask, None, bgd, fgd, 6, cv2.GC_INIT_WITH_MASK)
    fg = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(fg)
    if count <= 1:
        raise ValueError("상품 영역을 찾지 못했습니다.")
    keep = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    # keep the main part plus sizeable pieces (e.g. a handle separated by a thin gap)
    big = [i for i in range(1, count) if stats[i, cv2.CC_STAT_AREA] >= 0.05 * stats[keep, cv2.CC_STAT_AREA]]
    fg = np.isin(labels, big).astype(np.uint8) * 255
    fg = _fill_logo_holes(fg, lab, background)
    return Image.fromarray(cv2.GaussianBlur(fg, (3, 3), 0), "L")


def _fill_logo_holes(fg: np.ndarray, lab: np.ndarray, background: np.ndarray) -> np.ndarray:
    """Re-fill enclosed holes that are part of the product (logos, prints), keep see-through ones (handles).

    A see-through hole shows the backdrop's color; a logo/print does not, or is tiny.
    """
    contours, hierarchy = cv2.findContours((fg > 0).astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return fg
    area = max(1, int((fg > 0).sum()))
    out = fg.copy()
    for contour, info in zip(contours, hierarchy[0]):
        if info[3] < 0:
            continue
        hole = np.zeros(fg.shape, np.uint8)
        cv2.drawContours(hole, [contour], -1, 1, thickness=-1)
        inside = (hole > 0) & (fg == 0)
        if not inside.any():
            continue
        delta = float(np.linalg.norm(np.median(lab[inside], axis=0) - background))
        if delta > 12 or inside.sum() < 0.025 * area:
            out[hole > 0] = 255
    return out


# ------------------------------------------------------------------ composite
SURFACE_MARGIN = 0.035   # product base this far (of image height) above the surface's front edge


def find_surface(background: Image.Image, region: tuple[float, float, float, float]) -> float | None:
    """Height (0..1) of the supporting surface's front edge under the subject region, or None.

    A front edge is a long horizontal transition that gets darker going down (lit top surface -> front face),
    strongest in the subject's column. Measured on generated table/counter/desk/windowsill scenes (2026-10-04);
    the back edge of a table goes dark -> bright and is ignored.
    """
    gray = np.array(background.convert("L"), np.float32)
    height, width = gray.shape
    x0, x1 = int(region[0] * width), max(int(region[0] * width) + 8, int((region[0] + region[2]) * width))
    band = cv2.GaussianBlur(gray[:, x0:x1], (0, 0), 3)
    dy = cv2.Sobel(band, cv2.CV_32F, 0, 1, ksize=5)
    score = np.abs(dy).mean(axis=1) * np.clip(-np.sign(dy).mean(axis=1), 0, None)
    score[: int(0.30 * height)] = 0
    score[int(0.97 * height):] = 0
    score = np.convolve(score, np.ones(9) / 9, mode="same")
    y = int(score.argmax())
    if score[y] < 25:   # no clear surface edge (soft/flat backgrounds)
        return None
    return y / height


@dataclass(slots=True)
class Placement:
    baseline: float      # product bottom, fraction of image height
    center_x: float      # product center, fraction of image width
    scale: float         # product height as a fraction of image height

    def to_dict(self) -> dict[str, float]:
        return {"baseline": round(self.baseline, 4), "center_x": round(self.center_x, 4), "scale": round(self.scale, 4)}


def auto_placement(background: Image.Image, region: tuple[float, float, float, float], fill: float = 0.5) -> Placement:
    rx, ry, rw, rh = region
    surface = find_surface(background, region)
    baseline = (surface - SURFACE_MARGIN) if surface is not None else ry + rh * 0.94
    baseline = min(0.97, max(ry + 0.2 * rh, baseline))
    return Placement(baseline, rx + rw / 2, rh * fill)


def _place(canvas_size: tuple[int, int], cutout: Image.Image, placement: Placement) -> tuple[tuple[int, int], int, int]:
    width, height = canvas_size
    target_h = max(1.0, placement.scale * height)
    target_h = min(target_h, placement.baseline * height * 0.98)       # never past the top edge
    scale = target_h / cutout.height
    size = (max(1, round(cutout.width * scale)), max(1, round(cutout.height * scale)))
    if size[0] > width * 0.95:
        factor = width * 0.95 / size[0]
        size = (max(1, round(size[0] * factor)), max(1, round(size[1] * factor)))
    left = round(placement.center_x * width - size[0] / 2)
    left = min(max(0, left), width - size[0])
    bottom = round(placement.baseline * height)
    return size, left, bottom


def composite_product(background: Image.Image, cutout: Image.Image,
                      subject_region: tuple[float, float, float, float], fill: float = 0.5,
                      shadow: bool = True, placement: Placement | None = None) -> tuple[Image.Image, list[float]]:
    """원본 그대로 합성: the product's own pixels (only scaled) standing on the detected surface, contact shadow."""
    canvas = background.convert("RGB").copy()
    placement = placement or auto_placement(canvas, subject_region, fill)
    size, left, bottom = _place(canvas.size, cutout, placement)
    width, height = canvas.size
    product = cutout.resize(size, Image.Resampling.LANCZOS)
    top = bottom - size[1]
    if shadow:
        contact = Image.new("L", canvas.size, 0)
        ellipse = Image.new("L", (size[0], max(8, size[1] // 7)), 0)
        ImageDraw.Draw(ellipse).ellipse((size[0] * 0.06, 0, size[0] * 0.94, ellipse.height - 1), fill=150)
        contact.paste(ellipse, (left, bottom - ellipse.height // 2))
        contact = contact.filter(ImageFilter.GaussianBlur(max(4, size[0] // 18)))
        canvas = Image.composite(Image.new("RGB", canvas.size, (0, 0, 0)), canvas, contact.point(lambda v: v * 0.55))
    canvas.paste(product, (left, top), product)
    return canvas, [left / width, top / height, size[0] / width, size[1] / height]


def natural_composite(background: Image.Image, cutout: Image.Image, subject_region: tuple[float, float, float, float],
                      fill: float = 0.5, strength: float = 0.15, placement: Placement | None = None) -> tuple[Image.Image, list[float]]:
    """자연광 보정 합성: the BACKGROUND's brightness moves a little toward the product's (no hue shift: grading toward
    the product's color tinted whole scenes), plus a soft shadow falling away from the brighter side. The product's
    pixels are never changed (only scaled), so logo/text/geometry stay exactly as photographed."""
    base = background.convert("RGB")
    placement = placement or auto_placement(base, subject_region, fill)
    alpha = np.array(cutout.getchannel("A")) > 128
    product_l = np.array(cutout.convert("L"), np.float32)[alpha]
    lab = cv2.cvtColor(np.array(base), cv2.COLOR_RGB2LAB).astype(np.float32)
    if product_l.size:
        lab[..., 0] = np.clip(lab[..., 0] + (product_l.mean() - lab[..., 0].mean()) * strength, 0, 255)
    graded = Image.fromarray(cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB))
    gray = np.array(base.convert("L"), np.float32)
    lean = 1 if gray[:, : gray.shape[1] // 2].mean() > gray[:, gray.shape[1] // 2:].mean() else -1
    size, left, bottom = _place(graded.size, cutout, placement)
    shadow = Image.new("L", graded.size, 0)
    draw = ImageDraw.Draw(shadow)
    draw.ellipse((left + size[0] * 0.05, bottom - size[1] * 0.05, left + size[0] * 0.95, bottom + size[1] * 0.03), fill=170)
    offset = int(size[0] * 0.35) * lean
    draw.polygon([(left + size[0] * 0.15, bottom), (left + size[0] * 0.85, bottom),
                  (left + size[0] * 0.85 + offset, bottom - size[1] * 0.08),
                  (left + size[0] * 0.15 + offset, bottom - size[1] * 0.08)], fill=70)
    shadow = shadow.filter(ImageFilter.GaussianBlur(max(5, size[0] // 14)))
    graded = Image.composite(Image.new("RGB", graded.size, (0, 0, 0)), graded, shadow.point(lambda v: v * 0.5))
    return composite_product(graded, cutout, subject_region, fill, shadow=False, placement=placement)


# ------------------------------------------------------------------ checks
@dataclass(slots=True)
class ProductCheck:
    located: bool
    color_drift: float | None = None
    text_reference: str = ""
    text_candidate: str = ""
    box: list[float] | None = None
    exact: bool = False                  # True only for composites (reference pixels)
    warnings: list[str] = field(default_factory=list)

    @property
    def distorted(self) -> bool:
        return not self.exact

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "distorted": self.distorted}


def _median_lab(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab = lab * np.array([100 / 255, 1, 1], np.float32) - np.array([0, 128, 128], np.float32)
    return np.median(lab[mask > 0], axis=0)


def locate_by_color(cutout: Image.Image, candidate: Image.Image) -> tuple[np.ndarray | None, list[float] | None]:
    """Largest region whose chroma matches the product's (a/b histogram back-projection). Mask at candidate size."""
    ref = np.array(cutout.convert("RGB"))
    ref_mask = (np.array(cutout.getchannel("A")) > 128).astype(np.uint8) if cutout.mode == "RGBA" else None
    hist = cv2.calcHist([cv2.cvtColor(ref, cv2.COLOR_RGB2LAB)], [1, 2], ref_mask, [32, 32], [0, 256, 0, 256])
    cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
    rgb = np.array(candidate.convert("RGB"))
    projection = cv2.calcBackProject([cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)], [1, 2], hist, [0, 256, 0, 256], 1)
    projection = cv2.GaussianBlur(projection, (9, 9), 0)
    _, binary = cv2.threshold(projection, 30, 255, cv2.THRESH_BINARY)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
    if count <= 1:
        return None, None
    index = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if stats[index, cv2.CC_STAT_AREA] < MIN_PRODUCT_AREA * rgb.shape[0] * rgb.shape[1]:
        return None, None
    x, y, w, h = stats[index, :4]
    height, width = rgb.shape[:2]
    pad_x, pad_y = int(w * 0.5), int(h * 0.2)
    box = [max(0, x - pad_x) / width, max(0, y - pad_y) / height,
           min(width - max(0, x - pad_x), w + 2 * pad_x) / width, min(height - max(0, y - pad_y), h + 2 * pad_y) / height]
    return (labels == index).astype(np.uint8), box


def check_product(cutout: Image.Image, candidate: Image.Image) -> ProductCheck:
    """Checks for a regenerated product. Always carries REGENERATED_WARNING; adds color/text warnings."""
    mask, box = locate_by_color(cutout, candidate)
    check = ProductCheck(mask is not None, box=box, warnings=[REGENERATED_WARNING])
    if mask is None:
        check.warnings.append("상품 색상 영역을 찾지 못했습니다 — 색상이 바뀌었거나 상품이 빠졌을 수 있습니다")
        return check
    ref = np.array(cutout.convert("RGB"))
    ref_mask = (np.array(cutout.getchannel("A")) > 128).astype(np.uint8) if cutout.mode == "RGBA" else np.ones(ref.shape[:2], np.uint8)
    check.color_drift = round(float(np.linalg.norm(_median_lab(np.array(candidate.convert("RGB")), mask)
                                                   - _median_lab(ref, ref_mask))), 2)
    if check.color_drift > COLOR_DRIFT_CEILING:
        check.warnings.append(f"상품 색상이 참조와 다릅니다(ΔE {check.color_drift:.1f})")
    return check


def composite_check(box: list[float]) -> ProductCheck:
    return ProductCheck(True, color_drift=0.0, box=box, exact=True,
                        warnings=[] if box else ["상품 배치 영역을 확인하세요"])


def _norm(text: str) -> str:
    return "".join(ch for ch in text.casefold() if ch.isalnum())


def ocr_texts(paths: list[Path], timeout: float = 240.0) -> list[str] | None:
    """OCR in a separate process (memory returns when it exits). None when unavailable or failing."""
    import subprocess
    import sys

    from .ocr_worker import weights_installed
    if not paths or not weights_installed():
        return None
    if getattr(sys, "frozen", False):
        command = [sys.executable, "--ocr-worker", *map(str, paths)]
    else:
        command = [sys.executable, "-m", "covermorph.ocr_worker", *map(str, paths)]
    try:
        done = subprocess.run(command, capture_output=True, timeout=timeout, stdin=subprocess.DEVNULL,
                              cwd=str(Path(__file__).resolve().parents[1]),
                              creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        texts = json.loads(done.stdout.decode("utf-8", errors="replace").strip().splitlines()[-1])
        return texts if isinstance(texts, list) and len(texts) == len(paths) else None
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def apply_ocr(cutout: Image.Image, items: list[tuple[Image.Image, ProductCheck]], workdir: Path) -> bool:
    """Compare legible product text/logo between the reference and each candidate (one OCR process)."""
    workdir.mkdir(parents=True, exist_ok=True)
    paths = [workdir / "ocr_ref.png"]
    cutout.convert("RGB").save(paths[0])
    located = [(image, check) for image, check in items if check.box]
    for index, (image, check) in enumerate(located):
        x, y, w, h = check.box
        path = workdir / f"ocr_{index}.png"
        image.crop((int(x * image.width), int(y * image.height), int((x + w) * image.width),
                    int((y + h) * image.height))).save(path)
        paths.append(path)
    texts = ocr_texts(paths)
    if texts is None:
        return False
    reference = texts[0]
    for (image, check), text in zip(located, texts[1:]):
        check.text_reference, check.text_candidate = reference, text
        if reference and _norm(reference) != _norm(text):
            check.warnings.append(f"상품 글자/로고가 다릅니다: '{reference}' → '{text or '없음'}'")
    return True


def product_crop(candidate: Image.Image, check: ProductCheck, size: int = 320) -> Image.Image | None:
    if not check.box:
        return None
    x, y, w, h = check.box
    box = (int(x * candidate.width), int(y * candidate.height), int((x + w) * candidate.width), int((y + h) * candidate.height))
    crop = candidate.crop(box)
    crop.thumbnail((size, size), Image.Resampling.LANCZOS)
    return crop

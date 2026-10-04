"""AI 이미지 스튜디오 jobs: payload -> candidates on disk -> export. No GUI code here.

A job payload (stored in the persistent queue) looks like::

    {"kind": "generate"|"edit", "purpose": "shopify_hero", "canvas": [1800, 700], "prompt": "...",
     "prompt_preset": "tc_solo_woman", "channel": "Tokyo Chill", "people": 1, "composition": "SOLO_MEDIUM",
     "references": [{"path": "...", "role": "PRODUCT", "mask": "<optional corrected mask png>"}],
     "product_mode": "strict"|"natural"|"ai"|"none", "product_scale": 0.5,
     "quality": "balanced", "memory": "interactive_low_memory", "seed": 123, "candidates": 2,
     "edit_image": "...", "edit_instruction": "...", "title": "...", "subtitle": "...", "cta": "..."}

Generation always goes through ``quality_modes.run_quality_job`` (Z-Image / FLUX.2 / RealVis fallback,
one engine at a time, unloaded afterwards). Product preservation adds exact-pixel composites.
"""
from __future__ import annotations

import json
import re
import shutil
import time
from datetime import datetime
from pathlib import Path
from threading import Event
from typing import Any, Callable

from PIL import Image, ImageDraw, ImageFont

from .creator_presets import BUILTIN_PROMPT_PRESETS, Purpose, apply_appearance_hint, load_presets
from .person_quality import YUNET_FILENAME, detect_faces_yunet
from .product_checks import apply_ocr, check_product, composite_check, composite_product, extract_product
from .quality_engines import EngineCancelled, resource_snapshot
from .quality_modes import run_quality_job

Progress = Callable[[dict[str, Any]], None]
_BAD_NAME = re.compile(r'[<>:"/\\|?*\x00-\x1f]+')
COMPOSITE_HARMONIZE = ("Keep the product and the whole scene exactly as they are, with the same shape, parts, colors "
                       "and logo; only make the lighting, shadow and reflections on the product match the scene "
                       "naturally")


def safe_name(text: str, limit: int = 40) -> str:
    """Windows-safe file name part; Unicode (Korean/Japanese) is kept."""
    name = _BAD_NAME.sub(" ", text or "").strip().strip(".")
    name = re.sub(r"\s+", "_", name)[:limit].rstrip("._")
    return name or "image"


def _region_words(region: tuple[float, float, float, float]) -> str:
    cx, cy = region[0] + region[2] / 2, region[1] + region[3] / 2
    horizontal = "left" if cx < 0.4 else "right" if cx > 0.6 else "center"
    vertical = "lower " if cy > 0.55 else ""
    return f"{vertical}{horizontal}".strip()


def background_prompt(prompt: str, region: tuple[float, float, float, float]) -> str:
    """Scene prompt for a composite: the real product is pasted later, so the scene must not contain one.

    Positive wording only (measured): "product photography background plate" produced cameras and white boards,
    "nothing stands on the surface" still produced bottles; "a wide, clear stretch of bare surface" gave 6/6 bare tables.
    """
    scene = re.sub(r"\b(the|a|an|our|this)\s+product\s+(on|in|at|by|under|beside|next to)\b", r"\2", prompt,
                   flags=re.IGNORECASE)
    scene = re.sub(r"\b(the|a|an|our|this)\s+product\b", "", scene, flags=re.IGNORECASE).strip(" ,.")
    return f"{scene}, with a wide, clear stretch of bare surface in the {_region_words(region)} part of the frame"


def _job_dir(root: Path, payload: dict[str, Any]) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    title = payload.get("title") or payload.get("prompt") or payload.get("edit_instruction") or ""
    folder = Path(root).resolve() / f"{stamp}_{payload.get('purpose', 'image')}_{safe_name(title, 24)}"
    index = 1
    while folder.exists():
        index += 1
        folder = folder.with_name(f"{folder.name}_{index}")
    folder.mkdir(parents=True)
    return folder


def _copy_references(job_dir: Path, references: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """References live in the job folder with their roles, so the project keeps them."""
    target = job_dir / "refs"
    target.mkdir(exist_ok=True)
    copied = []
    for index, ref in enumerate(references, 1):
        source = Path(ref["path"])
        destination = target / f"{index:02d}_{ref['role'].lower()}{source.suffix.lower() or '.png'}"
        shutil.copy2(source, destination)
        entry = {"path": str(destination), "role": ref["role"], "source": str(source)}
        if ref.get("mask") and Path(ref["mask"]).exists():  # the corrected mask is part of the project
            mask_copy = target / f"{index:02d}_{ref['role'].lower()}_mask.png"
            shutil.copy2(ref["mask"], mask_copy)
            entry["mask"] = str(mask_copy)
        copied.append(entry)
    return copied


def _save_candidate(job_dir: Path, index: int, image: Image.Image, meta: dict[str, Any], purpose: Purpose,
                    yunet: Path, people: int, product_box: list[float] | None) -> dict[str, Any]:
    folder = job_dir / "candidates"
    folder.mkdir(exist_ok=True)
    base = f"c{index:02d}_{meta['engine']}_{meta['seed']}"
    full = folder / f"{base}.png"
    image.save(full)
    files = {"full": str(full)}
    for width in ((340, 180) if purpose.youtube else (340,)):
        path = folder / f"{base}_{width}.png"
        image.resize((width, max(1, round(width * image.height / image.width))), Image.Resampling.LANCZOS).save(path)
        files[f"preview_{width}"] = str(path)
    if people and yunet.exists():
        faces = detect_faces_yunet(image, yunet, 0.5)
        if faces:
            x, y, w, h = faces[0]["box"]
            side = max(w * image.width, h * image.height) * 1.8
            cx, cy = (x + w / 2) * image.width, (y + h / 2) * image.height
            face = image.crop((int(cx - side / 2), int(cy - side / 2), int(cx + side / 2), int(cy + side / 2)))
            face = face.resize((320, 320), Image.Resampling.LANCZOS)
            files["face"] = str(folder / f"{base}_face.png")
            face.save(files["face"])
    if product_box:
        x, y, w, h = product_box
        crop = image.crop((int(x * image.width), int(y * image.height), int((x + w) * image.width), int((y + h) * image.height)))
        crop.thumbnail((320, 320), Image.Resampling.LANCZOS)
        files["product"] = str(folder / f"{base}_product.png")
        crop.save(files["product"])
    record = {**meta, "index": index, "files": files}
    (folder / f"{base}.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return record


def _meta_from(candidate: dict[str, Any], kind: str) -> dict[str, Any]:
    manifest = candidate["manifest"]
    return {"kind": kind, "engine": manifest["backend"], "model": manifest["model"], "seed": manifest["seed"],
            "seconds": manifest["timing"]["engine_seconds"], "peak_vram_mib": manifest["peak_vram_mib"],
            "score": candidate["score"], "warnings": list(candidate["qa"].get("problems") or [])
            + list(candidate["qa"].get("warnings") or []),
            "original_prompt": manifest["original_prompt"], "translated_prompt": manifest.get("translated_prompt", ""),
            "translation_applied": manifest.get("translation_method", "none") != "none",
            "compiled_prompt": manifest["compiled_prompt"], "license": manifest["model_license"],
            "commercial_use": manifest["commercial_use_flag"], "quality_mode": manifest["quality_mode"],
            "memory_profile": manifest["memory_profile"]}


def run_creator_job(payload: dict[str, Any], cancel: Event | None, *, models_dir: Path, output_root: Path,
                    progress: Progress | None = None, app_root: Path | None = None) -> dict[str, Any]:
    cancel = cancel or Event()
    started = time.perf_counter()
    note = progress or (lambda event: None)
    purposes, prompt_presets, _ = load_presets(app_root) if app_root else ({}, BUILTIN_PROMPT_PRESETS, [])
    from .creator_presets import BUILTIN_PURPOSES
    purpose = purposes.get(payload.get("purpose", "")) or BUILTIN_PURPOSES.get(payload.get("purpose", ""), BUILTIN_PURPOSES["custom"])
    canvas = tuple(payload.get("canvas") or (purpose.width, purpose.height))
    preset = prompt_presets.get(payload.get("prompt_preset") or "")
    prompt = apply_appearance_hint(str(payload.get("prompt") or ""), preset)
    people = int(payload.get("people") or 0)
    job_dir = Path(payload["job_dir"]) if payload.get("job_dir") else _job_dir(output_root, payload)
    references = _copy_references(job_dir, payload.get("references") or [])
    (job_dir / "job.json").write_text(json.dumps({**payload, "references": references, "job_dir": str(job_dir)},
                                                 ensure_ascii=False, indent=2), encoding="utf-8")
    memory_before = resource_snapshot().to_dict()
    yunet = Path(models_dir) / "face_detection" / YUNET_FILENAME
    count = int(payload.get("candidates") or 2)
    base = {"models_dir": str(models_dir), "channel": payload.get("channel", ""), "purpose": purpose.compiler_purpose,
            "mode": payload.get("quality", "balanced"), "seed": int(payload.get("seed") or 1), "people": people,
            "composition": payload.get("composition", ""), "text_side": purpose.text_side,
            "memory_profile": payload.get("memory", "interactive_low_memory"), "canvas": list(canvas),
            "max_candidates_per_engine": count}
    records: list[dict[str, Any]] = []
    warnings: list[str] = []
    product_refs = [r for r in references if r["role"] == "PRODUCT"]
    other_refs = [r for r in references if r["role"] != "PRODUCT"]

    def stage(text: str, fraction: float) -> None:
        note({"phase": "stage", "message": text, "fraction": fraction})

    if payload.get("kind") == "edit":
        stage("편집 생성 중 (FLUX.2)", 0.1)
        result = run_quality_job({**base, "prompt": payload.get("prompt", ""), "edit_image": payload["edit_image"],
                                  "edit_instruction": payload.get("edit_instruction", ""), "references": other_refs + product_refs},
                                 cancel, note)
        warnings += result["warnings"]
        for candidate in result["candidates"]:
            records.append(_save_candidate(job_dir, len(records) + 1, candidate["image"], _meta_from(candidate, "edit"),
                                           purpose, yunet, people, None))
    elif product_refs and product_mode(payload) != "none":
        records, extra = _product_job(payload, base, prompt, purpose, canvas, product_refs, other_refs, job_dir,
                                      yunet, people, cancel, note, stage)
        warnings += extra
    else:
        stage("생성 중", 0.1)
        result = run_quality_job({**base, "prompt": prompt, "references": references}, cancel, note)
        warnings += result["warnings"]
        for candidate in result["candidates"]:
            records.append(_save_candidate(job_dir, len(records) + 1, candidate["image"], _meta_from(candidate, "generated"),
                                           purpose, yunet, people, None))
    if cancel.is_set():
        raise EngineCancelled("Cancelled.")
    time.sleep(1.0)  # let the driver report freed memory after the engine process exited
    memory_after = resource_snapshot().to_dict()
    summary = {"job_dir": str(job_dir), "purpose": purpose.key, "canvas": list(canvas), "candidates": records,
               "warnings": list(dict.fromkeys(warnings)), "seconds": round(time.perf_counter() - started, 1),
               "memory_before": memory_before, "memory_after": memory_after}
    (job_dir / "result.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    stage("완료", 1.0)
    return summary


PRODUCT_MODES = {"strict": "원본 그대로 합성", "natural": "자연광 보정 합성", "ai": "AI 재구성", "none": "보존 안 함"}
COMPOSITE_NOTES = {
    "composite": "원본 그대로 합성: 상품 사진의 픽셀을 그대로 사용(크기 조절·가장자리·접지 그림자만).",
    "natural": "자연광 보정 합성: 배경만 상품 색감에 맞춰 살짝 보정하고 빛 방향 그림자를 더함. 상품 픽셀은 그대로.",
}


def product_mode(payload: dict[str, Any]) -> str:
    """strict / natural / ai / none. Older payloads: product_preserve=True meant the AI path."""
    mode = str(payload.get("product_mode") or "")
    if mode in PRODUCT_MODES:
        return mode
    return "ai" if payload.get("product_preserve") else "none"


def _product_job(payload, base, prompt, purpose, canvas, product_refs, other_refs, job_dir, yunet, people,
                 cancel, note, stage) -> tuple[list[dict[str, Any]], list[str]]:
    """Original-pixel composites first; only the AI mode adds FLUX.2 redraws (checked and labelled)."""
    from .product_checks import natural_composite
    from .product_masks import load_mask

    records: list[dict[str, Any]] = []
    warnings: list[str] = []
    preserve = product_mode(payload)
    stage("상품 분리 중", 0.05)
    with Image.open(product_refs[0]["path"]) as opened:
        mask = load_mask(product_refs[0].get("mask"), opened.size)
        cutout = extract_product(opened, mask)
    if mask is None:
        warnings.append("자동 상품 마스크를 사용했습니다. 결과에서 상품 가장자리를 확인하세요.")
    cutout_path = job_dir / "refs" / "product_cutout.png"
    cutout.save(cutout_path)
    region = purpose.subject_region
    scale = float(payload.get("product_scale") or 0.5)
    mode = base["mode"]
    composites = 1 if mode == "preview" else max(1, int(payload.get("candidates") or 2)) if preserve != "ai" else 2
    stage("배경 생성 중", 0.15)
    backgrounds = run_quality_job({**base, "purpose": "background_scene",
                                   "prompt": background_prompt(prompt, region), "original_prompt": prompt,
                                   "references": other_refs, "max_candidates_per_engine": composites}, cancel, note)
    warnings += backgrounds["warnings"]
    exact = []
    from .product_checks import auto_placement
    kind = "natural" if preserve == "natural" else "composite"
    backgrounds_dir = job_dir / "backgrounds"
    backgrounds_dir.mkdir(exist_ok=True)
    for index, candidate in enumerate(backgrounds["candidates"][:composites], 1):
        placement = auto_placement(candidate["image"], region, scale)
        compose = natural_composite if kind == "natural" else composite_product
        image, box = compose(candidate["image"], cutout, region, scale, placement=placement)
        meta = _meta_from(candidate, kind)
        background_path = backgrounds_dir / f"bg{index:02d}_{candidate['manifest']['seed']}.png"
        candidate["image"].save(background_path)
        meta["background"] = str(background_path)       # kept so position/size can be adjusted without regenerating
        meta["placement"] = placement.to_dict()
        meta["warnings"] = [w for w in meta["warnings"] if "face" not in w.casefold()]
        meta["product_check"] = composite_check(box).to_dict()
        meta["product_mode"] = preserve
        meta["note"] = COMPOSITE_NOTES[kind]
        exact.append((image, meta, box))
    for image, meta, box in exact:
        records.append(_save_candidate(job_dir, len(records) + 1, image, meta, purpose, yunet, people, box))
    if preserve != "ai" or mode == "preview" or cancel.is_set():
        return records, warnings  # strict / natural never redraw the product
    regenerated = []
    if exact:  # lighting harmonization of the best composite (FLUX.2 redraws it: checked like any regeneration)
        stage("합성 조명 보정 중 (FLUX.2)", 0.55)
        source = job_dir / "candidates" / Path(records[0]["files"]["full"]).name
        harmonized = run_quality_job({**base, "prompt": "", "edit_image": str(source),
                                      "edit_instruction": COMPOSITE_HARMONIZE, "references": product_refs,
                                      "max_candidates_per_engine": 1}, cancel, note)
        warnings += harmonized["warnings"]
        regenerated += [(c, "harmonized") for c in harmonized["candidates"][:1]]
    if mode == "best" and not cancel.is_set():
        stage("상품 참조 생성 중 (FLUX.2)", 0.75)
        # FLUX.2 only: the BEST reference plan would also load RealVis here, whose candidate is not used and
        # whose in-process CUDA context (~150 MiB) stays until the app exits.
        generated = run_quality_job({**base, "prompt": prompt, "references": product_refs + other_refs,
                                     "only_engine": "flux2_klein_4b", "max_candidates_per_engine": 2}, cancel, note)
        warnings += generated["warnings"]
        regenerated += [(c, "regenerated") for c in generated["candidates"] if c["manifest"]["backend"] == "flux2_klein_4b"]
    checks = [(candidate, kind, check_product(cutout, candidate["image"])) for candidate, kind in regenerated]
    if checks:
        stage("상품 글자/로고 확인 중", 0.95)
        if not apply_ocr(cutout, [(c["image"], check) for c, _, check in checks], job_dir / "refs" / "ocr"):
            warnings.append("상품 글자/로고 OCR 검사를 건너뛰었습니다(EasyOCR 모델 없음 또는 실패).")
    for candidate, kind, check in checks:
        meta = _meta_from(candidate, kind)
        meta["product_check"] = check.to_dict()
        meta["warnings"] = check.warnings + meta["warnings"]
        meta["score"] = round(meta["score"] - 2.0, 3)  # never ranked above an exact composite
        records.append(_save_candidate(job_dir, len(records) + 1, candidate["image"], meta, purpose, yunet, people, check.box))
    return records, warnings


def recomposite(job_dir: Path, record: dict[str, Any], purpose: Purpose, *, baseline: float, center_x: float,
                scale: float) -> dict[str, Any]:
    """'상품 위치·크기 조정': re-place the original product on the same background (no AI, exact pixels)."""
    from .product_checks import Placement, natural_composite
    job_dir = Path(job_dir)
    cutout = Image.open(job_dir / "refs" / "product_cutout.png").convert("RGBA")
    with Image.open(record["background"]) as opened:
        background = opened.convert("RGB")
    placement = Placement(baseline, center_x, scale)
    compose = natural_composite if record.get("kind") == "natural" else composite_product
    image, box = compose(background, cutout, purpose.subject_region, scale, placement=placement)
    existing = sorted((job_dir / "candidates").glob("c*.json"))
    meta = {key: record[key] for key in record if key not in ("index", "files")}
    meta.update(placement=placement.to_dict(), product_check=composite_check(box).to_dict(),
                note=(record.get("note") or "") + " (위치·크기 직접 조정)")
    people = 0
    yunet = Path("__none__")
    return _save_candidate(job_dir, len(existing) + 1, image, meta, purpose, yunet, people, box)


# ------------------------------------------------------------------ export
def _font(size: int, text: str) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    fonts = Path("C:/Windows/Fonts")
    kana = re.search(r"[぀-ヿ]", text or "")
    for name in (["YuGothB.ttc", "meiryob.ttc"] if kana else []) + ["malgunbd.ttf", "malgun.ttf", "YuGothB.ttc", "arialbd.ttf"]:
        if (fonts / name).exists():
            return ImageFont.truetype(str(fonts / name), size)
    return ImageFont.load_default()


def _fit_text(draw, text, box, max_size, min_size=14):
    x, y, w, h = box
    size = max_size
    while size > min_size:
        font = _font(size, text)
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        if right - left <= w and bottom - top <= h:
            return font, (right - left, bottom - top)
        size -= 2
    font = _font(min_size, text)
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    return font, (right - left, bottom - top)


def compose_text(image: Image.Image, purpose: Purpose, title: str = "", subtitle: str = "", cta: str = "") -> Image.Image:
    """Simple text layers in the preset's safe text/CTA regions (white text with shadow, dark CTA button)."""
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    width, height = canvas.size
    tx, ty, tw, th = purpose.text_region
    box = (int(tx * width), int(ty * height), int(tw * width), int(th * height))
    cursor = box[1]
    for text, share, colour in ((title, 0.62, (255, 255, 255)), (subtitle, 0.30, (235, 235, 235))):
        if not text:
            continue
        font, (text_w, text_h) = _fit_text(draw, text, (0, 0, box[2], int(box[3] * share)), int(box[3] * share))
        x = box[0] + (box[2] - text_w) // 2 if purpose.text_side == "top" else box[0]
        for dx, dy in ((2, 2), (1, 1)):
            draw.text((x + dx, cursor + dy), text, font=font, fill=(0, 0, 0))
        draw.text((x, cursor), text, font=font, fill=colour)
        cursor += text_h + int(box[3] * 0.06)
    if cta and purpose.cta_region:
        cx, cy, cw, ch = purpose.cta_region
        button = (int(cx * width), int(cy * height), int((cx + cw) * width), int((cy + ch) * height))
        draw.rounded_rectangle(button, radius=(button[3] - button[1]) // 2, fill=(20, 20, 20))
        font, (text_w, text_h) = _fit_text(draw, cta, (0, 0, int((button[2] - button[0]) * 0.8),
                                                       int((button[3] - button[1]) * 0.6)), int((button[3] - button[1]) * 0.6))
        draw.text((button[0] + (button[2] - button[0] - text_w) // 2, button[1] + (button[3] - button[1] - text_h) // 2 - 2),
                  cta, font=font, fill=(255, 255, 255))
    return canvas


def export_candidate(record: dict[str, Any], purpose: Purpose, *, export_dir: Path, name: str,
                     formats: tuple[str, ...] = ("jpg", "png"), jpg_quality: int = 92, title: str = "",
                     subtitle: str = "", cta: str = "", canvas: tuple[int, int] | None = None) -> list[str]:
    """Textless export always; composed export too when any text is given. Native preset size."""
    export_dir.mkdir(parents=True, exist_ok=True)
    with Image.open(record["files"]["full"]) as opened:
        image = opened.convert("RGB")
    size = canvas or (purpose.width, purpose.height)
    if image.size != tuple(size):
        image = image.resize(tuple(size), Image.Resampling.LANCZOS)
    stem = safe_name(name, 60)
    versions = {"textless": image}
    if title or subtitle or cta:
        versions["composed"] = compose_text(image, purpose, title, subtitle, cta)
    written = []
    for label, version in versions.items():
        for fmt in formats:
            path = export_dir / f"{stem}_{label}.{fmt}"
            if fmt == "jpg":
                version.save(path, "JPEG", quality=int(jpg_quality), optimize=True, progressive=True)
            else:
                version.save(path, "PNG", optimize=True)
            written.append(str(path))
    return written

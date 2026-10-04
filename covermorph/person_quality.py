"""Person-quality profiles: photoreal model selection, YuNet faces, face-detail pass, QA gate.

Generation still goes through ``SDXLTextToImageEngine``; this module only chooses the
checkpoint/settings, finds faces with OpenCV YuNet (Haar fallback), refines faces with an
SDXL img2img pass that shares the loaded pipeline's components, and measures technical
face quality. It never infers gender or identity.
"""
from __future__ import annotations

import threading
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageFilter

# ------------------------------------------------------------------ model profiles
MODEL_PROFILES: dict[str, dict[str, Any]] = {
    "sdxl_base": {
        "folder": "sdxl_base_1.0",
        "repository": "stabilityai/stable-diffusion-xl-base-1.0",
        "revision": None,
        "license": "openrail++",
        "model_card": "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0",
        "variant": None,
        "guidance_scale": 7.0,
        "scheduler": "default",
        "role": "compatibility fallback",
    },
    "photoreal_sdxl": {
        "folder": "realvisxl_v5.0",
        "repository": "SG161222/RealVisXL_V5.0",
        "revision": "ac93e0dda1f6d448cae19bbfab8c5e720a5e48bc",
        "license": "openrail++",
        "model_card": "https://huggingface.co/SG161222/RealVisXL_V5.0",
        "variant": "fp16",
        # Model card recommendation: DPM++ 2M Karras/SDE Karras, CFG 3-7 (lower keeps skin natural).
        "guidance_scale": 5.0,
        "scheduler": "dpmpp_2m_karras",
        "role": "default for scenes with people",
    },
}
YUNET_REPOSITORY = "opencv/face_detection_yunet"
YUNET_REVISION = "3cc26e7f1014a5ee5d74a42acee58bafc9d0a310"
YUNET_FILENAME = "face_detection_yunet_2023mar.onnx"
YUNET_LICENSE = "MIT"
IDENTITY_BACKEND_NOTE = ("InstantID is not enabled: its face encoder (insightface antelopev2) is licensed for "
                         "non-commercial research only, which does not fit a monetized YouTube workflow.")

# Quality modes: FAST = one pass; QUALITY = more steps + face detail + QA regenerate-once.
QUALITY_MODES: dict[str, dict[str, Any]] = {
    "fast": {"steps": 22, "face_detail": False, "qa_retry": False},
    # Face detail is opt-in: on RealVisXL output it gave no visible gain at identity-safe strengths (<=0.35)
    # and changed the person at 0.55 (measured 2026-10-03), so QUALITY does not run it unless asked.
    "quality": {"steps": 30, "face_detail": False, "qa_retry": True},
}


def inspect_photoreal_model(path: Path) -> dict[str, Any]:
    from .generation import inspect_sdxl_model

    result = inspect_sdxl_model(path)
    if result["ready"] and not (path / "unet" / "diffusion_pytorch_model.fp16.safetensors").is_file():
        result.update(ready=False, failure_reason="fp16 UNet weights are missing; run prepare_person_quality_models.py")
    return result


def model_profile_status(models_dir: Path) -> dict[str, Any]:
    status = {}
    for name, profile in MODEL_PROFILES.items():
        path = models_dir / profile["folder"]
        inspection = inspect_photoreal_model(path) if profile["variant"] else _inspect_base(path)
        status[name] = {"path": str(path), "ready": inspection["ready"], "failure_reason": inspection["failure_reason"],
                        **{key: profile[key] for key in ("repository", "revision", "license", "model_card", "role")}}
    return status


def _inspect_base(path: Path) -> dict[str, Any]:
    from .generation import inspect_sdxl_model

    return inspect_sdxl_model(path)


def inspect_yunet(models_dir: Path) -> dict[str, Any]:
    path = models_dir / "face_detection" / YUNET_FILENAME
    ready = path.is_file() and path.stat().st_size > 100_000 and hasattr(cv2, "FaceDetectorYN")
    return {"path": str(path), "ready": ready, "license": YUNET_LICENSE, "repository": YUNET_REPOSITORY,
            "revision": YUNET_REVISION,
            "failure_reason": None if ready else "YuNet model file missing (run prepare_person_quality_models.py)"}


# ------------------------------------------------------------------ face detection
_YUNET_LOCK = threading.Lock()


@lru_cache(maxsize=2)
def _yunet(model_path: str) -> Any:
    """FaceDetectorYN from in-memory bytes so Korean/Japanese install paths work."""
    data = np.frombuffer(Path(model_path).read_bytes(), dtype=np.uint8)
    return cv2.FaceDetectorYN.create("onnx", data, np.array([], dtype=np.uint8), (320, 320), 0.6, 0.3, 50)


def detect_faces_yunet(image: Image.Image, model_path: Path, min_score: float = 0.6) -> list[dict[str, Any]]:
    """Normalized face boxes with confidence and 5 landmarks, largest first."""
    rgb = image.convert("RGB")
    width, height = rgb.size
    scale = min(1.0, 1280 / max(width, height))
    work = rgb.resize((round(width * scale), round(height * scale)), Image.Resampling.LANCZOS) if scale < 1 else rgb
    bgr = cv2.cvtColor(np.asarray(work), cv2.COLOR_RGB2BGR)
    with _YUNET_LOCK:
        detector = _yunet(str(model_path))
        detector.setInputSize((bgr.shape[1], bgr.shape[0]))
        detector.setScoreThreshold(min_score)
        _, faces = detector.detect(bgr)
    results = []
    for row in faces if faces is not None else []:
        x, y, w, h = (float(v) for v in row[:4])
        landmarks = [(float(row[4 + 2 * i]) / bgr.shape[1], float(row[5 + 2 * i]) / bgr.shape[0]) for i in range(5)]
        box = (max(0.0, x / bgr.shape[1]), max(0.0, y / bgr.shape[0]), w / bgr.shape[1], h / bgr.shape[0])
        results.append({"box": box, "confidence": float(row[14]), "landmarks": landmarks})
    return sorted(results, key=lambda face: face["box"][2] * face["box"][3], reverse=True)


# ------------------------------------------------------------------ technical face metrics
def face_sharpness(image: Image.Image, box: tuple[float, float, float, float]) -> float:
    """Variance of the Laplacian on the face crop downscaled (never upscaled) to 128 px wide.

    Upscaling small crops deflated the old metric; measured on real outputs, soft SDXL-base faces score
    ~70-95 and sharp photoreal faces ~120-880 once crops are only ever reduced.
    """
    width, height = image.size
    x, y, w, h = box
    crop = image.crop((int(x * width), int(y * height), int((x + w) * width), int((y + h) * height)))
    if crop.width < 8 or crop.height < 8:
        return 0.0
    target = min(128, crop.width)
    gray = np.asarray(crop.convert("L").resize((target, max(1, round(target * crop.height / crop.width))),
                                                Image.Resampling.LANCZOS), dtype=np.float32)
    return float(cv2.Laplacian(gray, cv2.CV_32F).var())


def exposure_stats(image: Image.Image) -> dict[str, float]:
    luma = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
    return {"mean": float(luma.mean()), "clipped_dark": float((luma < 0.02).mean()), "clipped_bright": float((luma > 0.98).mean())}


# ------------------------------------------------------------------ composition / prompts
COMPOSITION_PROFILES: dict[str, dict[str, Any]] = {
    # Face height = YuNet forehead-to-chin box / frame height. Measured on RealVisXL: detail words alone push
    # it to 0.45-0.7 (no room for typography), so framing words ask for visible surroundings explicitly.
    "SOLO_CLOSE": {"people": 1, "min_face_height": 0.16, "max_face_height": 0.40,
                   "framing": "waist-up shot, head and shoulders with the background visible, subject off-center"},
    "SOLO_MEDIUM": {"people": 1, "min_face_height": 0.10, "max_face_height": 0.30,
                    "framing": "medium-wide shot, upper body and surroundings visible, subject off-center"},
    "COUPLE_MEDIUM": {"people": 2, "min_face_height": 0.09, "max_face_height": 0.28,
                      "framing": "medium-wide shot of exactly two people, upper bodies and surroundings visible"},
    "COUPLE_EMOTIONAL": {"people": 2, "min_face_height": 0.12, "max_face_height": 0.35,
                         "framing": "medium shot of exactly two people facing each other, upper bodies visible"},
    # RC2 (preset-only): Tokyo Chill couple. Same seeds: the old wording gave two women (9301) and stiff forward stares;
    # this two-shot gave a man and a woman interacting with readable faces and the city still visible.
    "COUPLE_CLOSE": {"people": 2, "min_face_height": 0.14, "max_face_height": 0.38,
                     "framing": "medium-close two-shot of exactly two people, both faces clearly visible and turned "
                                "slightly toward each other, shoulders and a little of the surroundings visible, the pair "
                                "placed off-center"},
    "SCENERY_WITH_PERSON": {"people": 1, "min_face_height": 0.0, "max_face_height": 0.15,
                            "framing": "wide cinematic shot, a small person in the scene"},
}
PERSON_POSITIVE = ("cinematic film photograph, natural skin texture, realistic hair, natural facial proportions, "
                   "subtle expression, realistic lighting")
# Ordered by priority: fit_prompt keeps as many as the 77-token CLIP limit allows.
PERSON_NEGATIVE_PARTS = [
    "text, watermark, logo, letters, signage",
    "deformed face, asymmetrical eyes, malformed hands, extra fingers, fused fingers",
    "plastic skin, waxy skin, over-smoothed face, doll-like face, blurry face, low-detail face",
    "duplicate person, extra person, third person, twins, clone",
    "extreme close-up, cartoon, 3d render, cgi, lowres, oversaturated",
]
FACE_DETAIL_PROMPT = ("close-up photo of the same person's face, detailed natural skin texture with pores, "
                      "detailed eyes and eyelashes, realistic hair strands, natural lips, soft natural light")
# Measured on RealVisXL at 1344x768: faces taller than ~30% of the frame are already fully detailed and
# every refinement strength only smooths them, so the pass targets 5-30% faces.
FACE_DETAIL_DEFAULTS = {"strength": 0.35, "steps": 24, "guidance_scale": 4.0, "expand": 2.0, "max_faces": 2,
                        "min_face_height": 0.05, "max_face_height": 0.30, "work_long_side": 768}
MAX_FACE_DETAIL_STRENGTH = 0.40
IDENTITY_SIMILARITY_FLOOR = 0.985  # low-pass structure; same face >=0.99, different person 0.948 measured
_PEOPLE_TWO = ("two people", "two ", "couple", "pair", " and a ", " and an ", "lovers", "both")
_PEOPLE_ONE = ("man", "woman", "person", "girl", "boy", "lady", "gentleman", "portrait", "he ", "she ")


def people_count(prompt: str, translated: str = "") -> int:
    """People implied by the request wording (0 = scenery). Used for framing and QA, never for identity."""
    text = f" {prompt} {translated} ".casefold()
    if any(word in text for word in _PEOPLE_TWO):
        return 2
    if any(word in text for word in _PEOPLE_ONE) or "people" in text:
        return 2 if "people" in text and not any(word in text for word in _PEOPLE_ONE) else 1
    return 0


def choose_composition(people: int, candidate: str = "A", requested: str = "") -> str:
    if requested in COMPOSITION_PROFILES:
        return requested
    if people <= 0:
        return ""
    candidate = (candidate or "A").upper()[:1]
    if people == 1:
        return {"A": "SOLO_CLOSE", "B": "SOLO_MEDIUM"}.get(candidate, "SCENERY_WITH_PERSON")
    return "COUPLE_EMOTIONAL" if candidate == "A" else "COUPLE_MEDIUM"


# ------------------------------------------------------------------ QA gate
SHARPNESS_FLOOR = 110.0
MIN_SHARPNESS_FACE_PX = 64  # below this the Laplacian is noise; the face-size check covers tiny faces


def face_metrics(image: Image.Image, faces: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{"box": [round(v, 4) for v in face["box"]], "confidence": round(face["confidence"], 3),
             "height": round(face["box"][3], 4), "width_px": round(face["box"][2] * image.width),
             "sharpness": round(face_sharpness(image, face["box"]), 1)}
            for face in faces]


def quality_check(image: Image.Image, faces: list[dict[str, Any]], expected_people: int,
                  min_face_height: float, max_face_height: float = 1.0) -> dict[str, Any]:
    """Technical checks only (count, size, sharpness, exposure); never attractiveness or identity."""
    usable = [face for face in faces if face["confidence"] >= 0.7]
    metrics = face_metrics(image, usable)
    problems: list[str] = []
    warnings: list[str] = []
    if expected_people and len(usable) < expected_people:
        problems.append(f"expected {expected_people} face(s), found {len(usable)}")
    if expected_people and len(usable) > expected_people + 1:
        warnings.append(f"{len(usable)} faces found for a {expected_people}-person scene (possible duplicate/extra person)")
    main = metrics[:max(1, expected_people)] if expected_people else []
    if min_face_height and any(face["height"] < min_face_height for face in main):
        problems.append(f"face smaller than {min_face_height:.0%} of frame height")
    if main and any(face["height"] > max_face_height for face in main):
        warnings.append(f"face taller than {max_face_height:.0%} of frame height; little room for typography")
    measurable = [face for face in main if face["width_px"] >= MIN_SHARPNESS_FACE_PX]
    if measurable and min(face["sharpness"] for face in measurable) < SHARPNESS_FLOOR:
        problems.append("main face is soft/low-detail")
    exposure = exposure_stats(image)
    if exposure["mean"] < 0.06 or exposure["mean"] > 0.94 or exposure["clipped_dark"] > 0.35 or exposure["clipped_bright"] > 0.2:
        problems.append("gross exposure problem")
    score = sum(min(1.0, face["height"] / max(min_face_height, 0.05)) + min(1.0, face["sharpness"] / (SHARPNESS_FLOOR * 2))
                for face in main) / max(1, len(main)) if main else 0.0
    return {"passed": not problems, "problems": problems, "warnings": warnings, "faces": metrics,
            "exposure": {key: round(value, 3) for key, value in exposure.items()}, "score": round(score, 3)}


# ------------------------------------------------------------------ face detail pass
def _crop_box(face_box, size, expand: float) -> tuple[int, int, int, int]:
    width, height = size
    x, y, w, h = face_box
    side = max(w * width, h * height) * expand
    cx, cy = (x + w / 2) * width, (y + h / 2) * height - side * 0.04  # a little more hair than neck
    left, top = max(0, int(cx - side / 2)), max(0, int(cy - side / 2))
    right, bottom = min(width, int(cx + side / 2)), min(height, int(cy + side / 2))
    return left, top, right, bottom


def _work_size(size: tuple[int, int], long_side: int) -> tuple[int, int]:
    width, height = size
    scale = long_side / max(width, height)
    return max(64, int(round(width * scale / 8)) * 8), max(64, int(round(height * scale / 8)) * 8)


def _blend_mask(size: tuple[int, int]) -> Image.Image:
    from PIL import ImageDraw

    width, height = size
    mask = Image.new("L", size, 0)
    inset_x, inset_y = int(width * 0.12), int(height * 0.10)
    ImageDraw.Draw(mask).ellipse((inset_x, inset_y, width - inset_x, height - inset_y), fill=255)
    return mask.filter(ImageFilter.GaussianBlur(max(4, min(size) // 10)))


def structural_similarity(original: Image.Image, refined: Image.Image, size: int = 48) -> float:
    """Correlation of the blurred inner face region: tracks features (identity), ignores added micro-texture."""
    def prepare(image: Image.Image) -> np.ndarray:
        width, height = image.size
        inner = image.convert("L").crop((int(width * 0.2), int(height * 0.2), int(width * 0.8), int(height * 0.8)))
        array = np.asarray(inner.filter(ImageFilter.GaussianBlur(max(1, width // 150))).resize((size, size)), dtype=np.float64)
        return (array - array.mean()) / (array.std() + 1e-6)

    return float((prepare(original) * prepare(refined)).mean())


def _same_face(original: Image.Image, refined: Image.Image, model_path: Path | None) -> tuple[bool, str]:
    """Reject refinements that change the person, move/lose the face, or end up softer than the original."""
    if refined.size != original.size:
        return False, "size changed"
    similarity = structural_similarity(original, refined)
    if similarity < IDENTITY_SIMILARITY_FLOOR:
        return False, f"face structure changed (similarity {similarity:.3f}); possible identity change"
    if face_sharpness(refined, (0, 0, 1, 1)) < face_sharpness(original, (0, 0, 1, 1)) * 0.9:
        return False, "refined crop is softer"
    if model_path is None:
        return True, "no detector to verify geometry"
    before = detect_faces_yunet(original, model_path, 0.5)
    after = detect_faces_yunet(refined, model_path, 0.5)
    if not before:
        return True, "original face not re-detected in crop; kept geometry check skipped"
    if not after:
        return False, "face lost after refinement"
    b, a = before[0]["box"], after[0]["box"]
    shift = max(abs((b[0] + b[2] / 2) - (a[0] + a[2] / 2)), abs((b[1] + b[3] / 2) - (a[1] + a[3] / 2)))
    scale = (a[2] * a[3]) / max(b[2] * b[3], 1e-6)
    landmark_shift = max(abs(pb[0] - pa[0]) + abs(pb[1] - pa[1])
                         for pb, pa in zip(before[0]["landmarks"], after[0]["landmarks"]))
    if shift > 0.06 or not 0.75 <= scale <= 1.33 or landmark_shift > 0.08:
        return False, f"face geometry drifted (shift={shift:.3f}, scale={scale:.2f}, landmarks={landmark_shift:.3f})"
    return True, "ok"


def refine_faces(engine: Any, image: Image.Image, faces: list[dict[str, Any]], *, prompt: str, negative_prompt: str,
                 seed: int, model_path: Path | None, settings: dict[str, Any] | None = None,
                 reference_image: Image.Image | None = None) -> tuple[Image.Image, dict[str, Any]]:
    """Low-denoise SDXL img2img on enlarged face crops, blended back with a feathered ellipse.

    The img2img pipeline reuses the already loaded components (no second model in VRAM). Any
    crop whose refinement fails the geometry/sharpness check is left untouched.
    """
    import time

    import torch
    from diffusers import StableDiffusionXLImg2ImgPipeline

    options = {**FACE_DETAIL_DEFAULTS, **(settings or {})}
    options["strength"] = min(float(options["strength"]), MAX_FACE_DETAIL_STRENGTH)
    started = time.perf_counter()
    info: dict[str, Any] = {"settings": options, "faces": [], "applied": False}
    if torch.cuda.is_available():
        torch.cuda.empty_cache()  # release the text-to-image activations before the img2img pass
        torch.cuda.reset_peak_memory_stats()
    img2img = StableDiffusionXLImg2ImgPipeline(**engine.pipeline.components)
    if (getattr(engine, "memory_profile", None) or {}).get("cpu_offload"):
        img2img.enable_model_cpu_offload()
    result = image.copy()
    for index, face in enumerate(faces[: options["max_faces"]]):
        record: dict[str, Any] = {"box": [round(v, 4) for v in face["box"]], "applied": False}
        info["faces"].append(record)
        if face["box"][3] < options["min_face_height"] or face["confidence"] < 0.7:
            record["reason"] = "face too small or low confidence; left unchanged"
            continue
        if face["box"][3] > options["max_face_height"]:
            record["reason"] = "face already high-resolution; left unchanged"
            continue
        box = _crop_box(face["box"], result.size, options["expand"])
        crop = result.crop(box)
        work = crop.resize(_work_size(crop.size, options["work_long_side"]), Image.Resampling.LANCZOS)
        kwargs: dict[str, Any] = {
            "prompt": prompt, "negative_prompt": negative_prompt, "image": work, "strength": options["strength"],
            "num_inference_steps": options["steps"], "guidance_scale": options["guidance_scale"],
            "generator": torch.Generator(device="cuda").manual_seed(seed + 7919 * (index + 1)),
        }
        if getattr(engine, "ip_adapter_loaded", False) and reference_image is not None:
            kwargs["ip_adapter_image"] = reference_image
        try:
            refined_work = img2img(**kwargs).images[0]
        except Exception as exc:  # OOM or pipeline error: keep the original face
            record["reason"] = f"refinement failed: {type(exc).__name__}"
            continue
        refined = refined_work.resize(crop.size, Image.Resampling.LANCZOS)
        if options.get("keep_candidates"):  # validation/debug only: inspect rejected refinements too
            record["_candidate"], record["_original"] = refined, crop
        ok, reason = _same_face(crop, refined, model_path)
        record["reason"] = reason
        record["sharpness_before"] = round(face_sharpness(crop, (0, 0, 1, 1)), 1)
        record["sharpness_after"] = round(face_sharpness(refined, (0, 0, 1, 1)), 1)
        if not ok:
            continue
        result.paste(Image.composite(refined, crop, _blend_mask(crop.size)), box[:2])
        record["applied"] = True
        info["applied"] = True
    del img2img
    info["seconds"] = round(time.perf_counter() - started, 3)
    if torch.cuda.is_available():
        info["peak_memory_allocated"] = int(torch.cuda.max_memory_allocated())
        info["peak_memory_reserved"] = int(torch.cuda.max_memory_reserved())
    return result, info


# ------------------------------------------------------------------ final processing
def finish_image(image: Image.Image) -> Image.Image:
    """Restrained finish: faint local contrast + subtle sharpening; no saturation/WB/smoothing changes."""
    local = image.filter(ImageFilter.UnsharpMask(radius=40, percent=8, threshold=0))
    return local.filter(ImageFilter.UnsharpMask(radius=1.0, percent=30, threshold=3))


# ------------------------------------------------------------------ one complete person pass
def run_person_pass(engine: Any, first: Image.Image, *, config: Any, mode: str, prompt: str, negative_prompt: str,
                    seed: int, expected_people: int, composition: str, model_path: Path | None,
                    face_detail: bool | None = None, reference_image: Image.Image | None = None,
                    cancel_event: Any = None, keep_intermediate: bool = False) -> tuple[Image.Image, dict[str, Any]]:
    """QA (+ one deterministic regenerate in QUALITY), then the optional face-detail pass."""
    from threading import Event

    settings = QUALITY_MODES[mode]
    min_height = COMPOSITION_PROFILES.get(composition, {}).get("min_face_height", 0.0)
    max_height = COMPOSITION_PROFILES.get(composition, {}).get("max_face_height", 1.0)
    detector = "yunet_2023mar" if model_path is not None else "none"
    faces = detect_faces_yunet(first, model_path) if model_path else []
    qa = quality_check(first, faces, expected_people, min_height, max_height)
    info: dict[str, Any] = {"mode": mode, "composition": composition, "face_detector": detector, "qa_attempts": [qa],
                            "seed_used": seed, "regenerated": False}
    image = first
    if settings["qa_retry"] and expected_people and not qa["passed"]:
        retry_seed = (seed * 1103515245 + 12345) % (2**31 - 1)
        try:
            second = engine.generate_one(prompt, negative_prompt, config, retry_seed, cancel_event or Event(),
                                         reference_image=reference_image)
            second_faces = detect_faces_yunet(second, model_path) if model_path else []
            second_qa = quality_check(second, second_faces, expected_people, min_height, max_height)
            info["qa_attempts"].append(second_qa)
            info["regenerated"] = True
            info["retry_generation_seconds"] = engine.last_generation_metrics.get("generation_time_seconds")
            if (second_qa["passed"], second_qa["score"]) > (qa["passed"], qa["score"]):
                image, faces, qa, info["seed_used"] = second, second_faces, second_qa, retry_seed
        except Exception as exc:  # keep the first image; never loop
            info["retry_error"] = f"{type(exc).__name__}: {exc}"[:200]
    info["qa"] = qa
    if keep_intermediate:
        info["_before_face_detail"] = image
    use_detail = settings["face_detail"] if face_detail is None else face_detail
    info["face_detail_applied"] = False
    if use_detail and faces and expected_people:
        detail_prompt = f"{FACE_DETAIL_PROMPT}, {prompt}"
        image, detail = refine_faces(engine, image, faces, prompt=detail_prompt, negative_prompt=negative_prompt,
                                     seed=info["seed_used"], model_path=model_path, reference_image=reference_image)
        info["face_detail"] = detail
        info["face_detail_applied"] = detail["applied"]
    return image, info

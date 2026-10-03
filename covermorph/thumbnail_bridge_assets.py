"""Canvas analysis, sidecar builders and atomic output commit for the thumbnail bridge.

Everything here is CPU-only (PIL/numpy/OpenCV) so it is shared by real AI runs,
local recompose edits and unit tests. Coordinates in sidecars are normalized 0..1.
Roles come from request/project intent and image position, never from gender.
"""
from __future__ import annotations

import colorsys
import json
import os
import shutil
import threading
import uuid
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
from PIL import Image, ImageOps

from .project import utc_now
from .thumbnail_bridge import PROTOCOL_VERSION, standard_output_paths

FINAL_SIZE = (1280, 720)
REQUIRED_OUTPUTS = ("canvas_clean", "subject_boxes", "safe_zones", "palette", "composition", "project_manifest")
STAGING_PREFIX = ".thumbnail_bridge_staging_"
# Text-region candidates (x, y, w, h) evaluated on every canvas; ranked by busyness/collisions/preference.
TEXT_REGION_CANDIDATES = {
    "left_full": (0.04, 0.08, 0.38, 0.84),
    "left_upper": (0.04, 0.07, 0.40, 0.36),
    "left_lower": (0.04, 0.56, 0.40, 0.34),
    "right_full": (0.58, 0.08, 0.38, 0.84),
    "right_upper": (0.56, 0.07, 0.40, 0.36),
    "right_lower": (0.56, 0.56, 0.40, 0.34),
    "top_band": (0.06, 0.05, 0.88, 0.26),
    "bottom_band": (0.06, 0.70, 0.88, 0.25),
    "center_lower": (0.25, 0.62, 0.50, 0.30),
}

Box = tuple[float, float, float, float]


class BridgeOutputError(RuntimeError):
    pass


@dataclass(slots=True)
class Subject:
    role: str
    kind: str
    box: Box
    confidence: float
    source: str
    face: Box | None = None

    def to_dict(self, index: int) -> dict[str, Any]:
        x, y, w, h = (round(value, 4) for value in self.box)
        ids = {"protagonist": "main", "counterpart": "secondary"}
        data: dict[str, Any] = {
            "id": ids.get(self.role, f"other_{index}"),
            "role": self.role, "kind": self.kind, "x": x, "y": y, "w": w, "h": h,
            "confidence": round(self.confidence, 3), "source": self.source,
        }
        if self.face is not None:
            data["face"] = {key: round(value, 4) for key, value in zip(("x", "y", "w", "h"), self.face)}
        return data


@dataclass(slots=True)
class CanvasAnalysis:
    subjects: list[Subject]
    preferred_regions: list[dict[str, Any]]
    avoid_regions: list[dict[str, Any]]
    palette: dict[str, Any]
    text_side: str
    notes: list[str] = field(default_factory=list)


def clamp_box(box: Box) -> Box:
    x, y, w, h = (float(value) for value in box)
    if x < 0:
        w, x = w + x, 0.0
    if y < 0:
        h, y = h + y, 0.0
    x, y = min(x, 1.0), min(y, 1.0)
    return x, y, max(0.0, min(w, 1.0 - x)), max(0.0, min(h, 1.0 - y))


def _intersection(a: Box, b: Box) -> float:
    width = min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0])
    height = min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1])
    return max(0.0, width) * max(0.0, height)


def overlap_fraction(region: Box, other: Box) -> float:
    """Fraction of ``region`` covered by ``other``."""
    area = region[2] * region[3]
    return _intersection(region, other) / area if area > 0 else 0.0


def fit_16x9(image: Image.Image, size: tuple[int, int] = FINAL_SIZE) -> Image.Image:
    """Crop to the target aspect (never stretch) and Lanczos-resample to the final size."""
    return ImageOps.fit(image.convert("RGB"), size, method=Image.Resampling.LANCZOS, centering=(0.5, 0.5))


# ------------------------------------------------------------------ subjects
_CASCADE_LOCK = threading.Lock()


@lru_cache(maxsize=2)
def _face_cascade(name: str = "haarcascade_frontalface_default.xml") -> Any:
    """Load a Haar cascade from memory: cv2 cannot open files under Korean/Japanese paths."""
    cascade_path = Path(cv2.data.haarcascades) / name
    cascade = cv2.CascadeClassifier()
    try:
        storage = cv2.FileStorage(cascade_path.read_text(encoding="utf-8"), cv2.FILE_STORAGE_READ | cv2.FILE_STORAGE_MEMORY)
        if cascade.read(storage.getFirstTopLevelNode()) and not cascade.empty():
            return cascade
    except (OSError, cv2.error):
        pass
    return cv2.CascadeClassifier(str(cascade_path))


def _iou(a: Box, b: Box) -> float:
    inter = _intersection(a, b)
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def detect_faces(image: Image.Image) -> list[Box]:
    """Normalized frontal + profile (both directions) face boxes, deduplicated, largest first."""
    width, height = image.size
    scale = min(1.0, 960 / max(width, height))
    small = image.convert("L").resize((max(1, round(width * scale)), max(1, round(height * scale))), Image.Resampling.LANCZOS)
    gray = cv2.equalizeHist(np.asarray(small))
    flipped = np.ascontiguousarray(gray[:, ::-1])
    min_side = max(24, round(min(small.size) * 0.06))
    found: list[Box] = []
    passes = (("haarcascade_frontalface_default.xml", gray, False), ("haarcascade_profileface.xml", gray, False),
              ("haarcascade_profileface.xml", flipped, True))
    for name, pixels, mirrored in passes:
        cascade = _face_cascade(name)
        if cascade.empty():
            continue
        with _CASCADE_LOCK:
            hits = cascade.detectMultiScale(pixels, scaleFactor=1.08, minNeighbors=6, minSize=(min_side, min_side))
        for x, y, w, h in hits:
            x = small.width - x - w if mirrored else x
            found.append((float(x) / small.width, float(y) / small.height, float(w) / small.width, float(h) / small.height))
    # A face centered in the bottom 15% would put its body almost entirely off-frame: almost always a false hit.
    found = [box for box in found if box[1] + box[3] / 2 <= 0.85]
    unique: list[Box] = []
    for box in sorted(found, key=lambda item: item[2] * item[3], reverse=True):
        if all(_iou(box, kept) < 0.3 and overlap_fraction(box, kept) < 0.6 for kept in unique):
            unique.append(box)
    return unique


def body_box_from_face(face: Box) -> Box:
    """Conservative upper-body region implied by a face box."""
    x, y, w, h = face
    cx = x + w / 2
    return clamp_box((cx - w * 1.7, y - h * 0.6, w * 3.4, h * 4.9))


def side_of(box: Box) -> str:
    center = box[0] + box[2] / 2
    return "left" if center < 0.45 else ("right" if center > 0.55 else "center")


def _assign_roles(boxes: list[tuple[Box, Box | None, float]], protagonist_side: str, source: str) -> list[Subject]:
    """Protagonist = largest (or the one on the requested side); counterpart = next; rest = other."""
    ordered = list(boxes)
    if protagonist_side in ("left", "right") and len(ordered) > 1:
        ordered.sort(key=lambda item: (side_of(item[0]) != protagonist_side, -(item[0][2] * item[0][3])))
    subjects = []
    for index, (box, face, confidence) in enumerate(ordered):
        role = "protagonist" if index == 0 else ("counterpart" if index == 1 else "other")
        subjects.append(Subject(role, "person", clamp_box(box), confidence, source, face))
    return subjects


def subjects_from_known(records: Any) -> list[Subject]:
    """Re-read subjects from an existing subject_boxes.json (highest-priority source)."""
    subjects = []
    for record in records if isinstance(records, list) else []:
        try:
            box = clamp_box((record["x"], record["y"], record["w"], record["h"]))
        except (KeyError, TypeError, ValueError):
            continue
        if box[2] <= 0 or box[3] <= 0:
            continue
        face = record.get("face")
        try:
            face_box = clamp_box((face["x"], face["y"], face["w"], face["h"])) if isinstance(face, dict) else None
        except (KeyError, TypeError, ValueError):
            face_box = None
        role = str(record.get("role") or "other")
        subjects.append(Subject(role if role in ("protagonist", "counterpart", "other") else "other",
                                str(record.get("kind") or "person"), box, float(record.get("confidence") or 0.5),
                                str(record.get("source") or "project_known"), face_box))
    return subjects


def detect_subjects(image: Image.Image, *, text_side: str = "left", protagonist_side: str = "",
                    expect_people: bool = True, max_people: int = 3) -> tuple[list[Subject], list[str]]:
    """Faces -> upper-body boxes; conservative composition fallback when nothing is detected."""
    notes: list[str] = []
    faces = detect_faces(image)[:max_people]
    if faces:
        boxes = [(body_box_from_face(face), face, 0.72) for face in faces]
        return _assign_roles(boxes, protagonist_side, "face_detection"), notes
    if not expect_people:
        return [], notes
    # Fallback: the generator was asked to keep people opposite the text side.
    subject_side = {"left": "right", "right": "left"}.get(text_side, "center")
    x = {"right": 0.56, "left": 0.08, "center": 0.32}[subject_side]
    notes.append("No face detected; subject box is a conservative composition fallback")
    return [Subject("protagonist", "person", (x, 0.12, 0.36, 0.80), 0.3, "composition_fallback")], notes


def located_people(subjects: list[Subject]) -> list[Subject]:
    """Subjects whose position came from real evidence, not the composition fallback guess."""
    return [subject for subject in subjects if subject.face is not None or subject.source != "composition_fallback"]


def merge_subjects(known: list[Subject], fresh: list[Subject]) -> list[Subject]:
    """Keep known roles; grow each known box to cover an overlapping fresh detection; add unmatched people."""
    merged = [Subject(s.role, s.kind, s.box, s.confidence, s.source, s.face) for s in known]
    for candidate in fresh:
        match = next((s for s in merged if _iou(s.box, candidate.box) > 0.2 or overlap_fraction(candidate.box, s.box) > 0.5), None)
        if match is None:
            role = "other" if any(s.role == "counterpart" for s in merged) else ("counterpart" if merged else "protagonist")
            merged.append(Subject(role, candidate.kind, candidate.box, candidate.confidence, candidate.source, candidate.face))
            continue
        x0, y0 = min(match.box[0], candidate.box[0]), min(match.box[1], candidate.box[1])
        x1 = max(match.box[0] + match.box[2], candidate.box[0] + candidate.box[2])
        y1 = max(match.box[1] + match.box[3], candidate.box[1] + candidate.box[3])
        match.box = clamp_box((x0, y0, x1 - x0, y1 - y0))
        match.face = match.face or candidate.face
    return merged


def shift_subjects(subjects: list[Subject], dx: float) -> list[Subject]:
    shifted = []
    for subject in subjects:
        box = clamp_box((subject.box[0] + dx, subject.box[1], subject.box[2], subject.box[3]))
        face = clamp_box((subject.face[0] + dx, *subject.face[1:])) if subject.face is not None else None
        if box[2] > 0.01:
            shifted.append(Subject(subject.role, subject.kind, box, subject.confidence, subject.source, face))
    return shifted


# ------------------------------------------------------------------ safe zones
def _busyness_map(image: Image.Image) -> tuple[np.ndarray, np.ndarray]:
    small = np.asarray(image.convert("L").resize((320, 180), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
    gx = cv2.Sobel(small, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(small, cv2.CV_32F, 0, 1, ksize=3)
    return np.sqrt(gx * gx + gy * gy), small


def region_stats(image: Image.Image, box: Box) -> dict[str, float]:
    edges, luma = _busyness_map(image)
    return dict(zip(("edge_density", "luma_std", "luma_mean"), _region_stats(edges, luma, box)))


def _region_stats(edges: np.ndarray, luma: np.ndarray, box: Box) -> tuple[float, float, float]:
    h, w = edges.shape
    x0, y0 = int(box[0] * w), int(box[1] * h)
    x1, y1 = max(x0 + 1, int((box[0] + box[2]) * w)), max(y0 + 1, int((box[1] + box[3]) * h))
    edge_density = float(np.mean(edges[y0:y1, x0:x1] > 0.18))
    region = luma[y0:y1, x0:x1]
    return edge_density, float(np.std(region)), float(np.mean(region))


def _box_dict(box: Box) -> dict[str, float]:
    return {key: round(value, 4) for key, value in zip(("x", "y", "w", "h"), clamp_box(box))}


def compute_safe_zones(image: Image.Image, subjects: list[Subject], preferred_side: str = "left",
                       limit: int = 4) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    edges, luma = _busyness_map(image)
    avoid: list[dict[str, Any]] = []
    faces: list[Box] = []
    for subject in subjects:
        if subject.face is not None:
            x, y, w, h = subject.face
            face = clamp_box((x - w * 0.15, y - h * 0.15, w * 1.3, h * 1.3))
            avoid.append({"name": f"{subject.role}_face", **_box_dict(face)})
            faces.append(face)
        avoid.append({"name": f"{subject.role}_body", **_box_dict(subject.box)})
    ranked = []
    for name, box in TEXT_REGION_CANDIDATES.items():
        edge_density, luma_std, luma_mean = _region_stats(edges, luma, box)
        # Share of each face covered by the region: any real face overlap disqualifies it.
        face_hit = max((overlap_fraction(face, box) for face in faces), default=0.0)
        body_hit = max((overlap_fraction(box, subject.box) for subject in subjects), default=0.0)
        calm = 1.0 - min(1.0, edge_density * 2.2 + luma_std * 1.4)
        if preferred_side in ("", "auto", "none"):
            side_bonus = 0.0
        else:
            side_bonus = 0.12 if name.startswith(preferred_side) else -0.04
        score = 0.55 * calm + 0.35 * (1.0 - body_hit) + side_bonus - 1.2 * face_hit
        ranked.append((score, name, box, face_hit, luma_mean))
    ranked.sort(key=lambda item: item[0], reverse=True)
    preferred = [{"name": name, **_box_dict(box), "score": round(max(0.0, min(1.0, score)), 3),
                  "mean_luminance": round(luma_mean, 3)}
                 for score, name, box, face_hit, luma_mean in ranked if face_hit < 0.05][:limit]
    return preferred, avoid


# ------------------------------------------------------------------ palette
def _hex(rgb: tuple[float, float, float]) -> str:
    return "#{:02X}{:02X}{:02X}".format(*(int(round(min(255, max(0, c)))) for c in rgb))


def _tint(rgb: tuple[int, int, int], lightness: float, saturation: float) -> tuple[float, float, float]:
    hue, _, sat = colorsys.rgb_to_hls(*(c / 255.0 for c in rgb))
    r, g, b = colorsys.hls_to_rgb(hue, lightness, min(sat, saturation))
    return r * 255, g * 255, b * 255


def extract_palette(image: Image.Image, text_zone_luminance: float | None = None) -> dict[str, Any]:
    small = image.convert("RGB").resize((160, 90), Image.Resampling.BILINEAR)
    quantized = small.quantize(colors=12, method=Image.Quantize.MEDIANCUT)
    raw = quantized.getpalette() or []
    counts = sorted(quantized.getcolors() or [], reverse=True)
    colors = [(count, tuple(raw[index * 3:index * 3 + 3])) for count, index in counts]
    dominant = [rgb for _, rgb in colors[:3]]

    def saturation(rgb):
        return colorsys.rgb_to_hls(*(c / 255.0 for c in rgb))[2]

    accents = sorted((rgb for count, rgb in colors if count >= 40 and saturation(rgb) > 0.15), key=saturation, reverse=True)[:3]
    base = dominant[0] if dominant else (128, 128, 128)
    light, dark = _tint(base, 0.97, 0.25), _tint(base, 0.11, 0.45)
    stroke, shadow = _tint(base, 0.15, 0.55), _tint(base, 0.06, 0.4)
    use_light = text_zone_luminance is None or text_zone_luminance < 0.55
    highlight = accents[0] if accents else (255, 215, 108)
    return {
        "version": 1,
        "dominant": [_hex(rgb) for rgb in dominant],
        "accent": [_hex(rgb) for rgb in accents],
        "recommended_text_light": _hex(light),
        "recommended_text_dark": _hex(dark),
        "recommended_stroke": _hex(stroke if use_light else light),
        "recommended_shadow": _hex(shadow),
        # youtubesum's Image Bridge reads these keys directly.
        "fill_color": _hex(light if use_light else dark),
        "stroke_color": _hex(stroke if use_light else light),
        "highlight_color": _hex(highlight),
    }


# ------------------------------------------------------------------ analysis + sidecars
def analyze_canvas(image: Image.Image, *, preferred_side: str = "left", known_subjects: list[Subject] | None = None,
                   protagonist_side: str = "", expect_people: bool = True) -> CanvasAnalysis:
    notes: list[str] = []
    if known_subjects:
        subjects = known_subjects
    else:
        subjects, notes = detect_subjects(image, text_side=preferred_side, protagonist_side=protagonist_side,
                                          expect_people=expect_people)
    preferred, avoid = compute_safe_zones(image, subjects, preferred_side)
    text_side = preferred_side if preferred_side in ("left", "right") else "left"
    if preferred:
        best = preferred[0]
        if best["name"].endswith("band"):
            text_side = "top" if best["name"].startswith("top") else "bottom"
        elif best["name"].startswith(("left", "right")):
            text_side = best["name"].split("_")[0]
        else:
            text_side = "center"
    palette = extract_palette(image, preferred[0]["mean_luminance"] if preferred else None)
    return CanvasAnalysis(subjects, preferred, avoid, palette, text_side, notes)


def _text_positions(region: dict[str, Any]) -> dict[str, list[float]]:
    """Suggested youtubesum role boxes inside the best text region (normalized)."""
    x, y, w, h = region["x"], region["y"], region["w"], region["h"]
    return {
        "channel_label": [round(x, 4), round(y, 4), round(w * 0.6, 4), round(h * 0.12, 4)],
        "main_title": [round(x, 4), round(y + h * 0.18, 4), round(w, 4), round(h * 0.46, 4)],
        "subtitle": [round(x, 4), round(y + h * 0.68, 4), round(w, 4), round(h * 0.14, 4)],
        "episode_badge": [round(x, 4), round(y + h * 0.86, 4), round(w * 0.35, 4), round(h * 0.12, 4)],
    }


def flag_missing_faces(analysis: CanvasAnalysis, expected_people: int) -> str:
    """Record (and return a warning) when fewer faces were found than the scene is known to contain."""
    detected = sum(1 for subject in analysis.subjects if subject.face is not None)
    if expected_people <= detected:
        return ""
    analysis.notes.append(f"face_detection_incomplete: {detected} of {expected_people} expected faces detected")
    return (f"Only {detected} of {expected_people} expected faces were detected; text regions may overlap an "
            "undetected face. Check placement in the editor.")


def build_sidecars(analysis: CanvasAnalysis, *, request: Any, final_size: tuple[int, int], source_size: tuple[int, int],
                   scene_type: str, generation: dict[str, Any] | None = None,
                   history: list[dict[str, Any]] | None = None) -> dict[str, dict[str, Any]]:
    width, height = final_size
    subjects = [subject.to_dict(index) for index, subject in enumerate(analysis.subjects)]
    by_role = {subject.role: subject for subject in analysis.subjects}
    notes = list(analysis.notes)
    if "protagonist" in by_role and "counterpart" in by_role:
        notes.append("Do not cover either face; keep the gaze line between protagonist and counterpart clear")
    elif "protagonist" in by_role:
        notes.append("Keep the protagonist face clear of typography")
    composition = {
        "version": 1,
        "canvas": {"width": width, "height": height},
        "protagonist_side": side_of(by_role["protagonist"].box) if "protagonist" in by_role else "",
        "counterpart_side": side_of(by_role["counterpart"].box) if "counterpart" in by_role else "",
        "recommended_text_side": analysis.text_side,
        "scene_type": scene_type,
        "notes": notes,
        "source_generation_size": {"width": source_size[0], "height": source_size[1]},
        "final_size": {"width": width, "height": height},
        "units": "normalized",
        # youtubesum's composer imports suggested role boxes from "positions".
        "positions": _text_positions(analysis.preferred_regions[0]) if analysis.preferred_regions else {},
    }
    manifest = {
        "version": 1,
        "source_app": "CoverMorph Studio",
        "bridge_protocol_version": PROTOCOL_VERSION,
        "channel": request.channel,
        "story_type": request.story_type,
        "episode": request.episode,
        "title": request.title,
        "subtitle": request.subtitle,
        "preferred_typography": request.preferred_typography,
        "generated_at": utc_now(),
        "request_id": request.request_id,
        "last_action": request.action,
    }
    if generation is not None:
        manifest["generation"] = generation
    if history:
        manifest["edit_history"] = history[-10:]
    return {
        "subject_boxes": {"version": 1, "canvas": {"width": width, "height": height}, "units": "normalized",
                          "subjects": subjects},
        "safe_zones": {
            "version": 1,
            "units": "normalized",
            "preferred_text_regions": analysis.preferred_regions,
            "avoid_regions": analysis.avoid_regions,
            # youtubesum treats its safe zones as hazards for text, so only avoid regions are mirrored there.
            "avoid": [{"role": item["name"], **{key: item[key] for key in ("x", "y", "w", "h")}}
                      for item in analysis.avoid_regions],
        },
        "palette": analysis.palette,
        "composition": composition,
        "project_manifest": manifest,
    }


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


# ------------------------------------------------------------------ atomic commit
def _fsync_write(path: Path, data: bytes) -> None:
    with path.open("wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def stage_outputs(project_dir: Path, canvas: Image.Image, sidecars: dict[str, dict[str, Any]],
                  preview: Image.Image | None = None) -> Path:
    staging = project_dir / f"{STAGING_PREFIX}{uuid.uuid4().hex[:12]}"
    staging.mkdir(parents=True)
    names = {key: path.name for key, path in standard_output_paths(staging).items()}
    try:
        canvas.convert("RGB").save(staging / names["canvas_clean"], format="PNG")
        if preview is not None:
            preview.convert("RGB").save(staging / names["preview_reference"], format="PNG")
        for key, payload in sidecars.items():
            _fsync_write(staging / names[key], json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8"))
        validate_staged(staging, canvas.size)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return staging


def validate_staged(staging: Path, expected_size: tuple[int, int]) -> None:
    paths = standard_output_paths(staging)
    for key in REQUIRED_OUTPUTS:
        if not paths[key].is_file() or paths[key].stat().st_size == 0:
            raise BridgeOutputError(f"staged output missing: {paths[key].name}")
    with Image.open(paths["canvas_clean"]) as opened:
        opened.verify()
    with Image.open(paths["canvas_clean"]) as opened:
        if opened.size != expected_size:
            raise BridgeOutputError(f"staged canvas is {opened.size}, expected {expected_size}")
    for key in REQUIRED_OUTPUTS[1:]:
        if not read_json(paths[key]):
            raise BridgeOutputError(f"staged sidecar is invalid JSON: {paths[key].name}")


def commit_outputs(project_dir: Path, staging: Path, replace: Callable[[str, str], None] = os.replace) -> list[str]:
    """Move staged files over the finals; on any failure restore every previous file."""
    finals = standard_output_paths(project_dir)
    staged = standard_output_paths(staging)
    backup = staging / ".previous"
    backup.mkdir()
    keys = [key for key in finals if staged[key].is_file()]
    moved_old: list[str] = []
    installed: list[str] = []
    try:
        for key in keys:
            if finals[key].exists():
                replace(str(finals[key]), str(backup / finals[key].name))
                moved_old.append(key)
            replace(str(staged[key]), str(finals[key]))
            installed.append(key)
        # A previous preview this run did not replace would describe the old canvas.
        if "preview_reference" not in keys and finals["preview_reference"].exists():
            finals["preview_reference"].unlink()
    except Exception:
        for key in installed:
            try:
                finals[key].unlink()
            except OSError:
                pass
        for key in moved_old:
            try:
                os.replace(str(backup / finals[key].name), str(finals[key]))
            except OSError:
                pass
        raise
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return [finals[key].name for key in keys]


def cleanup_stale_staging(project_dir: Path) -> None:
    for stale in project_dir.glob(f"{STAGING_PREFIX}*"):
        if stale.is_dir():
            shutil.rmtree(stale, ignore_errors=True)

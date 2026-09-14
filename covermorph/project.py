from __future__ import annotations

import json
import os
import shutil
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps, UnidentifiedImageError

PROJECT_SCHEMA_VERSION = 1
PROJECT_FILENAME = "covermorph_project.json"

INPUT_TYPE_TEXTLESS = "textless"
INPUT_TYPE_EXISTING_COVER = "existing_cover"
INPUT_TYPES = {INPUT_TYPE_TEXTLESS, INPUT_TYPE_EXISTING_COVER}


class ProjectError(RuntimeError):
    pass


class ProjectLoadError(ProjectError):
    pass


class ProjectAssetError(ProjectError):
    pass


@dataclass(slots=True)
class ProjectIssue:
    candidate_id: str
    kind: str
    path: str
    message: str


@dataclass(slots=True)
class CandidateSettings:
    out_square: bool = True
    out_thumb: bool = True
    out_shorts: bool = True
    manual_boxes: tuple[tuple[int, int, int, int], ...] = ()
    ocr_boxes: tuple[tuple[int, int, int, int], ...] = ()
    preset_name: str = "OldPopLounge"
    ocr_languages: tuple[str, ...] = ("en",)
    extension_mode: str = "ai_natural"
    subject_offset_x: float = 0.0
    subject_offset_y: float = 0.0
    subject_scale: float = 1.0
    outpaint_prompt: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> CandidateSettings:
        if not isinstance(data, dict):
            return cls()
        languages = data.get("ocr_languages", ("en",))
        if not isinstance(languages, (list, tuple)):
            languages = ("en",)
        manual_boxes = boxes_from_data(data.get("manual_boxes"))
        ocr_boxes = boxes_from_data(data.get("ocr_boxes"))
        return cls(
            out_square=bool(data.get("out_square", True)),
            out_thumb=bool(data.get("out_thumb", True)),
            out_shorts=bool(data.get("out_shorts", True)),
            manual_boxes=manual_boxes,
            ocr_boxes=ocr_boxes,
            preset_name=str(data.get("preset_name") or "OldPopLounge"),
            ocr_languages=tuple(str(item) for item in languages),
            extension_mode=str(data.get("extension_mode") or "ai_natural"),
            subject_offset_x=float(data.get("subject_offset_x") or 0.0),
            subject_offset_y=float(data.get("subject_offset_y") or 0.0),
            subject_scale=float(data.get("subject_scale") or 1.0),
            outpaint_prompt=str(data.get("outpaint_prompt") or ""),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "out_square": self.out_square,
            "out_thumb": self.out_thumb,
            "out_shorts": self.out_shorts,
            "manual_boxes": [list(box) for box in self.manual_boxes],
            "ocr_boxes": [list(box) for box in self.ocr_boxes],
            "preset_name": self.preset_name,
            "ocr_languages": list(self.ocr_languages),
            "extension_mode": self.extension_mode,
            "subject_offset_x": self.subject_offset_x,
            "subject_offset_y": self.subject_offset_y,
            "subject_scale": self.subject_scale,
            "outpaint_prompt": self.outpaint_prompt,
        }


@dataclass(slots=True)
class CandidateRecord:
    candidate_id: str
    display_name: str
    input_type: str
    original_path: str
    external_source_path: str = ""
    working_source_path: str = ""
    removal_preview_path: str = ""
    selected: bool = True
    settings: CandidateSettings = field(default_factory=CandidateSettings)
    generated_paths: dict[str, str] = field(default_factory=dict)
    text_removal_engine: str = "Skipped"
    text_removal_status: str = "not_requested"
    auto_quality_status: str = "not_checked"
    user_approval_status: str = "not_required"
    working_source_approved: bool = False
    created_at: str = ""
    updated_at: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CandidateRecord:
        candidate_id = str(data.get("candidate_id") or data.get("id") or new_id("cand"))
        input_type = str(data.get("input_type") or INPUT_TYPE_TEXTLESS)
        if input_type not in INPUT_TYPES:
            input_type = INPUT_TYPE_TEXTLESS
        return cls(
            candidate_id=candidate_id,
            display_name=str(data.get("display_name") or data.get("original_filename") or candidate_id),
            input_type=input_type,
            original_path=str(data.get("original_path") or ""),
            external_source_path=str(data.get("external_source_path") or ""),
            working_source_path=str(data.get("working_source_path") or ""),
            removal_preview_path=str(data.get("removal_preview_path") or ""),
            selected=bool(data.get("selected", True)),
            settings=CandidateSettings.from_dict(data.get("settings")),
            generated_paths={
                str(key): str(value)
                for key, value in (data.get("generated_paths") or {}).items()
                if value is not None
            },
            text_removal_engine=str(data.get("text_removal_engine") or "Skipped"),
            text_removal_status=str(data.get("text_removal_status") or "not_requested"),
            auto_quality_status=str(data.get("auto_quality_status") or "not_checked"),
            user_approval_status=str(data.get("user_approval_status") or "not_required"),
            working_source_approved=bool(data.get("working_source_approved", False)),
            created_at=str(data.get("created_at") or ""),
            updated_at=str(data.get("updated_at") or ""),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "display_name": self.display_name,
            "input_type": self.input_type,
            "original_path": self.original_path,
            "external_source_path": self.external_source_path,
            "working_source_path": self.working_source_path,
            "removal_preview_path": self.removal_preview_path,
            "selected": self.selected,
            "settings": self.settings.to_dict(),
            "generated_paths": dict(self.generated_paths),
            "text_removal_engine": self.text_removal_engine,
            "text_removal_status": self.text_removal_status,
            "auto_quality_status": self.auto_quality_status,
            "user_approval_status": self.user_approval_status,
            "working_source_approved": self.working_source_approved,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


@dataclass(slots=True)
class CoverMorphProject:
    project_id: str
    name: str
    project_file: Path
    schema_version: int = PROJECT_SCHEMA_VERSION
    channel_name: str = ""
    series_name: str = ""
    lyric_mood_text: str = ""
    song_count: int = 0
    selected_candidate_ids: list[str] = field(default_factory=list)
    candidates: list[CandidateRecord] = field(default_factory=list)
    created_at: str = ""
    updated_at: str = ""

    @property
    def project_dir(self) -> Path:
        return self.project_file.parent

    @classmethod
    def from_dict(cls, data: dict[str, Any], project_file: Path) -> CoverMorphProject:
        version = int(data.get("schema_version") or 0)
        if version > PROJECT_SCHEMA_VERSION:
            raise ProjectLoadError(
                f"Project schema {version} is newer than this app supports ({PROJECT_SCHEMA_VERSION})."
            )
        candidates_data = data.get("candidates") or []
        if not isinstance(candidates_data, list):
            candidates_data = []
        selected_ids = data.get("selected_candidate_ids") or []
        if not isinstance(selected_ids, list):
            selected_ids = []
        return cls(
            schema_version=PROJECT_SCHEMA_VERSION,
            project_id=str(data.get("project_id") or new_id("project")),
            name=str(data.get("name") or project_file.parent.name or "Untitled Project"),
            project_file=project_file,
            channel_name=str(data.get("channel_name") or ""),
            series_name=str(data.get("series_name") or ""),
            lyric_mood_text=str(data.get("lyric_mood_text") or ""),
            song_count=int(data.get("song_count") or 0),
            selected_candidate_ids=[str(item) for item in selected_ids],
            candidates=[CandidateRecord.from_dict(item) for item in candidates_data if isinstance(item, dict)],
            created_at=str(data.get("created_at") or ""),
            updated_at=str(data.get("updated_at") or ""),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": PROJECT_SCHEMA_VERSION,
            "project_id": self.project_id,
            "name": self.name,
            "channel_name": self.channel_name,
            "series_name": self.series_name,
            "lyric_mood_text": self.lyric_mood_text,
            "song_count": self.song_count,
            "selected_candidate_ids": list(self.selected_candidate_ids),
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def boxes_from_data(data: Any) -> tuple[tuple[int, int, int, int], ...]:
    if not isinstance(data, (list, tuple)):
        return ()
    boxes: list[tuple[int, int, int, int]] = []
    for item in data:
        if not isinstance(item, (list, tuple)) or len(item) != 4:
            continue
        try:
            boxes.append(tuple(int(value) for value in item))
        except (TypeError, ValueError):
            continue
    return tuple(boxes)


def project_file_from_path(path: Path) -> Path:
    if path.suffix.lower() == ".json":
        return path
    return path / PROJECT_FILENAME


def create_project(project_dir: Path, name: str | None = None) -> CoverMorphProject:
    project_file = project_file_from_path(project_dir)
    now = utc_now()
    return CoverMorphProject(
        project_id=new_id("project"),
        name=name or project_file.parent.name or "Untitled Project",
        project_file=project_file,
        created_at=now,
        updated_at=now,
    )


def ensure_project_dirs(project: CoverMorphProject) -> None:
    for relative in ("assets/originals", "assets/removal_previews", "assets/textless"):
        (project.project_dir / relative).mkdir(parents=True, exist_ok=True)


def path_to_project_string(project: CoverMorphProject, path: Path) -> str:
    try:
        return path.resolve().relative_to(project.project_dir.resolve()).as_posix()
    except (OSError, ValueError):
        return str(path)


def resolve_project_path(project: CoverMorphProject, stored_path: str) -> Path:
    path = Path(stored_path)
    if path.is_absolute():
        return path
    return project.project_dir / stored_path


def load_project(path: Path) -> CoverMorphProject:
    project_file = project_file_from_path(path)
    try:
        data = json.loads(project_file.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ProjectLoadError(f"Project file not found: {project_file}") from exc
    except json.JSONDecodeError as exc:
        raise ProjectLoadError(f"Project file is damaged JSON: {project_file}") from exc
    except OSError as exc:
        raise ProjectLoadError(f"Project file could not be read: {project_file}") from exc
    if not isinstance(data, dict):
        raise ProjectLoadError(f"Project file has an invalid root object: {project_file}")
    return CoverMorphProject.from_dict(data, project_file)


def save_project_atomic(project: CoverMorphProject) -> None:
    project.project_file.parent.mkdir(parents=True, exist_ok=True)
    ensure_project_dirs(project)
    project.schema_version = PROJECT_SCHEMA_VERSION
    project.updated_at = utc_now()
    if not project.created_at:
        project.created_at = project.updated_at
    payload = json.dumps(project.to_dict(), ensure_ascii=False, indent=2)
    temp_path = project.project_file.with_name(f".{project.project_file.name}.{uuid.uuid4().hex}.tmp")
    try:
        temp_path.write_text(payload, encoding="utf-8")
        os.replace(temp_path, project.project_file)
    finally:
        try:
            temp_path.unlink()
        except FileNotFoundError:
            pass


def _safe_suffix(path: Path) -> str:
    suffix = path.suffix.lower()
    return suffix if suffix else ".png"


def _load_rgb_image(path: Path) -> Image.Image:
    try:
        with Image.open(path) as opened:
            return ImageOps.exif_transpose(opened).convert("RGB")
    except UnidentifiedImageError as exc:
        raise ProjectAssetError(f"Unsupported or damaged image: {path}") from exc
    except OSError as exc:
        raise ProjectAssetError(f"Image could not be read: {path}") from exc


def validate_working_image(img: Image.Image, expected_size: tuple[int, int] | None = None) -> str:
    if img.width < 1 or img.height < 1:
        return "failed_empty_image"
    if expected_size is not None and img.size != expected_size:
        return "failed_size_changed"
    extrema = img.convert("L").getextrema()
    if extrema[0] == extrema[1]:
        return "basic_passed_flat_image"
    return "basic_passed"


def add_candidate_from_file(
    project: CoverMorphProject,
    source_path: Path,
    input_type: str,
    settings: CandidateSettings | None = None,
) -> CandidateRecord:
    if input_type not in INPUT_TYPES:
        raise ProjectAssetError(f"Unsupported input type: {input_type}")
    if not source_path.is_file():
        raise ProjectAssetError(f"Input image not found: {source_path}")

    ensure_project_dirs(project)
    candidate_id = new_id("cand")
    now = utc_now()
    original_dest = project.project_dir / "assets" / "originals" / f"{candidate_id}{_safe_suffix(source_path)}"
    shutil.copy2(source_path, original_dest)
    original = _load_rgb_image(original_dest)

    candidate = CandidateRecord(
        candidate_id=candidate_id,
        display_name=source_path.name,
        input_type=input_type,
        original_path=path_to_project_string(project, original_dest),
        external_source_path=str(source_path),
        selected=True,
        settings=settings or CandidateSettings(),
        created_at=now,
        updated_at=now,
    )
    if input_type == INPUT_TYPE_TEXTLESS:
        working_dest = project.project_dir / "assets" / "textless" / f"{candidate_id}_textless.png"
        original.save(working_dest, "PNG")
        candidate.working_source_path = path_to_project_string(project, working_dest)
        candidate.working_source_approved = True
        candidate.user_approval_status = "approved_as_textless_input"
        candidate.auto_quality_status = validate_working_image(original)
        candidate.text_removal_status = "not_needed"
        candidate.text_removal_engine = "Skipped"
    else:
        candidate.user_approval_status = "pending"
        candidate.text_removal_status = "pending"
        candidate.auto_quality_status = "not_checked"

    project.candidates.append(candidate)
    project.selected_candidate_ids = [item.candidate_id for item in project.candidates if item.selected]
    return candidate


def save_removal_preview(
    project: CoverMorphProject,
    candidate: CandidateRecord,
    img: Image.Image,
    engine: str,
    expected_size: tuple[int, int] | None = None,
) -> str:
    ensure_project_dirs(project)
    quality = validate_working_image(img, expected_size)
    preview_dest = project.project_dir / "assets" / "removal_previews" / f"{candidate.candidate_id}_preview.png"
    img.convert("RGB").save(preview_dest, "PNG")
    candidate.removal_preview_path = path_to_project_string(project, preview_dest)
    candidate.text_removal_engine = engine
    candidate.text_removal_status = "preview_ready"
    candidate.auto_quality_status = quality
    candidate.user_approval_status = "pending"
    candidate.working_source_approved = False
    candidate.updated_at = utc_now()
    return quality


def adopt_removal_preview(project: CoverMorphProject, candidate: CandidateRecord) -> Path:
    if not candidate.removal_preview_path:
        raise ProjectAssetError("No removal preview exists for this candidate.")
    if not candidate.auto_quality_status.startswith("basic_passed"):
        raise ProjectAssetError(f"Removal preview did not pass basic validation: {candidate.auto_quality_status}")

    preview_path = resolve_project_path(project, candidate.removal_preview_path)
    img = _load_rgb_image(preview_path)
    original_path = resolve_project_path(project, candidate.original_path)
    expected_size = _load_rgb_image(original_path).size if original_path.is_file() else None
    quality = validate_working_image(img, expected_size)
    if not quality.startswith("basic_passed"):
        candidate.auto_quality_status = quality
        candidate.user_approval_status = "blocked_by_quality_check"
        candidate.updated_at = utc_now()
        raise ProjectAssetError(f"Removal preview did not pass basic validation: {quality}")

    working_dest = project.project_dir / "assets" / "textless" / f"{candidate.candidate_id}_approved.png"
    img.save(working_dest, "PNG")
    candidate.working_source_path = path_to_project_string(project, working_dest)
    candidate.working_source_approved = True
    candidate.user_approval_status = "approved_by_user"
    candidate.text_removal_status = "adopted"
    candidate.auto_quality_status = quality
    candidate.updated_at = utc_now()
    return working_dest


def validate_project_assets(project: CoverMorphProject) -> list[ProjectIssue]:
    issues: list[ProjectIssue] = []
    for candidate in project.candidates:
        checks = [
            ("original", candidate.original_path, "Input original is missing."),
            ("working_source", candidate.working_source_path, "Textless working original is missing."),
        ]
        if candidate.input_type == INPUT_TYPE_EXISTING_COVER and candidate.removal_preview_path:
            checks.append(("removal_preview", candidate.removal_preview_path, "Removal preview is missing."))
        for kind, stored_path, message in checks:
            if not stored_path:
                if kind == "working_source" and candidate.input_type == INPUT_TYPE_EXISTING_COVER:
                    continue
                issues.append(ProjectIssue(candidate.candidate_id, kind, stored_path, message))
                continue
            resolved = resolve_project_path(project, stored_path)
            if not resolved.is_file():
                issues.append(ProjectIssue(candidate.candidate_id, kind, stored_path, message))
    return issues

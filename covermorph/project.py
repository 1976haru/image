from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps, UnidentifiedImageError

PROJECT_SCHEMA_VERSION = 3
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


REFERENCE_ROLES = {"person", "style", "background_composition"}
REFERENCE_USES = {"face", "upper_body", "full_body", "other"}
INPUT_RECORD_TYPES = {"lyrics", "image_prompt", "theme_series_mood"}


@dataclass(slots=True)
class ChannelGenerationPreset:
    preset_id: str
    name: str
    version: int = 1
    style: str = "사용자 지정"
    mood: str = ""
    color: str = ""
    lighting: str = ""
    era_region: str = ""
    person_background_balance: str = ""
    composition: str = ""
    keep_features: str = ""
    avoid_elements: str = ""
    master_prompt: str = ""
    negative_prompt: str = ""
    textless_default: bool = True
    is_draft: bool = True

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ChannelGenerationPreset":
        values = {field: data.get(field) for field in cls.__dataclass_fields__}
        values["preset_id"] = str(values.get("preset_id") or new_id("preset"))
        values["name"] = str(values.get("name") or "새 생성 채널")
        values["version"] = int(values.get("version") or 1)
        values["textless_default"] = bool(values.get("textless_default", True))
        values["is_draft"] = bool(values.get("is_draft", True))
        for field_name in (set(cls.__dataclass_fields__) - {"preset_id", "name", "version", "textless_default", "is_draft"}):
            values[field_name] = str(values.get(field_name) or "")
        return cls(**values)

    def to_dict(self) -> dict[str, Any]:
        return {field: getattr(self, field) for field in self.__dataclass_fields__}


def default_generation_presets() -> list[ChannelGenerationPreset]:
    names = [
        ("시니어", "OldPopLounge"),
        ("Tokyo ChillRap Love Story", "남녀 이야기"),
        ("Tokyo ChillRap", "남자 시점"),
        ("Tokyo ChillRap", "여자 시점"),
        ("Tokyo ChillRap", "카페"),
    ]
    return [
        ChannelGenerationPreset(
            preset_id=f"builtin_{index + 1}",
            name=f"{channel} / {variant}",
            style="사용자 지정",
            mood=f"{variant} 분위기 초안",
            era_region="사용자 입력 전까지 미정",
            master_prompt="글자 없는 이미지. 사용자 장면 설명을 중심으로 구성.",
            negative_prompt="text, typography, logo, watermark",
        )
        for index, (channel, variant) in enumerate(names)
    ]


@dataclass(slots=True)
class ReferenceImage:
    image_id: str
    path: str
    role: str = "person"
    use: str = "other"
    note: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ReferenceImage":
        role = str(data.get("role") or "person")
        use = str(data.get("use") or "other")
        return cls(str(data.get("image_id") or new_id("ref")), str(data.get("path") or ""), role if role in REFERENCE_ROLES else "person", use if use in REFERENCE_USES else "other", str(data.get("note") or ""))

    def to_dict(self) -> dict[str, Any]:
        return {"image_id": self.image_id, "path": self.path, "role": self.role, "use": self.use, "note": self.note}


@dataclass(slots=True)
class PersonRecord:
    person_id: str
    name: str
    appearance: str = ""
    hair: str = ""
    base_outfit: str = ""
    reference_images: list[ReferenceImage] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PersonRecord":
        refs = data.get("reference_images") or []
        return cls(str(data.get("person_id") or new_id("person")), str(data.get("name") or "새 인물"), str(data.get("appearance") or ""), str(data.get("hair") or ""), str(data.get("base_outfit") or ""), [ReferenceImage.from_dict(item) for item in refs if isinstance(item, dict)])

    def to_dict(self) -> dict[str, Any]:
        return {"person_id": self.person_id, "name": self.name, "appearance": self.appearance, "hair": self.hair, "base_outfit": self.base_outfit, "reference_images": [item.to_dict() for item in self.reference_images]}


@dataclass(slots=True)
class InputRecord:
    input_id: str
    input_type: str
    title: str = ""
    lyrics: str = ""
    image_prompt: str = ""
    theme_mood: str = ""
    source_path: str = ""
    selected: bool = True

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "InputRecord":
        return cls(str(data.get("input_id") or new_id("input")), str(data.get("input_type") or "theme_series_mood"), str(data.get("title") or ""), str(data.get("lyrics") or ""), str(data.get("image_prompt") or ""), str(data.get("theme_mood") or ""), str(data.get("source_path") or ""), bool(data.get("selected", True)))

    def to_dict(self) -> dict[str, Any]:
        return {field: getattr(self, field) for field in self.__dataclass_fields__}


@dataclass(slots=True)
class SceneCard:
    scene_id: str
    order: int = 1
    input_id: str = ""
    location: str = ""
    time_of_day: str = ""
    weather: str = ""
    action: str = ""
    emotion: str = ""
    composition: str = ""
    person_ids: list[str] = field(default_factory=list)
    reference_image_ids: list[str] = field(default_factory=list)
    outfit: str = ""
    output_ratio: str = "1:1"
    candidate_count: int = 1
    user_description: str = ""
    prompt_auto: str = ""
    negative_prompt_auto: str = ""
    prompt_user: str = ""
    negative_prompt_user: str = ""
    prompt_confirmed: bool = False
    prompt_source_preset_id: str = ""
    prompt_source_preset_version: int = 0
    structured_request: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SceneCard":
        values = {field: data.get(field) for field in cls.__dataclass_fields__}
        values["scene_id"] = str(values.get("scene_id") or new_id("scene"))
        values["order"] = int(values.get("order") or 1)
        values["candidate_count"] = max(1, int(values.get("candidate_count") or 1))
        values["person_ids"] = [str(x) for x in (values.get("person_ids") or [])]
        values["reference_image_ids"] = [str(x) for x in (values.get("reference_image_ids") or [])]
        values["structured_request"] = dict(values.get("structured_request") or {})
        for field_name in (set(cls.__dataclass_fields__) - {"scene_id", "order", "candidate_count", "person_ids", "reference_image_ids", "structured_request", "prompt_confirmed", "prompt_source_preset_version"}):
            values[field_name] = str(values.get(field_name) or "")
        values["prompt_confirmed"] = bool(values.get("prompt_confirmed", False))
        values["prompt_source_preset_version"] = int(values.get("prompt_source_preset_version") or 0)
        return cls(**values)

    def to_dict(self) -> dict[str, Any]:
        return {field: getattr(self, field) for field in self.__dataclass_fields__}


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
    scene_id: str = ""
    generation_status: str = "not_generated"
    quality_status: str = "unverified"
    generation_metadata: dict[str, Any] = field(default_factory=dict)
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
            scene_id=str(data.get("scene_id") or ""),
            generation_status=str(data.get("generation_status") or "not_generated"),
            quality_status=str(data.get("quality_status") or "unverified"),
            generation_metadata=dict(data.get("generation_metadata") or {}),
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
            "scene_id": self.scene_id,
            "generation_status": self.generation_status,
            "quality_status": self.quality_status,
            "generation_metadata": dict(self.generation_metadata),
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
    channel_preset_id: str = ""
    channel_preset: dict[str, Any] = field(default_factory=dict)
    people: list[PersonRecord] = field(default_factory=list)
    inputs: list[InputRecord] = field(default_factory=list)
    scenes: list[SceneCard] = field(default_factory=list)
    generation_runs: list[dict[str, Any]] = field(default_factory=list)
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
            channel_preset_id=str(data.get("channel_preset_id") or ""),
            channel_preset=dict(data.get("channel_preset") or {}),
            people=[PersonRecord.from_dict(item) for item in (data.get("people") or []) if isinstance(item, dict)],
            inputs=[InputRecord.from_dict(item) for item in (data.get("inputs") or []) if isinstance(item, dict)],
            scenes=[SceneCard.from_dict(item) for item in (data.get("scenes") or []) if isinstance(item, dict)],
            generation_runs=[dict(item) for item in (data.get("generation_runs") or []) if isinstance(item, dict)],
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
            "channel_preset_id": self.channel_preset_id,
            "channel_preset": dict(self.channel_preset),
            "people": [person.to_dict() for person in self.people],
            "inputs": [item.to_dict() for item in self.inputs],
            "scenes": [scene.to_dict() for scene in self.scenes],
            "generation_runs": [dict(item) for item in self.generation_runs],
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
    for relative in ("assets/originals", "assets/removal_previews", "assets/textless", "assets/references", "assets/references/processed", "assets/inputs", "assets/generated"):
        (project.project_dir / relative).mkdir(parents=True, exist_ok=True)


def load_generation_presets(path: Path | None = None) -> list[ChannelGenerationPreset]:
    """Load user-owned generation presets; built-ins are only used when absent."""
    preset_file = path or Path("config/channel_generation_presets.json")
    try:
        data = json.loads(preset_file.read_text(encoding="utf-8-sig"))
        if isinstance(data, list):
            return [ChannelGenerationPreset.from_dict(item) for item in data if isinstance(item, dict)]
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        pass
    return default_generation_presets()


def save_generation_presets(presets: list[ChannelGenerationPreset], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temp.write_text(json.dumps([item.to_dict() for item in presets], ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temp, path)
    finally:
        try:
            temp.unlink()
        except FileNotFoundError:
            pass


def add_person(project: CoverMorphProject, name: str = "새 인물") -> PersonRecord:
    person = PersonRecord(person_id=new_id("person"), name=name)
    project.people.append(person)
    return person


def add_person_reference(project: CoverMorphProject, person: PersonRecord, source_path: Path, role: str, use: str = "other", note: str = "") -> ReferenceImage:
    if role not in REFERENCE_ROLES or use not in REFERENCE_USES:
        raise ProjectAssetError(f"Unsupported reference role/use: {role}/{use}")
    if not source_path.is_file():
        raise ProjectAssetError(f"Reference image not found: {source_path}")
    ensure_project_dirs(project)
    image_id = new_id("ref")
    dest = project.project_dir / "assets" / "references" / f"{image_id}{_safe_suffix(source_path)}"
    shutil.copy2(source_path, dest)
    _load_rgb_image(dest).close()
    reference = ReferenceImage(image_id, path_to_project_string(project, dest), role, use, note)
    person.reference_images.append(reference)
    return reference


def prepare_reference_image(project: CoverMorphProject, reference: ReferenceImage, crop_box: tuple[int, int, int, int] | None = None) -> tuple[Path, dict[str, Any]]:
    """Validate, orient, RGB-normalize, and optionally crop one reference without touching its source."""
    source = resolve_project_path(project, reference.path)
    if not source.is_file():
        raise ProjectAssetError(f"Reference image is missing: {source}")
    try:
        source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
        with Image.open(source) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
    except (OSError, UnidentifiedImageError) as exc:
        raise ProjectAssetError(f"Reference image is damaged or unreadable: {source}") from exc
    processed = image
    crop_key = "full"
    if crop_box is not None:
        left, top, right, bottom = (int(value) for value in crop_box)
        if not (0 <= left < right <= image.width and 0 <= top < bottom <= image.height):
            raise ProjectAssetError("Reference crop box is outside the image.")
        processed = image.crop((left, top, right, bottom))
        crop_key = f"{left}_{top}_{right}_{bottom}"
    cache_key = hashlib.sha256(f"{source_hash}|{crop_key}|exif_transpose|rgb|ip-adapter-sdxl-v1".encode()).hexdigest()[:20]
    destination = project.project_dir / "assets" / "references" / "processed" / f"{reference.image_id}_{cache_key}.png"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.is_file():
        processed.save(destination, "PNG")
    return destination, {"source_sha256": source_hash, "processed_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(), "crop_box": list(crop_box) if crop_box else None, "preprocess": "EXIF transpose, RGB, optional user crop", "cache_key": cache_key}


def parse_input_file(path: Path) -> list[InputRecord]:
    """Parse common TXT/JSON shapes without pretending to understand unknown schemas."""
    try:
        raw = path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeError) as exc:
        raise ProjectLoadError(f"Input file could not be read as UTF-8: {path}") from exc
    if path.suffix.lower() == ".txt":
        return [InputRecord(new_id("input"), "lyrics", path.stem, lyrics=raw, source_path=str(path))]
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ProjectLoadError(f"Input JSON is damaged: {path}") from exc
    items = data if isinstance(data, list) else data.get("songs") or data.get("tracks") or data.get("items") if isinstance(data, dict) else None
    if isinstance(items, list):
        records = []
        for item in items:
            if not isinstance(item, dict):
                continue
            records.append(InputRecord(new_id("input"), "lyrics", str(item.get("title") or item.get("name") or ""), str(item.get("lyrics") or item.get("lyric") or ""), str(item.get("prompt") or item.get("image_prompt") or ""), str(item.get("theme") or item.get("mood") or ""), str(path)))
        if records:
            return records
    if isinstance(data, dict):
        known = {"title", "name", "lyrics", "lyric", "prompt", "image_prompt", "theme", "mood"}
        if known.intersection(data):
            return [InputRecord(new_id("input"), "lyrics", str(data.get("title") or data.get("name") or path.stem), str(data.get("lyrics") or data.get("lyric") or ""), str(data.get("prompt") or data.get("image_prompt") or ""), str(data.get("theme") or data.get("mood") or ""), str(path))]
    raise ProjectLoadError(f"Unknown JSON input structure; choose title/lyrics/prompt fields: {path}")


def add_input_records(project: CoverMorphProject, records: list[InputRecord]) -> None:
    ensure_project_dirs(project)
    for record in records:
        source = Path(record.source_path)
        if not source.is_file():
            raise ProjectAssetError(f"Input source is missing: {source}")
        destination = project.project_dir / "assets" / "inputs" / f"{record.input_id}{source.suffix.lower() or '.txt'}"
        shutil.copy2(source, destination)
        record.source_path = path_to_project_string(project, destination)
        project.inputs.append(record)
    project.song_count = len(project.inputs)


def compose_scene_prompt(project: CoverMorphProject, scene: SceneCard, preset: ChannelGenerationPreset) -> tuple[str, str, dict[str, Any]]:
    people = {person.person_id: person for person in project.people}
    person_text = []
    refs = []
    for person_id in scene.person_ids:
        person = people.get(person_id)
        if person:
            person_text.append(f"{person.name}: {person.appearance}; hair: {person.hair}; outfit: {person.base_outfit}")
            refs.extend(ref.image_id for ref in person.reference_images if ref.image_id in scene.reference_image_ids)
    parts = [preset.master_prompt, scene.user_description, scene.location, scene.time_of_day, scene.weather, scene.action, scene.emotion, scene.composition, scene.outfit, " ".join(person_text)]
    prompt = ", ".join(part.strip() for part in parts if part and part.strip())
    negative = ", ".join(part.strip() for part in (preset.negative_prompt, preset.avoid_elements, "text, typography, logo, watermark") if part and part.strip())
    request = {"preset_id": preset.preset_id, "preset_version": preset.version, "scene_id": scene.scene_id, "person_ids": list(scene.person_ids), "reference_image_ids": refs, "output_ratio": scene.output_ratio, "candidate_count": scene.candidate_count, "textless": preset.textless_default}
    return prompt, negative, request


def configure_scene_prompt(project: CoverMorphProject, scene: SceneCard, preset: ChannelGenerationPreset, *, refresh: bool = False) -> None:
    if scene.prompt_confirmed and not refresh:
        return
    scene.prompt_auto, scene.negative_prompt_auto, scene.structured_request = compose_scene_prompt(project, scene, preset)
    scene.prompt_user = scene.prompt_auto
    scene.negative_prompt_user = scene.negative_prompt_auto
    scene.prompt_source_preset_id = preset.preset_id
    scene.prompt_source_preset_version = preset.version
    scene.prompt_confirmed = False


def duplicate_scene(scene: SceneCard) -> SceneCard:
    copied = SceneCard.from_dict(scene.to_dict())
    copied.scene_id = new_id("scene")
    copied.order = scene.order + 1
    copied.prompt_confirmed = False
    return copied


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


def add_generated_candidate(
    project: CoverMorphProject,
    scene: SceneCard,
    image_path: Path,
    metadata: dict[str, Any],
) -> CandidateRecord:
    """Register a generated PNG without treating it as a reviewed work source."""
    if not image_path.is_file():
        raise ProjectAssetError(f"Generated image is missing: {image_path}")
    ensure_project_dirs(project)
    candidate_id = new_id("gen")
    destination = project.project_dir / "assets" / "generated" / f"{candidate_id}.png"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(image_path, destination)
    _load_rgb_image(destination).close()
    candidate = CandidateRecord(
        candidate_id=candidate_id,
        display_name=f"{scene.scene_id} candidate",
        input_type=INPUT_TYPE_TEXTLESS,
        original_path=path_to_project_string(project, destination),
        working_source_path="",
        selected=False,
        scene_id=scene.scene_id,
        generation_status="succeeded",
        quality_status="unverified",
        generation_metadata=dict(metadata),
        text_removal_status="not_needed",
        text_removal_engine="Skipped",
        user_approval_status="pending",
        working_source_approved=False,
        created_at=utc_now(),
        updated_at=utc_now(),
    )
    project.candidates.append(candidate)
    return candidate


def adopt_generated_candidate(project: CoverMorphProject, candidate: CandidateRecord) -> Path:
    if candidate.generation_status != "succeeded" or not candidate.original_path:
        raise ProjectAssetError("Only a successfully generated candidate can be adopted.")
    source = resolve_project_path(project, candidate.original_path)
    if not source.is_file():
        raise ProjectAssetError(f"Generated candidate is missing: {source}")
    destination = project.project_dir / "assets" / "textless" / f"{candidate.candidate_id}_approved.png"
    shutil.copy2(source, destination)
    candidate.working_source_path = path_to_project_string(project, destination)
    candidate.working_source_approved = True
    candidate.user_approval_status = "approved_by_user"
    candidate.quality_status = "user_selected_unverified_quality"
    candidate.updated_at = utc_now()
    return destination


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
    for person in project.people:
        for reference in person.reference_images:
            resolved = resolve_project_path(project, reference.path)
            if not resolved.is_file():
                issues.append(ProjectIssue(person.person_id, "reference", reference.path, "Person reference image is missing."))
    for record in project.inputs:
        if record.source_path and not resolve_project_path(project, record.source_path).is_file():
            issues.append(ProjectIssue(record.input_id, "input", record.source_path, "Input source is missing."))
    return issues

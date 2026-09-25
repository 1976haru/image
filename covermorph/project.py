from __future__ import annotations

import copy
import hashlib
import io
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


REFERENCE_ROLES = {"lyrics_series", "image_master_prompt", "person", "style", "background_composition"}
REFERENCE_USES = {"face", "upper_body", "full_body", "other"}
INPUT_RECORD_TYPES = {"lyrics", "image_prompt", "theme_series_mood"}

CREATION_PURPOSES = {
    "music_cover_candidate",
    "thumbnail_background",
    "video_background",
    "shorts_background",
    "shopify_app_image",
}
INPUT_MODES = {
    "lyrics",
    "json_file",
    "keywords",
    "master_prompt",
    "reference_images",
    "image_prompt",
}
CANDIDATE_COUNTS = (1, 4, 5, 6, 8, 10)
REFERENCE_ASSET_ROLES = {"lyrics_series", "image_master_prompt", "person", "style", "background_composition"}


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
    default_label: str = ""
    default_character: str = ""
    preferred_places_compositions: str = ""
    cover_text_language: str = ""
    lyrics_language: str = "auto"
    story_viewpoint: str = "neutral"
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
    shared_negative = "text, typography, logo, watermark, oversaturated colors, excessive contrast"
    return [
        ChannelGenerationPreset("builtin_senior_kr", "한국 시니어", style="자연스러운 실사", mood="잔잔함, 추억, 계절감", color="절제된 자연색", default_label="OldPopLounge", preferred_places_compositions="계절이 느껴지는 일상 공간과 풍경, 자연스러운 중거리 구도", avoid_elements="과한 주황빛, 과포화, 과도한 대비, 획일적인 흑백", cover_text_language="한국어 또는 사용자 지정", master_prompt="natural photorealistic senior music cover, quiet memory, seasonal atmosphere, restrained natural colors, textless image", negative_prompt=shared_negative, is_draft=False),
        ChannelGenerationPreset("builtin_jp_en", "일본 칠리랩 영어 버전", style="실사 또는 사용자 선택 일러스트", mood="도시의 일상과 관계 감정", default_label="Tokyo ChillRap", cover_text_language="영어", master_prompt="Tokyo chill rap cover, everyday relationship emotion, cinematic Japanese urban atmosphere, textless image", negative_prompt=shared_negative, is_draft=False),
        ChannelGenerationPreset("builtin_jp_ja", "일본 칠리랩 일본어 버전", style="실사 또는 사용자 선택 일러스트", mood="도시의 일상과 관계 감정", default_label="Tokyo ChillRap", cover_text_language="일본어", master_prompt="Tokyo chill rap cover, everyday relationship emotion, cinematic urban atmosphere, textless image", negative_prompt=shared_negative, is_draft=False),
        ChannelGenerationPreset("builtin_jp_male", "일본 칠리랩 남자 이야기", style="실사 또는 사용자 선택 일러스트", mood="남자 주인공의 일상, 관계, 감정, 행동", default_label="Tokyo ChillRap", story_viewpoint="male protagonist", cover_text_language="사용자 지정", master_prompt="Tokyo chill rap story cover, male protagonist viewpoint, everyday relationship and emotion, textless image", negative_prompt=shared_negative, is_draft=False),
        ChannelGenerationPreset("builtin_jp_female", "일본 칠리랩 여자 이야기", style="실사 또는 사용자 선택 일러스트", mood="여자 주인공의 일상, 관계, 감정, 행동", default_label="Tokyo ChillRap", story_viewpoint="female protagonist", cover_text_language="사용자 지정", master_prompt="Tokyo chill rap story cover, female protagonist viewpoint, everyday relationship and emotion, textless image", negative_prompt=shared_negative, is_draft=False),
        ChannelGenerationPreset("builtin_jp_cafe", "일본 칠리랩 카페", style="실사 또는 사용자 선택 일러스트", mood="카페 공간과 인물의 균형", default_label="Tokyo ChillRap", preferred_places_compositions="카페 공간 중심, 인물 중심, 인물 없는 장면 모두 허용", cover_text_language="사용자 지정", master_prompt="Tokyo chill rap cafe cover, balanced cafe space and human presence, optional empty interior, textless image", negative_prompt=shared_negative, is_draft=False),
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
    music_prompt: str = ""
    series_description: str = ""
    source_path: str = ""
    selected: bool = True

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "InputRecord":
        return cls(str(data.get("input_id") or new_id("input")), str(data.get("input_type") or "theme_series_mood"), str(data.get("title") or ""), str(data.get("lyrics") or ""), str(data.get("image_prompt") or ""), str(data.get("theme_mood") or ""), str(data.get("music_prompt") or ""), str(data.get("series_description") or ""), str(data.get("source_path") or ""), bool(data.get("selected", True)))

    def to_dict(self) -> dict[str, Any]:
        return {field: getattr(self, field) for field in self.__dataclass_fields__}


@dataclass(slots=True)
class ImagePlanningBrief:
    """User-confirmed bridge from lyrics/music material to an image scene.

    Extraction is deliberately rule-based.  The fields are a reviewable draft,
    never presented as an AI interpretation of the complete lyrics.
    """

    core_subject: str = ""
    emotion: str = ""
    location: str = ""
    time_or_season: str = ""
    characters: str = ""
    action: str = ""
    props: str = ""
    brightness_color: str = ""
    source_input_ids: list[str] = field(default_factory=list)
    extraction_method: str = "rule_based_draft_user_confirmation_required"
    confirmed: bool = False

    @classmethod
    def from_dict(cls, data: Any) -> "ImagePlanningBrief":
        if not isinstance(data, dict):
            return cls()
        values = {name: str(data.get(name) or "") for name in ("core_subject", "emotion", "location", "time_or_season", "characters", "action", "props", "brightness_color")}
        ids = data.get("source_input_ids") or []
        values["source_input_ids"] = [str(item) for item in ids] if isinstance(ids, list) else []
        values["extraction_method"] = str(data.get("extraction_method") or cls.extraction_method)
        values["confirmed"] = bool(data.get("confirmed", False))
        return cls(**values)

    def to_dict(self) -> dict[str, Any]:
        return {
            "core_subject": self.core_subject,
            "emotion": self.emotion,
            "location": self.location,
            "time_or_season": self.time_or_season,
            "characters": self.characters,
            "action": self.action,
            "props": self.props,
            "brightness_color": self.brightness_color,
            "source_input_ids": list(self.source_input_ids),
            "extraction_method": self.extraction_method,
            "confirmed": self.confirmed,
        }


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


def cover_plan_version(plan: dict[str, Any]) -> str:
    """Stable version for the exact user-approved planning-card inputs."""
    payload = {key: plan.get(key) for key in ("plan_id", "scene_ko", "characters_action", "place_time_weather_season", "background_props", "composition_distance", "brightness_color", "image_prompt_en", "negative_prompt", "title", "reference")}
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def scene_from_approved_cover_plan(plan: dict[str, Any], order: int = 1) -> SceneCard:
    """Build one frozen, textless SDXL scene only from an explicit user approval."""
    if str(plan.get("user_decision") or "") != "approved_for_generation":
        raise ProjectError("기획 카드는 사용자가 최종 입력을 승인한 뒤에만 생성할 수 있습니다.")
    approval = dict(plan.get("generation_approval") or {})
    version = cover_plan_version(plan)
    if not approval.get("approved") or approval.get("card_version") != version:
        raise ProjectError("기획 카드가 수정되었거나 최종 생성 입력 승인이 없습니다.")
    title = dict(plan.get("title") or {})
    return SceneCard(
        scene_id=new_id("scene"),
        order=order,
        input_id=str((plan.get("related_input_ids") or [""])[0]),
        location=str(plan.get("place_time_weather_season") or ""),
        action=str(plan.get("characters_action") or ""),
        emotion=str(plan.get("connection_reason") or ""),
        composition=str(plan.get("composition_distance") or ""),
        output_ratio="1:1",
        candidate_count=1,
        user_description=str(plan.get("scene_ko") or ""),
        prompt_auto=str(plan.get("image_prompt_en") or ""),
        negative_prompt_auto=str(plan.get("negative_prompt") or ""),
        prompt_user=str(plan.get("image_prompt_en") or ""),
        negative_prompt_user=str(plan.get("negative_prompt") or ""),
        prompt_confirmed=True,
        prompt_source_preset_id="cover_plan",
        prompt_source_preset_version=1,
        structured_request={
            "cover_plan_id": str(plan.get("plan_id") or ""),
            "cover_plan_version": version,
            "cover_plan_title": title,
            "reference_mode": "off",
            "reference_image_id": "",
            "reference_strength": 0.0,
            "reference_crop_box": None,
            "candidate_options": {"variation": "manual_per_candidate", "candidate_count": 1},
            "textless": True,
            "approval_snapshot": {
                "scene_ko": str(plan.get("scene_ko") or ""),
                "image_prompt_en": str(plan.get("image_prompt_en") or ""),
                "negative_prompt": str(plan.get("negative_prompt") or ""),
            },
        },
    )

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
    creation_purpose: str = "music_cover_candidate"
    primary_input_mode: str = "lyrics"
    input_selection_scope: str = "single"
    selected_input_ids: list[str] = field(default_factory=list)
    input_materials: dict[str, str] = field(default_factory=dict)
    image_planning_brief: ImagePlanningBrief = field(default_factory=ImagePlanningBrief)
    cover_text: dict[str, str] = field(default_factory=dict)
    selected_reference_ids: dict[str, str] = field(default_factory=dict)
    candidate_options: dict[str, Any] = field(default_factory=dict)
    song_count: int = 0
    channel_preset_id: str = ""
    channel_preset: dict[str, Any] = field(default_factory=dict)
    people: list[PersonRecord] = field(default_factory=list)
    inputs: list[InputRecord] = field(default_factory=list)
    scenes: list[SceneCard] = field(default_factory=list)
    generation_runs: list[dict[str, Any]] = field(default_factory=list)
    cover_planning: dict[str, Any] = field(default_factory=dict)
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
            creation_purpose=str(data.get("creation_purpose") or "music_cover_candidate") if str(data.get("creation_purpose") or "music_cover_candidate") in CREATION_PURPOSES else "music_cover_candidate",
            primary_input_mode=str(data.get("primary_input_mode") or "lyrics") if str(data.get("primary_input_mode") or "lyrics") in INPUT_MODES else "lyrics",
            input_selection_scope=str(data.get("input_selection_scope") or "single") if str(data.get("input_selection_scope") or "single") in {"single", "album"} else "single",
            selected_input_ids=[str(item) for item in (data.get("selected_input_ids") or []) if item is not None],
            input_materials={str(key): str(value) for key, value in (data.get("input_materials") or {}).items() if value is not None},
            image_planning_brief=ImagePlanningBrief.from_dict(data.get("image_planning_brief")),
            cover_text={str(key): str(value) for key, value in (data.get("cover_text") or {}).items() if value is not None},
            selected_reference_ids={str(key): str(value) for key, value in (data.get("selected_reference_ids") or {}).items() if value is not None},
            candidate_options=dict(data.get("candidate_options") or {}),
            song_count=int(data.get("song_count") or 0),
            channel_preset_id=str(data.get("channel_preset_id") or ""),
            channel_preset=dict(data.get("channel_preset") or {}),
            people=[PersonRecord.from_dict(item) for item in (data.get("people") or []) if isinstance(item, dict)],
            inputs=[InputRecord.from_dict(item) for item in (data.get("inputs") or []) if isinstance(item, dict)],
            scenes=[SceneCard.from_dict(item) for item in (data.get("scenes") or []) if isinstance(item, dict)],
            generation_runs=[dict(item) for item in (data.get("generation_runs") or []) if isinstance(item, dict)],
            cover_planning=dict(data.get("cover_planning") or {}),
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
            "creation_purpose": self.creation_purpose,
            "primary_input_mode": self.primary_input_mode,
            "input_selection_scope": self.input_selection_scope,
            "selected_input_ids": list(self.selected_input_ids),
            "input_materials": dict(self.input_materials),
            "image_planning_brief": self.image_planning_brief.to_dict(),
            "cover_text": dict(self.cover_text),
            "selected_reference_ids": dict(self.selected_reference_ids),
            "candidate_options": dict(self.candidate_options),
            "song_count": self.song_count,
            "channel_preset_id": self.channel_preset_id,
            "channel_preset": dict(self.channel_preset),
            "people": [person.to_dict() for person in self.people],
            "inputs": [item.to_dict() for item in self.inputs],
            "scenes": [scene.to_dict() for scene in self.scenes],
            "generation_runs": [dict(item) for item in self.generation_runs],
            "cover_planning": copy.deepcopy(self.cover_planning),
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
        data = source.read_bytes()
        source_hash = hashlib.sha256(data).hexdigest()
        with Image.open(io.BytesIO(data)) as opened:
            rgba = ImageOps.exif_transpose(opened).convert("RGBA")
            image = Image.alpha_composite(Image.new("RGBA", rgba.size, "white"), rgba).convert("RGB")
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
    cache_key = hashlib.sha256(f"{source_hash}|{crop_key}|exif_transpose|white-alpha-rgb-v2".encode()).hexdigest()[:20]
    destination = project.project_dir / "assets" / "references" / "processed" / f"{reference.image_id}_{cache_key}.png"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.is_file():
        processed.save(destination, "PNG")
    try:
        with Image.open(destination) as cached:
            cached.verify()
    except (OSError, UnidentifiedImageError) as exc:
        raise ProjectAssetError(f"Processed reference cache is damaged: {destination}") from exc
    return destination, {
        "source_sha256": source_hash,
        "processed_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
        "crop_box": list(crop_box) if crop_box else None,
        "preprocessing": "EXIF transpose, alpha on white, RGB, optional user crop",
        "preprocess": "EXIF transpose, alpha on white, RGB, optional user crop",
        "cache_key": cache_key,
    }


def _json_field_value(item: dict[str, Any], aliases: tuple[str, ...]) -> str:
    for key in aliases:
        value = item.get(key)
        if value is not None and not isinstance(value, (dict, list)):
            return str(value)
    return ""


def parse_input_file_detailed(path: Path, field_mapping: dict[str, str] | None = None) -> dict[str, Any]:
    """Read UTF-8/BOM input and return records plus explicit schema information.

    Unknown JSON structures are rejected.  A caller may provide a field mapping
    after showing the available keys to the user; no unknown key is guessed.
    """
    try:
        raw = path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeError) as exc:
        raise ProjectLoadError(f"Input file could not be read as UTF-8: {path}") from exc
    if path.suffix.lower() == ".txt":
        record = InputRecord(new_id("input"), "lyrics", path.stem, lyrics=raw, source_path=str(path))
        return {"records": [record], "available_fields": ["lyrics"], "field_mapping": {"title": "", "lyrics": "lyrics"}, "needs_mapping": False, "source_path": str(path)}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ProjectLoadError(f"Input JSON is damaged: {path} (line {exc.lineno}, column {exc.colno})") from exc
    cover_text = {"title": "", "subtitle": "", "label": ""}
    if isinstance(data, dict):
        for target, aliases in {
            "title": ("album_title", "albumTitle", "release_title"),
            "subtitle": ("album_subtitle", "albumSubtitle", "subtitle"),
            "label": ("channel_name", "channelName", "label", "record_label"),
        }.items():
            cover_text[target] = _json_field_value(data, aliases)
    collection_key = "root"
    items: Any = data
    if isinstance(data, dict):
        for key in ("songs", "tracks", "items", "records", "曲목록", "곡목록"):
            if isinstance(data.get(key), list):
                collection_key, items = key, data[key]
                break
        else:
            items = [data] if any(key in data for key in ("title", "name", "lyrics", "lyric", "prompt", "image_prompt", "theme", "mood", "music_prompt", "series_description")) else None
            if items is None and field_mapping and any(field_mapping.values()):
                items = [data]
    if not isinstance(items, list) or not items or not all(isinstance(item, dict) for item in items):
        available = sorted(data.keys()) if isinstance(data, dict) else []
        if isinstance(data, dict):
            return {"records": [], "available_fields": available, "field_mapping": {}, "needs_mapping": True, "collection_key": collection_key, "source_path": str(path), "cover_text": cover_text}
        raise ProjectLoadError(f"Unknown JSON input structure: {path}; choose a song-list and field mapping")
    available_fields = sorted({str(key) for item in items for key in item})
    mapping = dict(field_mapping or {})
    aliases = {
        "title": ("title", "name", "곡명", "제목"),
        "lyrics": ("lyrics", "lyric", "가사"),
        "image_prompt": ("image_prompt", "prompt", "imagePrompt", "이미지프롬프트"),
        "theme_mood": ("theme", "mood", "series_description", "series_mood", "주제", "분위기"),
        "music_prompt": ("music_prompt", "suno_prompt", "musicPrompt", "음악프롬프트"),
    }
    for target, names in aliases.items():
        if target not in mapping:
            mapping[target] = next((name for name in names if name in available_fields), "")
    needs_mapping = not any(mapping.get(key) for key in ("title", "lyrics", "image_prompt", "theme_mood", "music_prompt"))
    if needs_mapping:
        return {"records": [], "available_fields": available_fields, "field_mapping": mapping, "needs_mapping": True, "collection_key": collection_key, "source_path": str(path), "cover_text": cover_text}
    records = []
    for index, item in enumerate(items, start=1):
        title = _json_field_value(item, (mapping.get("title", ""),)) or f"{path.stem} {index}"
        lyrics = _json_field_value(item, (mapping.get("lyrics", ""),))
        image_prompt = _json_field_value(item, (mapping.get("image_prompt", ""),))
        mood = _json_field_value(item, (mapping.get("theme_mood", ""),))
        music_prompt = _json_field_value(item, (mapping.get("music_prompt", ""),))
        records.append(InputRecord(new_id("input"), "lyrics", title, lyrics=lyrics, image_prompt=image_prompt, theme_mood=mood, music_prompt=music_prompt, source_path=str(path)))
    return {"records": records, "available_fields": available_fields, "field_mapping": mapping, "needs_mapping": False, "collection_key": collection_key, "source_path": str(path), "cover_text": cover_text}


def parse_input_file(path: Path) -> list[InputRecord]:
    result = parse_input_file_detailed(path)
    if result["needs_mapping"]:
        raise ProjectLoadError(f"Unknown JSON input structure; available keys={result['available_fields']}: {path}")
    return result["records"]


def rule_based_image_planning(records: list[InputRecord], keywords: str = "") -> ImagePlanningBrief:
    """Create a transparent, editable draft; this is not semantic AI analysis."""
    selected = [record for record in records if record.selected]
    text = " ".join([keywords, *(record.theme_mood for record in selected), *(record.image_prompt for record in selected)]).strip()
    tokens = [token.strip(" ,。.!?、") for token in text.split() if token.strip(" ,。.!?、")]
    return ImagePlanningBrief(core_subject=keywords or (tokens[0] if tokens else ""), emotion="", location="", time_or_season="", characters="", action="", props="", brightness_color="", source_input_ids=[record.input_id for record in selected], extraction_method="rule_based_draft_user_confirmation_required", confirmed=False)


def validate_candidate_count(count: int) -> int:
    if count not in CANDIDATE_COUNTS:
        raise ProjectError(f"Candidate count must be one of {CANDIDATE_COUNTS}; received {count}")
    return count


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
    reference_settings = {key: value for key, value in scene.structured_request.items() if key.startswith("reference_")}
    scene.prompt_auto, scene.negative_prompt_auto, scene.structured_request = compose_scene_prompt(project, scene, preset)
    scene.structured_request.update(reference_settings)
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

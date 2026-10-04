"""Creator workflow definitions: purposes (YouTube / Shopify / custom), Korean UI labels, prompt presets.

Everything here is data. Built-ins can be overridden or extended by ``config/creator_presets.json``
(same shape) so new Shopify themes need no code change. Regions are normalized (x, y, w, h) in 0..1.
Prompt presets never add nationality/appearance words unless their ``appearance_hint`` says so.
"""
from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# ------------------------------------------------------------------ UI labels -> engine settings
QUALITY_LABELS = {"빠른 미리보기": "preview", "일반": "balanced", "최고 품질": "best"}
MEMORY_LABELS = {"작업 중 PC 우선": "interactive_low_memory", "균형": "balanced_idle",
                 "자리 비움/최고 품질": "night_best"}
DEFAULT_QUALITY_LABEL = "일반"
DEFAULT_MEMORY_LABEL = "작업 중 PC 우선"  # the user is usually at the PC: favour low memory
ROLE_LABELS = {"인물": "PERSON", "상품": "PRODUCT", "스타일": "STYLE", "구도": "COMPOSITION", "배경": "BACKGROUND"}
ROLE_NAMES = {value: key for key, value in ROLE_LABELS.items()}


def label_for(mapping: dict[str, str], value: str) -> str:
    return next((label for label, key in mapping.items() if key == value), value)


# ------------------------------------------------------------------ purposes
@dataclass(slots=True)
class Purpose:
    key: str
    label: str
    width: int
    height: int
    compiler_purpose: str            # envelope wording in prompt_compilers.PURPOSES
    text_region: tuple[float, float, float, float]
    subject_region: tuple[float, float, float, float]
    cta_region: tuple[float, float, float, float] | None = None
    text_side: str = "left"          # where the compiled prompt asks for calm negative space
    alternates: dict[str, tuple[int, int]] = field(default_factory=dict)
    exports: tuple[str, ...] = ("jpg", "png")
    youtube: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


BUILTIN_PURPOSES: dict[str, Purpose] = {p.key: p for p in [
    Purpose("youtube_thumbnail", "YouTube 썸네일", 1280, 720, "youtube_thumbnail_background",
            text_region=(0.04, 0.10, 0.44, 0.80), subject_region=(0.50, 0.05, 0.48, 0.95), youtube=True),
    Purpose("shopify_hero", "Shopify 히어로 배너", 1800, 700, "shopify_hero_banner",
            text_region=(0.05, 0.18, 0.40, 0.64), subject_region=(0.52, 0.06, 0.44, 0.88),
            cta_region=(0.05, 0.72, 0.20, 0.13)),
    Purpose("shopify_collection", "Shopify 컬렉션 배너", 1600, 900, "shopify_collection_banner",
            text_region=(0.05, 0.20, 0.42, 0.60), subject_region=(0.50, 0.08, 0.46, 0.84),
            alternates={"정사각형": (1200, 1200)}),
    Purpose("shopify_product_lifestyle", "Shopify 상품 라이프스타일", 1600, 1200, "shopify_product_lifestyle",
            text_region=(0.05, 0.05, 0.90, 0.20), subject_region=(0.20, 0.25, 0.60, 0.70), text_side="top"),
    Purpose("shopify_promo_tile", "Shopify 프로모션 타일", 1080, 1080, "shopify_promo_tile",
            text_region=(0.08, 0.06, 0.84, 0.26), subject_region=(0.15, 0.34, 0.70, 0.60),
            cta_region=(0.30, 0.86, 0.40, 0.08), text_side="top"),
    Purpose("shopify_mobile", "Shopify 모바일 배너", 1080, 1350, "shopify_mobile_banner",
            text_region=(0.08, 0.05, 0.84, 0.24), subject_region=(0.10, 0.32, 0.80, 0.62),
            cta_region=(0.28, 0.88, 0.44, 0.07), text_side="top"),
    Purpose("custom", "사용자 지정 캔버스", 1280, 720, "youtube_thumbnail_background",
            text_region=(0.04, 0.10, 0.44, 0.80), subject_region=(0.50, 0.05, 0.48, 0.95)),
]}


# ------------------------------------------------------------------ prompt presets
@dataclass(slots=True)
class PromptPreset:
    key: str
    label: str
    purpose: str
    prompt: str
    channel: str = ""
    people: int = 0
    composition: str = ""
    locale: str = ""
    language: str = ""
    audience: str = ""
    appearance_hint: str = ""        # used verbatim only when set (e.g. "Japanese")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


OLD_POP_LIVED = ("candid unposed moment, natural skin texture with fine lines and real complexion, lived-in setting "
                 "with personal objects, {light}, gentle film grain, not a stock photo, not a studio")

BUILTIN_PROMPT_PRESETS: dict[str, PromptPreset] = {p.key: p for p in [
    PromptPreset("tc_solo_woman", "Tokyo Chill · 여성 1인", "youtube_thumbnail",
                 "side profile of a young woman on a quiet Tokyo street at dusk", "Tokyo Chill", 1, "SOLO_MEDIUM",
                 locale="ja-JP", language="ja", audience="Japanese city-pop listeners", appearance_hint="Japanese"),
    PromptPreset("tc_solo_man", "Tokyo Chill · 남성 1인", "youtube_thumbnail",
                 "a young man standing on a train platform in the evening", "Tokyo Chill", 1, "SOLO_MEDIUM",
                 locale="ja-JP", language="ja", audience="Japanese city-pop listeners", appearance_hint="Japanese"),
    PromptPreset("tc_couple", "Tokyo Chill · 커플", "youtube_thumbnail",
                 "a young man and woman couple sharing a quiet moment by a cafe window, she glances at him and he "
                 "smiles softly, candid and unposed, everyday clothes, the city street softly visible outside",
                 "Tokyo Chill", 2, "COUPLE_CLOSE",
                 locale="ja-JP", language="ja", audience="Japanese city-pop listeners", appearance_hint="Japanese"),
    # OLD POP (RC2): lived-in, story-telling wording replaced the plain "in an autumn park" style, which read as stock
    # photos with flat light. No appearance hint by default (do not force Asian or Western); set one in the JSON or
    # write it in the prompt if the channel needs it.
    PromptPreset("op_mature_couple", "OLD POP · 중년 커플", "youtube_thumbnail",
                 "a mature man and woman couple in their fifties walking arm in arm on a quiet autumn path, laughing at "
                 "a shared memory, " + OLD_POP_LIVED.format(light="soft late-afternoon light"),
                 "OLD POP LOUNGE", 2, "COUPLE_MEDIUM"),
    PromptPreset("op_mature_solo", "OLD POP · 중년 여성", "youtube_thumbnail",
                 "a mature woman in her fifties sitting by the window of a small retro coffee shop she has visited for "
                 "years, holding a warm cup, " + OLD_POP_LIVED.format(light="warm practical lamp light"),
                 "OLD POP LOUNGE", 1, "SOLO_CLOSE"),
    PromptPreset("op_mature_man", "OLD POP · 중년 남성", "youtube_thumbnail",
                 "a mature man in his fifties at the counter of an old record shop, flipping through vinyl records, "
                 + OLD_POP_LIVED.format(light="warm practical lamp light"), "OLD POP LOUNGE", 1, "SOLO_MEDIUM"),
    PromptPreset("op_seasonal", "OLD POP · 계절 장면", "youtube_thumbnail",
                 "a mature woman in her fifties stepping out of a small neighborhood bakery into the first snow of "
                 "winter, scarf and wool coat, " + OLD_POP_LIVED.format(light="warm shop light spilling onto the snowy street"),
                 "OLD POP LOUNGE", 1, "SOLO_CLOSE"),
    PromptPreset("sh_clean_studio", "Shopify · 깔끔한 스튜디오", "shopify_product_lifestyle",
                 "the product on a clean seamless studio backdrop with soft even light and a gentle shadow"),
    PromptPreset("sh_editorial", "Shopify · 에디토리얼 라이프스타일", "shopify_hero",
                 "an airy editorial living space with linen textiles, ceramic vases, light oak furniture and soft "
                 "directional light"),
    PromptPreset("sh_daylight", "Shopify · 부드러운 자연광", "shopify_collection",
                 "a bright home interior with soft morning daylight, linen textures and green plants"),
    PromptPreset("sh_luxury", "Shopify · 럭셔리 상품", "shopify_product_lifestyle",
                 "the product on dark polished stone with dramatic rim light and elegant reflections"),
    PromptPreset("sh_seasonal", "Shopify · 시즌 프로모션", "shopify_promo_tile",
                 "a festive wooden tabletop with warm string lights and tasteful seasonal decorations softly blurred "
                 "in the background"),
]}


def apply_appearance_hint(prompt: str, preset: PromptPreset | None) -> str:
    """Add the preset's explicit appearance hint to people words; nothing is inferred."""
    if not preset or not preset.appearance_hint or not preset.people:
        return prompt
    hint = preset.appearance_hint.strip()
    if hint.casefold() in prompt.casefold():
        return prompt
    for word in ("young man and woman", "man and woman", "young woman", "young man", "woman", "man", "couple", "person"):
        index = prompt.casefold().find(word)
        if index >= 0:
            return prompt[:index] + f"{hint} " + prompt[index:]
    return f"{prompt}, {hint} people"


# ------------------------------------------------------------------ user overrides
def presets_path(app_root: Path) -> Path:
    return Path(app_root) / "config" / "creator_presets.json"


def _tuple(value: Any) -> tuple | None:
    return tuple(float(v) for v in value) if isinstance(value, (list, tuple)) else None


def load_presets(app_root: Path) -> tuple[dict[str, Purpose], dict[str, PromptPreset], list[str]]:
    """Built-ins merged with ``config/creator_presets.json`` ({"purposes": {...}, "prompt_presets": {...}})."""
    purposes = copy.deepcopy(BUILTIN_PURPOSES)
    prompts = copy.deepcopy(BUILTIN_PROMPT_PRESETS)
    problems: list[str] = []
    path = presets_path(app_root)
    if not path.exists():
        return purposes, prompts, problems
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return purposes, prompts, [f"creator_presets.json ignored: {exc}"]
    for key, value in (data.get("purposes") or {}).items():
        try:
            base = purposes.get(key)
            fields = {**(base.to_dict() if base else {}), **value, "key": key}
            for region in ("text_region", "subject_region", "cta_region"):
                fields[region] = _tuple(fields.get(region))
            fields["alternates"] = {k: tuple(v) for k, v in (fields.get("alternates") or {}).items()}
            fields["exports"] = tuple(fields.get("exports") or ("jpg", "png"))
            purposes[key] = Purpose(**fields)
        except (TypeError, ValueError) as exc:
            problems.append(f"purpose {key!r} ignored: {exc}")
    for key, value in (data.get("prompt_presets") or {}).items():
        try:
            base = prompts.get(key)
            prompts[key] = PromptPreset(**{**(base.to_dict() if base else {}), **value, "key": key})
        except TypeError as exc:
            problems.append(f"prompt preset {key!r} ignored: {exc}")
    return purposes, prompts, problems


def validate_canvas(width: int, height: int) -> tuple[int, int]:
    width, height = int(width), int(height)
    if not (256 <= width <= 4096 and 256 <= height <= 4096):
        raise ValueError("캔버스 크기는 256~4096 px 사이여야 합니다.")
    return width, height

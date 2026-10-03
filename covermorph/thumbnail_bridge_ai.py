"""Thumbnail bridge adapters around the existing SDXL/IP-Adapter engine.

No new generation stack lives here: generation always goes through
``SDXLTextToImageEngine.generate_one``. This module only decides the VRAM
profile, builds textless prompts, retries once on CUDA OOM, and classifies
edit instructions into the capability levels the bridge can honestly perform.
"""
from __future__ import annotations

import re
import sys
import time
from dataclasses import dataclass, field
from threading import Event
from typing import Any, Callable

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from .generation import GENERATION_SIZES, GenerationConfig, SDXLTextToImageEngine
from .thumbnail_bridge_assets import Subject, shift_subjects

GIB = 1024 ** 3

# Ordered from most to least memory hungry; an OOM retry moves exactly one step down.
MEMORY_PROFILES: dict[str, dict[str, Any]] = {
    "standard": {"vae_slicing": False, "vae_tiling": False, "cpu_offload": False, "working_size": GENERATION_SIZES["16:9"]},
    "balanced": {"vae_slicing": True, "vae_tiling": True, "cpu_offload": False, "working_size": GENERATION_SIZES["16:9"]},
    "conservative": {"vae_slicing": True, "vae_tiling": True, "cpu_offload": True, "working_size": (1024, 576)},
}
PROFILE_ORDER = tuple(MEMORY_PROFILES)

NEGATIVE_PROMPT = (
    "text, letters, words, typography, title, caption, subtitle, logo, watermark, signature, "
    "signage, readable sign, banner, numbers, kanji, hangul, frame, border, deformed face, "
    "distorted face, bad anatomy, extra fingers, lowres, blurry, jpeg artifacts"
)

CHANNEL_STYLES = {
    "tokyochill": ("relationship", "cinematic Japanese urban scene, quiet emotional relationship mood, soft city lights, shallow depth of field"),
    "oldpoplounge": ("lounge", "calm mature photographic mood, warm vintage film tones, soft light, simple uncluttered background"),
}
DEFAULT_STYLE = ("story", "cinematic photographic scene, natural light, emotional storytelling mood")

# Small, explicit CJK -> English scene vocabulary: CLIP understands English best.
SCENE_TERMS = {
    "도쿄": "Tokyo", "東京": "Tokyo", "오사카": "Osaka", "大阪": "Osaka", "교토": "Kyoto", "京都": "Kyoto",
    "서울": "Seoul", "신주쿠": "Shinjuku", "新宿": "Shinjuku", "시부야": "Shibuya", "渋谷": "Shibuya",
    "야경": "night cityscape", "夜景": "night cityscape", "밤": "night", "夜": "night",
    "비 오는": "rainy", "비오는": "rainy", "빗속": "in the rain", "雨": "rain", "눈 내리는": "snowfall", "雪": "snow",
    "기차역": "train station", "전철역": "train station", "지하철역": "subway station", "駅": "train station", "전철": "commuter train", "電車": "commuter train",
    "플랫폼": "station platform", "ホーム": "station platform", "버스": "bus", "バス": "bus",
    "카페": "cafe", "カフェ": "cafe", "喫茶": "retro coffee shop", "재즈바": "jazz bar", "バー": "bar",
    "강가": "riverside", "강변": "riverside", "川": "river", "바다": "sea", "海": "sea", "해변": "beach", "ビーチ": "beach",
    "노을": "sunset", "夕焼け": "sunset", "석양": "sunset", "아침": "morning", "朝": "morning",
    "저녁": "evening", "夕方": "evening", "거리": "city street", "街": "city street",
    "골목": "narrow alley", "路地": "narrow alley", "옥상": "rooftop", "屋上": "rooftop",
    "공원": "park", "公園": "park", "창가": "by the window", "窓辺": "by the window",
    "벚꽃": "cherry blossoms", "桜": "cherry blossoms", "봄": "spring", "春": "spring",
    "여름": "summer", "夏": "summer", "가을": "autumn", "秋": "autumn", "겨울": "winter", "冬": "winter",
    "네온": "neon lights", "ネオン": "neon lights", "레트로": "retro", "レトロ": "retro",
    "쇼와": "Showa era", "昭和": "Showa era", "우산": "umbrella", "傘": "umbrella",
    "두 사람": "two people", "二人": "two people", "커플": "couple", "カップル": "couple",
    "남자": "man", "男性": "man", "남성": "man", "여자": "woman", "女性": "woman", "여성": "woman",
    "젊은": "young", "若い": "young", "옆모습": "side profile", "横顔": "side profile",
    "중년": "middle-aged", "中年": "middle-aged", "재즈": "jazz lounge", "ジャズ": "jazz lounge",
}
_CJK = re.compile(r"[ᄀ-ᇿ぀-ヿ㄰-㆏㐀-鿿가-힯ｦ-ﾟ]")


class BridgeActionError(RuntimeError):
    """Structured bridge failure carrying a contract error_code."""

    def __init__(self, code: str, message: str, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.code = code
        self.details = details or {}


def log(message: str) -> None:
    print(f"[thumbnail-bridge] {message}", file=sys.stderr, flush=True)


# ------------------------------------------------------------------ memory profile
def select_memory_profile(vram_bytes: int | None, requested: str = "") -> str:
    """Pick the profile from detected VRAM.

    Measured on an RTX 3060 12 GB at 1344x768/28 steps: "standard" peaked at 14.8 GB reserved (spilling
    into shared system memory) while "balanced" (VAE slicing+tiling) peaked lower, ran faster and
    produced the same image, so 12 GB cards use "balanced".
    """
    if requested in MEMORY_PROFILES:
        return requested
    vram = (vram_bytes or 0) / GIB
    if vram >= 14:
        return "standard"
    if vram >= 8:
        return "balanced"
    return "conservative"


def next_lower_profile(profile: str) -> str | None:
    index = PROFILE_ORDER.index(profile)
    return PROFILE_ORDER[index + 1] if index + 1 < len(PROFILE_ORDER) else None


def is_cuda_oom(exc: BaseException) -> bool:
    seen = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if type(current).__name__ == "OutOfMemoryError" or "out of memory" in str(current).lower():
            return True
        current = current.__cause__ or current.__context__
    return False


# ------------------------------------------------------------------ prompts
def channel_style(channel: str) -> tuple[str, str]:
    key = re.sub(r"[^a-z]", "", channel.casefold())
    for name, style in CHANNEL_STYLES.items():
        if key.startswith(name) or (name == "oldpoplounge" and key.startswith("oldpop")):
            return style
    return DEFAULT_STYLE


def translate_scene_terms(text: str) -> tuple[str, list[str], bool]:
    """Keep ASCII words, translate known CJK scene terms, report whether CJK text remained."""
    found = []
    remaining = text
    for term in sorted(SCENE_TERMS, key=len, reverse=True):
        if term in remaining:
            found.append(SCENE_TERMS[term])
            remaining = remaining.replace(term, " ")
    ascii_part = " ".join(re.sub(r"[^\x20-\x7e]+", " ", text).split()).strip(" ,")
    untranslated = bool(_CJK.search(remaining))
    parts = [ascii_part] if ascii_part else []
    parts.extend(dict.fromkeys(term for term in found if term.casefold() not in ascii_part.casefold()))
    return ", ".join(parts), found, untranslated


def composition_hint(text_side: str, preserve_people: bool) -> str:
    if not preserve_people:
        return "no people, wide open composition"
    if text_side == "left":
        return "people positioned in the right half, calm empty uncluttered left side"
    if text_side == "right":
        return "people positioned in the left half, calm empty uncluttered right side"
    return "people slightly off-center, generous calm negative space"


def fit_prompt(parts: list[str], tokenizer: Any | None, warnings: list[str]) -> str:
    """Join prompt parts in priority order while the CLIP tokenizer limit allows."""
    parts = [part.strip(" ,") for part in parts if part and part.strip(" ,")]
    if tokenizer is None:
        return ", ".join(parts)
    limit = int(getattr(tokenizer, "model_max_length", 77) or 77)

    def length(text: str) -> int:
        return len(tokenizer(text, truncation=False, add_special_tokens=True)["input_ids"])

    chosen: list[str] = []
    for index, part in enumerate(parts):
        candidate = ", ".join([*chosen, part])
        if length(candidate) <= limit:
            chosen.append(part)
            continue
        if index == 0:
            words = part.split()
            while words and length(" ".join(words)) > limit:
                words.pop()
            chosen.append(" ".join(words))
            warnings.append("Prompt was longer than the SDXL text encoder limit and was shortened.")
        else:
            warnings.append(f"Prompt detail dropped to fit the text encoder limit: {part[:60]}")
    return ", ".join(chosen)


@dataclass(slots=True)
class PromptPlan:
    prompt_parts: list[str]
    negative_prompt: str
    scene_type: str
    translated_terms: list[str]
    untranslated_text: bool
    # Negative prompt in priority order; fitted to the text-encoder limit at generation time.
    negative_parts: list[str] = field(default_factory=list)


def build_prompt_plan(user_prompt: str, channel: str, text_side: str, preserve_people: bool,
                      extra_terms: list[str] | None = None, *, person: bool = False, framing: str = "") -> PromptPlan:
    from .person_quality import PERSON_NEGATIVE_PARTS, PERSON_POSITIVE

    scene_type, style = channel_style(channel)
    translated, found, untranslated = translate_scene_terms(user_prompt)
    subject = ", ".join(part for part in [translated, *(extra_terms or [])] if part) or "quiet cinematic city scene"
    if person:
        # People first: framing and photographic face quality outrank channel styling in the 77-token budget.
        parts = [subject, framing, "textless photograph", PERSON_POSITIVE, composition_hint(text_side, True), style]
        negative_parts = list(PERSON_NEGATIVE_PARTS)
    else:
        parts = [subject, "textless photograph", composition_hint(text_side, preserve_people), style,
                 "16:9 cinematic still, high detail, natural skin"]
        negative_parts = [NEGATIVE_PROMPT]
    return PromptPlan([part for part in parts if part], ", ".join(negative_parts), scene_type, found, untranslated,
                      negative_parts)


# ------------------------------------------------------------------ generation with one OOM retry
EngineFactory = Callable[[str], SDXLTextToImageEngine]


def generate_with_memory_fallback(engine_factory: EngineFactory, profile: str, plan: PromptPlan,
                                  config_kwargs: dict[str, Any], seed: int, warnings: list[str],
                                  reference_image: Image.Image | None = None,
                                  cancel_event: Event | None = None,
                                  after_generate: Callable[..., tuple[Image.Image, dict[str, Any]]] | None = None,
                                  ) -> tuple[Image.Image, dict[str, Any]]:
    """Run generate_one; on CUDA OOM free the pipeline and retry exactly once one profile lower.

    ``after_generate(engine, image, config, prompt, negative)`` runs with the pipeline still loaded
    (person QA / face-detail pass) and must handle its own failures by returning the input image.
    """
    cancel_event = cancel_event or Event()
    attempts: list[dict[str, Any]] = []
    current: str | None = profile
    while current is not None:
        engine = engine_factory(current)
        settings = MEMORY_PROFILES[current]
        config = GenerationConfig(output_ratio="16:9", working_size=tuple(settings["working_size"]), **config_kwargs)
        started = time.perf_counter()
        try:
            log(f"loading SDXL (profile={current}, working size={config.size[0]}x{config.size[1]})")
            engine.load(lambda event: log(f"{event}"))
            prompt = fit_prompt(plan.prompt_parts, getattr(engine.pipeline, "tokenizer", None), warnings)
            negative = fit_prompt(plan.negative_parts or [plan.negative_prompt], getattr(engine.pipeline, "tokenizer", None), [])
            log(f"generating seed={seed} steps={config.steps} guidance={config.guidance_scale}")
            image = engine.generate_one(prompt, negative, config, seed, cancel_event,
                                        progress=_step_logger(config.steps), reference_image=reference_image)
            metrics = dict(engine.last_generation_metrics)
            person_info: dict[str, Any] = {}
            if after_generate is not None:
                image, person_info = after_generate(engine, image, config, prompt, negative)
            peaks = [metrics.get("peak_memory_allocated"), (person_info.get("face_detail") or {}).get("peak_memory_allocated")]
            reserved = [metrics.get("peak_memory_reserved"), (person_info.get("face_detail") or {}).get("peak_memory_reserved")]
            attempts.append({"profile": current, "ok": True, "seconds": round(time.perf_counter() - started, 3)})
            return image, {
                **metrics,
                "person_pass": person_info,
                "peak_memory_allocated_overall": max((value for value in peaks if value), default=None),
                "peak_memory_reserved_overall": max((value for value in reserved if value), default=None),
                # Only the real CoverMorph SDXL engine counts as a real AI run (tests inject fakes).
                "real_ai": type(engine) is SDXLTextToImageEngine,
                "engine": type(engine).__name__,
                "memory_profile": current,
                "initial_memory_profile": profile,
                "oom_retry": len(attempts) > 1,
                "attempts": attempts,
                "working_size": list(config.size),
                "prompt": prompt,
                "negative_prompt": negative,
                "seed": seed,
                "steps": config.steps,
                "guidance_scale": config.guidance_scale,
                "reference_mode": config.reference_mode,
                "reference_applied": bool(engine.last_reference_applied),
                "load_and_generate_seconds": round(time.perf_counter() - started, 3),
            }
        except Exception as exc:
            oom = is_cuda_oom(exc)
            attempts.append({"profile": current, "ok": False, "cuda_out_of_memory": oom, "error": str(exc)[:300]})
            engine.unload()
            if not oom:
                raise
            if len(attempts) >= 2:
                raise BridgeActionError("CUDA_OUT_OF_MEMORY", "CUDA ran out of memory after one lower-memory retry.",
                                        {"attempts": attempts}) from exc
            lower = next_lower_profile(current)
            if lower is None:
                raise BridgeActionError("CUDA_OUT_OF_MEMORY", "CUDA ran out of memory in the lowest-memory profile.",
                                        {"attempts": attempts}) from exc
            warnings.append(f"CUDA out of memory in '{current}' profile; retried once with '{lower}'.")
            log(f"CUDA OOM in profile {current}; retrying once with {lower}")
            current = lower
        finally:
            if attempts and attempts[-1].get("ok"):
                engine.unload()
    raise BridgeActionError("CUDA_OUT_OF_MEMORY", "No memory profile available.", {"attempts": attempts})


def _step_logger(total: int) -> Callable[[dict[str, Any]], None]:
    def progress(event: dict[str, Any]) -> None:
        if event.get("phase") == "inference":
            step = int(event.get("step", 0))
            if step in (1, total) or step % 5 == 0:
                log(f"inference step {step}/{total}")
        else:
            log(str(event))
    return progress


# ------------------------------------------------------------------ edit planning
EDIT_LEVELS = ("recompose", "regenerate", "reference_regenerate")

_LEFT = r"(왼쪽|좌측|왼편|左|left)"
_RIGHT = r"(오른쪽|우측|오른편|右|right)"
_MOVE = r"(이동|옮기|옮겨|움직|밀어|move|shift|移動|寄せ|ずら)"
_TEXT_SPACE = r"(문구|텍스트|글자|자막|여백|공간|text|copy ?space|negative space|文字|余白|スペース)"
_WIDEN = r"(넓|확보|늘려|키워|비워|wider|widen|more|bigger|clear|広|空け)"
_SIMPLIFY = r"(단순|심플|깔끔|정리|simple|simplify|clean|calm|シンプル|すっきり)"
_DARK = r"(어둡|어둡게|darker|darken|dim|暗)"
_BRIGHT = r"(밝게|밝히|brighter|brighten|lighter|明る)"
_SMALL = r"(조금|살짝|약간|slightly|a little|a bit|少し|ちょっと)"
_LARGE = r"(많이|크게|훨씬|a lot|much|significantly|大きく|かなり)"
_BACKGROUND_CHANGE = (r"(배경.{0,12}(바꿔|바꾸|변경|교체|으로|로)|change (the )?background|background (to|into)|"
                      r"replace (the )?background|背景.{0,12}(変え|変更|に))")
_SCENE_CHANGE = r"(바꿔|바꾸|변경|교체|change|turn|変え|変更|にして)"
_REGENERATE = r"(다시 생성|새로 생성|재생성|새로 만들|regenerate|re-generate|generate again|new background|作り直|再生成)"
_PRESERVE = r"(유지|그대로|보존|살려|keep|preserve|retain|unchanged|保持|そのまま|維持|残)"
_PEOPLE = r"(인물|사람|두 사람|얼굴|포즈|자세|people|person|faces?|couple|pose|人物|二人|顔|ポーズ)"
_ADD_REMOVE = (r"(추가|넣어|넣고|넣어줘|지워|지우|삭제|제거|없애|\badd\b|insert|remove|erase|delete|"
               r"get rid|追加|入れ|消し|消して|削除|取り除)")
_PERSON_ATTRIBUTE = r"(표정|옷|의상|머리|헤어|나이|expression|clothes|outfit|hair|age|smile|表情|服|衣装|髪|年齢)"
_CHANGE = r"(바꿔|바꾸|변경|change|make|変え|変更)"
_TEXT_INSERT = r"((글자|텍스트|문구|제목|자막|로고).{0,6}(넣|추가|써)|add (the )?(text|title|logo)|文字を(入れ|追加))"


@dataclass(slots=True)
class EditPlan:
    level: str = ""
    shift: float = 0.0
    simplify_side: str = ""
    simplify_fraction: float = 0.0
    tone: float = 1.0
    regenerate: bool = False
    preserve_people: bool = False
    scene_terms: list[str] = field(default_factory=list)
    recognized: list[str] = field(default_factory=list)
    unsupported: list[str] = field(default_factory=list)

    @property
    def has_recompose_ops(self) -> bool:
        return bool(self.shift or self.simplify_side or abs(self.tone - 1.0) > 1e-6)

    def to_dict(self) -> dict[str, Any]:
        return {"level": self.level, "shift": self.shift, "simplify_side": self.simplify_side,
                "simplify_fraction": self.simplify_fraction, "tone": self.tone, "regenerate": self.regenerate,
                "preserve_people": self.preserve_people, "scene_terms": list(self.scene_terms),
                "recognized": list(self.recognized), "unsupported": list(self.unsupported)}


def _clauses(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"[,.;!?、。，]|그리고|하고|하되|되 |but |and |また|ながら", text) if part.strip()]


def _side_in(text: str) -> str:
    left, right = re.search(_LEFT, text, re.I), re.search(_RIGHT, text, re.I)
    if left and right:
        return "left" if left.start() < right.start() else "right"
    return "left" if left else ("right" if right else "")


def parse_edit_instruction(instruction: str, forced_level: str = "") -> EditPlan:
    """Classify a free-form KO/JA/EN instruction into supported operations, or mark it unsupported."""
    plan = EditPlan()
    text = " ".join(instruction.split())
    lowered = text.casefold()
    amount = 0.18 if re.search(_LARGE, lowered, re.I) else (0.08 if re.search(_SMALL, lowered, re.I) else 0.12)

    if re.search(_TEXT_INSERT, lowered, re.I):
        plan.unsupported.append("adding text/logo (typography belongs to the thumbnail editor)")
    for clause in _clauses(lowered):
        if re.search(_ADD_REMOVE, clause, re.I) and not re.search(_TEXT_INSERT, clause, re.I):
            plan.unsupported.append(f"adding/removing objects: {clause}")
        elif re.search(_PERSON_ATTRIBUTE, clause, re.I) and re.search(_CHANGE, clause, re.I) and not re.search(_PRESERVE, clause, re.I):
            plan.unsupported.append(f"changing a person's appearance: {clause}")

    for clause in _clauses(lowered):
        side = _side_in(clause)
        moves = re.search(_MOVE, clause, re.I)
        text_space = re.search(_TEXT_SPACE, clause, re.I) and re.search(_WIDEN + "|" + _SIMPLIFY, clause, re.I)
        simplify = re.search(_SIMPLIFY, clause, re.I)
        if moves and side and not text_space:
            plan.shift = amount if side == "right" else -amount
            plan.recognized.append(f"move subject {side}")
        if (text_space or simplify) and (side or plan.shift):
            target = side or ("left" if plan.shift > 0 else "right")
            percent = re.search(r"(\d{1,2})\s*%", clause)
            plan.simplify_side = target
            plan.simplify_fraction = min(0.55, max(0.2, int(percent.group(1)) / 100)) if percent else 0.4
            plan.recognized.append(f"text space {target} {plan.simplify_fraction:.0%}")
    if re.search(_DARK, lowered, re.I):
        plan.tone = 1.0 - (0.28 if re.search(_LARGE, lowered, re.I) else 0.16)
        plan.recognized.append("darken background")
    elif re.search(_BRIGHT, lowered, re.I):
        plan.tone = 1.0 + (0.25 if re.search(_LARGE, lowered, re.I) else 0.14)
        plan.recognized.append("brighten background")
    translated, found, _ = translate_scene_terms(instruction)
    scene_change = bool(found) and re.search(_SCENE_CHANGE, lowered, re.I) and not plan.unsupported
    if re.search(_BACKGROUND_CHANGE, lowered, re.I) or re.search(_REGENERATE, lowered, re.I) or scene_change:
        plan.regenerate = True
        plan.recognized.append("new background")
    for clause in _clauses(lowered):
        if re.search(_PEOPLE, clause, re.I) and re.search(_PRESERVE, clause, re.I):
            plan.preserve_people = True
    english_scene = re.search(r"background (?:to|into|with) (?:an? |the )?([a-z0-9 '-]+)", lowered)
    plan.scene_terms = found or ([english_scene.group(1).strip(" ,")] if english_scene else [])

    if forced_level:
        plan.level = forced_level
        if forced_level != "recompose":
            plan.regenerate = True
            plan.preserve_people = plan.preserve_people or forced_level == "reference_regenerate"
    elif plan.regenerate:
        plan.level = "reference_regenerate" if plan.preserve_people else "regenerate"
    elif plan.has_recompose_ops:
        plan.level = "recompose"
    return plan


# ------------------------------------------------------------------ local recompose
def _protect_mask(size: tuple[int, int], subjects: list[Subject], feather: int = 24) -> Image.Image:
    """Fully opaque over every subject box; the soft edge lies outside the box, never inside it."""
    from PIL import ImageDraw

    mask = Image.new("L", size, 0)
    width, height = size
    draw = ImageDraw.Draw(mask)
    for subject in subjects:
        x, y, w, h = subject.box
        draw.rectangle((x * width - feather, y * height - feather, (x + w) * width + feather, (y + h) * height + feather), fill=255)
    hard = Image.new("L", size, 0)
    hard_draw = ImageDraw.Draw(hard)
    for subject in subjects:
        x, y, w, h = subject.box
        hard_draw.rectangle((x * width, y * height, (x + w) * width, (y + h) * height), fill=255)
    from PIL import ImageChops

    return ImageChops.lighter(mask.filter(ImageFilter.GaussianBlur(feather / 2)), hard)


def shift_canvas(image: Image.Image, dx: float) -> Image.Image:
    """Translate content horizontally; fill the vacated strip with a soft mirrored extension."""
    width, height = image.size
    pixels = int(round(abs(dx) * width))
    if pixels <= 0:
        return image.copy()
    result = Image.new("RGB", image.size)
    if dx > 0:
        result.paste(image.crop((0, 0, width - pixels, height)), (pixels, 0))
        source = image.crop((0, 0, min(width, pixels * 2), height)).transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        strip = source.crop((source.width - pixels, 0, source.width, height))
        strip_x = 0
    else:
        result.paste(image.crop((pixels, 0, width, height)), (0, 0))
        source = image.crop((max(0, width - pixels * 2), 0, width, height)).transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        strip = source.crop((0, 0, pixels, height))
        strip_x = width - pixels
    strip = ImageEnhance.Contrast(strip.filter(ImageFilter.GaussianBlur(max(8, height // 40)))).enhance(0.85)
    result.paste(strip, (strip_x, 0))
    seam = max(12, pixels // 3)
    seam_box = (max(0, pixels - seam), 0, min(width, pixels + seam), height) if dx > 0 else \
        (max(0, width - pixels - seam), 0, min(width, width - pixels + seam), height)
    blurred = result.crop(seam_box).filter(ImageFilter.GaussianBlur(seam // 3))
    gradient = np.abs(np.linspace(-1.0, 1.0, seam_box[2] - seam_box[0]))
    alpha = Image.fromarray(((1.0 - gradient) * 255).astype(np.uint8)[None, :].repeat(height, axis=0), "L")
    result.paste(blurred, seam_box[:2], alpha)
    return result


def simplify_region(image: Image.Image, side: str, fraction: float, subjects: list[Subject]) -> Image.Image:
    """Lower detail/contrast in a side band for typography, never inside protected subjects."""
    width, height = image.size
    edge = int(width * fraction)
    feather = max(16, int(width * 0.06))
    ramp = np.clip((edge - np.arange(width)) / feather + 0.5, 0.0, 1.0)
    band = ramp if side == "left" else ramp[::-1]
    band_mask = (band[None, :].repeat(height, axis=0) * 255).astype(np.uint8)
    protect = np.asarray(_protect_mask(image.size, subjects), dtype=np.float32) / 255.0
    mask = Image.fromarray((band_mask * (1.0 - protect)).astype(np.uint8), "L")
    calm = image.filter(ImageFilter.GaussianBlur(max(10, height // 36)))
    calm = ImageEnhance.Contrast(calm).enhance(0.72)
    calm = ImageEnhance.Brightness(calm).enhance(0.9)
    return Image.composite(calm, image, mask)


def adjust_background_tone(image: Image.Image, factor: float, subjects: list[Subject]) -> Image.Image:
    adjusted = ImageEnhance.Brightness(image).enhance(factor)
    protect = _protect_mask(image.size, subjects)
    return Image.composite(image, adjusted, protect)


def apply_recompose(image: Image.Image, plan: EditPlan, subjects: list[Subject]) -> tuple[Image.Image, list[Subject], list[str]]:
    notes: list[str] = []
    result = image.convert("RGB")
    shift = plan.shift
    if plan.simplify_side and not shift:
        # Make room automatically when people intrude into the requested text band.
        band_edge = plan.simplify_fraction
        if plan.simplify_side == "left":
            intrusion = max((band_edge - subject.box[0] for subject in subjects if subject.box[0] < band_edge), default=0.0)
            shift = min(0.22, intrusion + 0.02) if intrusion > 0 else 0.0
        else:
            intrusion = max((subject.box[0] + subject.box[2] - (1 - band_edge) for subject in subjects
                             if subject.box[0] + subject.box[2] > 1 - band_edge), default=0.0)
            shift = -min(0.22, intrusion + 0.02) if intrusion > 0 else 0.0
        if shift:
            notes.append(f"Subjects moved {abs(shift):.0%} to clear the requested text space")
    if shift:
        result = shift_canvas(result, shift)
        subjects = shift_subjects(subjects, shift)
    if plan.simplify_side:
        result = simplify_region(result, plan.simplify_side, plan.simplify_fraction, subjects)
    if abs(plan.tone - 1.0) > 1e-6:
        result = adjust_background_tone(result, plan.tone, subjects)
    return result, subjects, notes

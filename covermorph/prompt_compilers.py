"""Engine-specific prompt compilers for Quality Engine V2.

The user's prompt is never overwritten: every compile returns the original text, the channel
envelope that was added and the engine-specific rewrite, so manifests can store all of them.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .person_quality import COMPOSITION_PROFILES, PERSON_NEGATIVE_PARTS, PERSON_POSITIVE
from .quality_engines import Reference
from .thumbnail_bridge_ai import NEGATIVE_PROMPT, build_prompt_plan, channel_style, translate_scene_terms

PURPOSES: dict[str, dict[str, Any]] = {
    "youtube_thumbnail_background": {"size": (1280, 720), "envelope": "16:9 cinematic YouTube thumbnail photograph"},
    # Neutral wording: "e-commerce/banner/promotional" made Z-Image invent a product (a camera) in product-less
    # store scenes (2026-10-04). Product jobs say what the product is through the prompt or a PRODUCT reference.
    "shopify_hero_banner": {"size": (1792, 768), "envelope": "wide editorial lifestyle photograph, premium look"},
    "shopify_collection_banner": {"size": (1600, 640), "envelope": "wide lifestyle photograph, clean premium styling"},
    "shopify_product_lifestyle": {"size": (1280, 1280), "envelope": "lifestyle product photograph, natural styling"},
    "shopify_promo_tile": {"size": (1024, 1024), "envelope": "square photograph, bold simple composition"},
    "shopify_mobile_banner": {"size": (768, 1024), "envelope": "portrait photograph, simple composition"},
}
# Engines need multiples of 16; purposes map to these generation sizes and are resized after.
CANVAS_PRESETS = {name: spec["size"] for name, spec in PURPOSES.items()}

TEXTLESS_PROSE = ("The image contains no text, letters, captions, logos, watermarks or readable signage.")
FLUX_TEXTLESS = "Every surface is free of lettering and logos; any signs are blurred and unreadable."
# Measured: without the viewing-angle/silhouette sentence FLUX.2-klein gave a single-handle mug two handles
# on 3/3 seeds; with it 2/4 were exact. Deviations stay seed-dependent, so product jobs also get a warning.
SHOPIFY_PRODUCT_RULE = ("Show the product from the same viewing angle as its reference, with an identical silhouette, "
                        "the same number of parts and the same colors, materials and logo; "
                        "do not add new claims, labels or packaging text.")
PRODUCT_WARNING = ("Product reference: check shape/part count against the reference (FLUX.2-klein can duplicate "
                   "parts such as handles on some seeds); keep 2+ candidates for product shots.")


@dataclass(slots=True)
class CompiledPrompt:
    engine: str
    original_prompt: str
    channel: str
    purpose: str
    envelope: str
    positive: str
    negative: str = ""
    reference_roles: list[dict[str, Any]] = field(default_factory=list)
    translated_terms: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    prompt_parts: list[str] = field(default_factory=list)      # SDXL: fitted to CLIP at generation time
    negative_parts: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


NO_PEOPLE = "The scene is empty of people; no person or figure appears anywhere."
CAMERA_PROSE = "Shot on a full-frame camera with a 50mm lens at f/2, natural light falloff, realistic color grading."
FLUX_CAMERA_PROSE = "Photographed with a full-frame camera and a 50mm lens, shallow depth of field, true-to-life color."
NO_PRODUCT = ("No product, camera, gadget, device, bottle or packaging is featured; the setting itself is the "
              "subject.")


def _no_product(purpose: str, user_prompt: str, roles: list[dict[str, Any]] | None = None) -> str:
    """Store scenes without a product reference or product wording must not get an invented hero product."""
    if not purpose.startswith("shopify"):
        return ""
    if any(r["role"] in ("PRODUCT", "EDIT") for r in roles or []) or "product" in user_prompt.casefold():
        return ""
    return NO_PRODUCT


def _negative_space(text_side: str, person: bool = True) -> str:
    side = (text_side or "left").lower()
    if side in ("left", "right"):
        other = "right" if side == "left" else "left"
        focus = "main subject" if person else "visual focal point"
        return (f"The {focus} sits in the {other} part of the frame, and the {side} third is calm, "
                f"softly out-of-focus negative space suitable for a large title.")
    if side in ("top", "bottom"):
        return f"The {side} part of the frame is calm, uncluttered negative space suitable for a large title."
    return "Leave a calm, uncluttered area of the frame for a large title."


_NO_PEOPLE_WORDS = ("no people", "no person", "nobody", "no one", "without people", "empty of people", "unpeopled",
                    "empty street", "deserted",
                    "사람 없", "사람없", "인물 없", "無人", "人がいない", "人のいない", "誰もいない")


def _scene(user_prompt: str) -> tuple[str, list[str]]:
    """English scene text with no CJK script left in it.

    Z-Image paints CJK prompt words into the image as lettering (measured), so callers translate first
    (prompt_translate.translate_prompt); anything still in CJK here falls back to the scene vocabulary.
    """
    from .prompt_translate import needs_translation

    translated, found, _ = translate_scene_terms(user_prompt)
    if needs_translation(user_prompt):
        from .prompt_translate import _CJK
        return " ".join(_CJK.sub(" ", translated).split()), found
    return user_prompt.strip() or translated, found


def _explicitly_empty(user_prompt: str) -> bool:
    text = user_prompt.casefold()
    return any(word in text for word in _NO_PEOPLE_WORDS)


def _framing(composition: str) -> str:
    profile = COMPOSITION_PROFILES.get(composition or "")
    return profile["framing"] if profile else ""


def _sentence(text: str) -> str:
    text = text.strip()
    return "" if not text else text[0].upper() + text[1:] + ("" if text.endswith(".") else ".")


def compile_zimage(user_prompt: str, channel: str = "", purpose: str = "youtube_thumbnail_background", *,
                   text_side: str = "left", composition: str = "", person: bool = False) -> CompiledPrompt:
    """Detailed natural-language prose, subject first, camera/light, explicit negative space."""
    subject, found = _scene(user_prompt)
    _, style = channel_style(channel)
    envelope = PURPOSES[purpose]["envelope"]
    parts = [_sentence(f"A {envelope} of {subject}" if not subject.lower().startswith(("a ", "an ", "the ")) else
                       f"{subject}, as a {envelope}"),
             _sentence(_framing(composition)),
             _sentence(f"Mood and look: {style}"),
             # Without people, the camera/lens words made Z-Image draw a camera into store scenes (2026-10-04).
             CAMERA_PROSE if person else "Shallow depth of field, natural light falloff, realistic color grading.",
             "Faces have natural skin texture, realistic eyes and hair, and natural proportions." if person else "",
             NO_PEOPLE if not person and _explicitly_empty(user_prompt) else "",
             _no_product(purpose, user_prompt),
             _negative_space(text_side, person) if purpose.startswith("youtube") or purpose.endswith("banner") else "",
             TEXTLESS_PROSE]
    positive = " ".join(part for part in parts if part)
    return CompiledPrompt("zimage_turbo", user_prompt, channel, purpose, f"{envelope}; {style}", positive,
                          negative="", translated_terms=found)


ROLE_INSTRUCTIONS = {
    "PERSON": "keep the same person from image {i}: same face, hairstyle, clothing and pose unless this prompt "
              "changes them",
    "PRODUCT": "use the exact product from image {i}, preserving its shape, material, color, packaging and visible "
               "branding",
    "STYLE": "take only the lighting, color palette and mood from image {i}, not its subjects or layout",
    "COMPOSITION": "follow the subject placement and negative space of image {i}, not the identity of anything in it",
    "BACKGROUND": "use the environment of image {i} as inspiration for the setting, without copying its foreground "
                  "subjects",
    "EDIT": "image {i} is the photo being edited: keep its people exactly as they are, with the same faces, "
            "hair, clothes and poses",
}


def compile_flux2(user_prompt: str, channel: str = "", purpose: str = "youtube_thumbnail_background", *,
                  text_side: str = "left", composition: str = "", person: bool = False,
                  references: list[Reference] | None = None, edit_instruction: str = "") -> CompiledPrompt:
    """Positive-only narrative: Subject + Action + Medium + Context + Lighting + Camera, refs by index."""
    subject, found = _scene(user_prompt)
    _, style = channel_style(channel)
    envelope = PURPOSES[purpose]["envelope"]
    roles = [{"index": i, "role": ref.role, "path": str(ref.path)} for i, ref in enumerate(references or [], 1)]
    reference_text = ""
    if roles:
        reference_text = _sentence("; ".join(ROLE_INSTRUCTIONS[r["role"]].format(i=r["index"]) for r in roles))
    parts = [_sentence(edit_instruction) if edit_instruction else "",
             _sentence(f"A {envelope}: {subject}") if subject else _sentence(f"A {envelope}"),
             reference_text,
             _sentence(_framing(composition)),
             _sentence(f"The atmosphere is {style}"),
             FLUX_CAMERA_PROSE if person else "Shallow depth of field and true-to-life color.",
             "Skin shows natural texture with realistic eyes and hair." if person else "",
             SHOPIFY_PRODUCT_RULE if any(r["role"] == "PRODUCT" for r in roles) else "",
             NO_PEOPLE if not person and not roles and _explicitly_empty(user_prompt) else "",
             _no_product(purpose, user_prompt, roles),
             _negative_space(text_side, person) if purpose.startswith("youtube") or purpose.endswith("banner") else "",
             FLUX_TEXTLESS]
    positive = " ".join(part for part in parts if part)
    warnings = [PRODUCT_WARNING] if any(r["role"] == "PRODUCT" for r in roles) else []
    return CompiledPrompt("flux2_klein_4b", user_prompt, channel, purpose, f"{envelope}; {style}", positive,
                          reference_roles=roles, translated_terms=found, warnings=warnings)


def compile_realvis(user_prompt: str, channel: str = "", purpose: str = "youtube_thumbnail_background", *,
                    text_side: str = "left", composition: str = "", person: bool = False) -> CompiledPrompt:
    """SDXL keyword prompt + negative: the tested person-quality path, unchanged."""
    from .thumbnail_bridge_ai import fit_prompt
    plan = build_prompt_plan(user_prompt, channel, text_side, person, person=person, framing=_framing(composition))
    warnings: list[str] = []
    positive = fit_prompt(plan.prompt_parts, None, warnings)
    negative = ", ".join(PERSON_NEGATIVE_PARTS) if person else NEGATIVE_PROMPT
    _, style = channel_style(channel)
    return CompiledPrompt("realvisxl_v5", user_prompt, channel, purpose, f"{PERSON_POSITIVE if person else ''}; {style}",
                          positive, negative, translated_terms=plan.translated_terms, warnings=warnings,
                          prompt_parts=list(plan.prompt_parts),
                          negative_parts=list(PERSON_NEGATIVE_PARTS) if person else [NEGATIVE_PROMPT])


COMPILERS = {"zimage_turbo": compile_zimage, "flux2_klein_4b": compile_flux2, "realvisxl_v5": compile_realvis}


def compile_prompt(engine: str, user_prompt: str, channel: str = "", purpose: str = "youtube_thumbnail_background",
                   **kwargs: Any) -> CompiledPrompt:
    if purpose not in PURPOSES:
        raise ValueError(f"Unknown purpose {purpose!r}")
    if engine != "flux2_klein_4b":
        kwargs.pop("references", None)
        kwargs.pop("edit_instruction", None)
    return COMPILERS[engine](user_prompt, channel, purpose, **kwargs)

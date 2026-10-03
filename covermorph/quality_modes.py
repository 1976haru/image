"""PREVIEW / BALANCED / BEST routing over the V2 engines, run strictly one engine at a time.

Defaults come from the 2026-10-04 RTX 3060 comparison (validation_results/quality_v2, docs/QUALITY_ENGINE_V2.md):
  * text-to-image: Z-Image-Turbo Q6_K — best realism and prompt adherence, 7.3 GiB peak, ~35 s at 1280x720
  * reference/edit: FLUX.2-klein-4B Q8_0 — kept the referenced person/product while following the new scene;
    IP-Adapter did neither. ~27 s, 7.3 GiB
  * fallback: RealVisXL V5 (Diffusers) — used when the sd.cpp engines are missing or fail; 11.9 GiB peak
Scoring is technical only and never hides candidates: every candidate is returned, best-scored first.
"""
from __future__ import annotations

import time
from pathlib import Path
from threading import Event
from typing import Any, Callable

from .prompt_compilers import PURPOSES, compile_prompt
from .quality_engines import (
    EngineCancelled,
    EngineError,
    EngineJob,
    MEMORY_POLICIES,
    Reference,
    check_resources,
    make_backend,
    resource_snapshot,
)

DEFAULT_T2I_ENGINE = "zimage_turbo"
DEFAULT_REFERENCE_ENGINE = "flux2_klein_4b"
FALLBACK_ENGINE = "realvisxl_v5"

# (engine, candidates) in run order. Each engine is unloaded before the next one starts.
QUALITY_MODES_V2: dict[str, dict[str, list[tuple[str, int]]]] = {
    "preview": {"t2i": [("flux2_klein_4b", 1)], "reference": [("flux2_klein_4b", 1)]},
    "balanced": {"t2i": [("zimage_turbo", 2)], "reference": [("flux2_klein_4b", 2)]},
    "best": {"t2i": [("zimage_turbo", 2), ("flux2_klein_4b", 2)],
             "reference": [("flux2_klein_4b", 3), ("realvisxl_v5", 1)]},
}
# Memory profile chosen per mode unless the caller asks for one.
MODE_MEMORY = {"preview": "interactive_low_memory", "balanced": "interactive_low_memory", "best": "night_best"}
# In low-memory mode, generations wider than this are made at a reduced size and upscaled (Lanczos):
# 1792x768 peaked at 10.4 GiB, 1280x720 at 7.3 GiB on the RTX 3060.
LOW_MEMORY_MAX_PIXELS = 1280 * 768


def plan_engines(mode: str, *, has_references: bool, edit: bool = False,
                 available: Callable[[str], bool] | None = None) -> list[tuple[str, int]]:
    mode = mode.casefold()
    if mode not in QUALITY_MODES_V2:
        raise ValueError(f"quality mode must be one of {', '.join(QUALITY_MODES_V2)}")
    plan = list(QUALITY_MODES_V2[mode]["reference" if has_references or edit else "t2i"])
    if edit:  # only FLUX.2 edits; RealVis cannot honour an edit instruction
        plan = [(engine, count) for engine, count in plan if engine == "flux2_klein_4b"]
    if available is not None:
        kept = [(engine, count) for engine, count in plan if available(engine)]
        if not kept and not edit:
            kept = [(FALLBACK_ENGINE, plan[0][1] if plan else 1)] if available(FALLBACK_ENGINE) else []
        plan = kept
    return plan


def _generation_size(purpose: str, memory_profile: str) -> tuple[tuple[int, int], tuple[int, int]]:
    final = PURPOSES[purpose]["size"]
    if memory_profile != "interactive_low_memory" or final[0] * final[1] <= LOW_MEMORY_MAX_PIXELS:
        return final, final
    scale = (LOW_MEMORY_MAX_PIXELS / (final[0] * final[1])) ** 0.5
    return final, (int(final[0] * scale) // 16 * 16, int(final[1] * scale) // 16 * 16)


def score_candidate(qa: dict[str, Any], people: int) -> float:
    """Technical only: QA pass, face size/sharpness score, exposure. Never appearance or identity."""
    score = 1.0 if qa.get("passed") else 0.0
    score += float(qa.get("score") or 0.0) if people else 0.5
    score -= 0.25 * len(qa.get("warnings") or [])
    return round(score, 3)


def run_quality_job(payload: dict[str, Any], cancel: Event | None = None,
                    progress: Callable[[dict[str, Any]], None] | None = None,
                    backend_factory: Callable[..., Any] = make_backend) -> dict[str, Any]:
    """Generate every planned candidate sequentially; returns candidates (PIL images) + manifest records.

    payload: models_dir, prompt, channel, purpose, mode, seed, people, composition, text_side,
             references [{path, role}], edit_image, edit_instruction, memory_profile.
    """
    from .person_quality import COMPOSITION_PROFILES, YUNET_FILENAME, detect_faces_yunet, quality_check
    from PIL import Image

    cancel = cancel or Event()
    models_dir = Path(payload["models_dir"])
    mode = str(payload.get("mode") or "balanced").casefold()
    purpose = str(payload.get("purpose") or "youtube_thumbnail_background")
    memory_profile = str(payload.get("memory_profile") or MODE_MEMORY[mode])
    if memory_profile not in MEMORY_POLICIES:
        raise ValueError(f"memory_profile must be one of {', '.join(MEMORY_POLICIES)}")
    references = [Reference(Path(r["path"]), r["role"]) for r in payload.get("references") or []]
    edit_image = Path(payload["edit_image"]) if payload.get("edit_image") else None
    people = int(payload.get("people") or 0)
    composition = str(payload.get("composition") or "")
    seed = int(payload.get("seed") or 1)

    def available(engine: str) -> bool:
        return bool(backend_factory(engine, models_dir).status()["ready"])

    plan = plan_engines(mode, has_references=bool(references), edit=edit_image is not None, available=available)
    if payload.get("only_engine"):
        plan = [(str(payload["only_engine"]), plan[0][1] if plan else 1)]
    if payload.get("max_candidates_per_engine"):
        plan = [(engine, min(count, int(payload["max_candidates_per_engine"]))) for engine, count in plan]
    if not plan:
        raise EngineError("No image engine is installed for this request (edit needs FLUX.2-klein).")
    final_size, work_size = _generation_size(purpose, memory_profile)
    yunet = models_dir / "face_detection" / YUNET_FILENAME
    candidates: list[dict[str, Any]] = []
    warnings: list[str] = []
    for engine_name, count in plan:
        ok, reasons = check_resources(memory_profile, resource_snapshot())
        if not ok:
            warnings.append(f"{engine_name} skipped: " + "; ".join(reasons))
            continue
        backend = backend_factory(engine_name, models_dir)
        compiled = compile_prompt(engine_name, str(payload.get("prompt") or ""), str(payload.get("channel") or ""),
                                  purpose, text_side=str(payload.get("text_side") or "left"),
                                  composition=composition, person=people > 0, references=references,
                                  edit_instruction=str(payload.get("edit_instruction") or ""))
        warnings += [w for w in compiled.warnings if w not in warnings]
        try:
            for index in range(count):
                if cancel.is_set():
                    raise EngineCancelled("Cancelled between candidates.")
                job = EngineJob(prompt=compiled.positive, negative_prompt=compiled.negative, width=work_size[0],
                                height=work_size[1], seed=seed + index, references=list(references),
                                edit_image=edit_image, prompt_parts=compiled.prompt_parts,
                                negative_parts=compiled.negative_parts)
                started = time.perf_counter()
                try:
                    result = backend.edit(job, cancel, progress) if edit_image else backend.generate(job, cancel, progress)
                except EngineCancelled:
                    raise
                except Exception as exc:  # engine failure: record it, keep the remaining engines
                    warnings.append(f"{engine_name} failed: {exc}".splitlines()[0])
                    break
                image = result.image
                if image.size != final_size:
                    image = image.resize(final_size, Image.Resampling.LANCZOS)
                profile = COMPOSITION_PROFILES.get(composition, {})
                faces = detect_faces_yunet(image, yunet, 0.5) if yunet.exists() else []
                qa = quality_check(image, faces, people, profile.get("min_face_height", 0.0),
                                   profile.get("max_face_height", 1.0))
                info = backend.info
                candidates.append({
                    "image": image, "score": score_candidate(qa, people), "qa": qa,
                    "manifest": {"backend": info.name, "model": info.model, "repository": info.repository,
                                 "quantization": info.quantization, "model_license": info.license_id,
                                 "commercial_use_flag": info.commercial_ok, "original_prompt": compiled.original_prompt,
                                 "compiled_prompt": compiled.positive, "negative_prompt": compiled.negative,
                                 "reference_roles": compiled.reference_roles, "quality_mode": mode,
                                 "memory_profile": memory_profile, "purpose": purpose, "seed": job.seed,
                                 "generation_size": list(work_size), "final_size": list(final_size),
                                 "timing": {"engine_seconds": result.seconds,
                                            "wall_seconds": round(time.perf_counter() - started, 2)},
                                 "peak_vram_mib": result.peak_vram_mib, "vram_delta_mib": result.vram_delta_mib,
                                 "system_commit_before_mib": result.commit_before_mib,
                                 "system_commit_after_mib": result.commit_after_mib}})
        finally:
            backend.unload()  # one resident model at most, even across a BEST ensemble
    candidates.sort(key=lambda c: c["score"], reverse=True)
    return {"candidates": candidates, "warnings": warnings, "plan": plan, "mode": mode,
            "memory_profile": memory_profile}

"""Headless runtime for the CoverMorph <-> thumbnail editor bridge.

Invocation forms (all produce exactly one JSON response on stdout):

* ``--thumbnail-bridge-json``  contract v1 request on stdin
* ``--image-bridge``           youtubesum ``IMAGE_BRIDGE_MODE=json-stdin`` request on stdin
* ``--action ... --project-dir ...``  youtubesum default ``IMAGE_BRIDGE_MODE=cli`` argv form
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from .thumbnail_bridge import (
    ThumbnailBridgeError,
    ThumbnailBridgeRequest,
    ThumbnailBridgeResponse,
    read_request_json,
    standard_output_paths,
    write_response_json,
)

BRIDGE_FLAGS = ("--thumbnail-bridge-json", "--image-bridge")
FINAL_SIZE = (1280, 720)
DEFAULT_STEPS = 28
DEFAULT_GUIDANCE = 7.0
# Strength selected in the RTX 3060 3-B1 quality revalidation for 16:9 person references.
DEFAULT_REFERENCE_STRENGTH = 0.35
EDIT_LEVEL_DESCRIPTIONS = {
    "recompose": "Local geometry edit: move subjects left/right, clear/simplify a text side, darken/brighten "
                 "the background with people protected. No AI model required.",
    "regenerate": "New SDXL background from the previous prompt plus the instruction's scene terms. "
                  "People are not preserved.",
    "reference_regenerate": "New SDXL background with IP-Adapter person reference taken from the current canvas. "
                            "Preserves people approximately (similarity), not pixel-exactly.",
}


def bridge_requested(argv: list[str]) -> bool:
    return any(flag in argv for flag in BRIDGE_FLAGS) or "--action" in argv


def _app_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[1]


def _log(message: str) -> None:
    print(f"[thumbnail-bridge] {message}", file=sys.stderr, flush=True)


# ------------------------------------------------------------------ environment / models
def resolve_models_dir(request: ThumbnailBridgeRequest) -> Path:
    configured = str(request.options.get("models_dir") or os.environ.get("COVERMORPH_MODELS_DIR") or "")
    return Path(configured).expanduser() if configured else _app_root() / "models"


def resolve_model_id(request: ThumbnailBridgeRequest, models_dir: Path) -> str:
    from .generation import DEFAULT_SDXL_MODEL

    explicit = str(request.options.get("model_id") or "")
    if explicit:
        return explicit
    local = models_dir / "sdxl_base_1.0"
    return str(local) if local.is_dir() else DEFAULT_SDXL_MODEL


def _environment(request: ThumbnailBridgeRequest) -> tuple[dict[str, Any], Path, str]:
    from .generation import detect_generation_environment

    models_dir = resolve_models_dir(request)
    model_id = resolve_model_id(request, models_dir)
    env = detect_generation_environment(_app_root(), model_id=model_id, models_dir=models_dir)
    return env, models_dir, model_id


def _project_writable(project_dir: Path) -> bool:
    try:
        project_dir.mkdir(parents=True, exist_ok=True)
        probe = project_dir / ".thumbnail_bridge_write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
        return True
    except OSError:
        return False


def _capabilities(env: dict[str, Any], writable: bool) -> dict[str, Any]:
    generate_ready = env.get("status") == "ready"
    reference_ready = env.get("reference_status") == "ready"
    levels = ["recompose"] + (["regenerate"] if generate_ready else []) + (["reference_regenerate"] if reference_ready else [])
    return {
        "status": True,
        "generate": generate_ready and writable,
        "edit": levels,
        "edit_levels": {level: {"available": level in levels, "description": EDIT_LEVEL_DESCRIPTIONS[level]}
                        for level in EDIT_LEVEL_DESCRIPTIONS},
        "unsupported_edits": ["adding/removing objects", "changing a person's appearance, expression or clothes",
                              "baking text/logos into the canvas", "pixel-exact background replacement behind people"],
        "headless": True,
        "atomic_output": True,
        "invocation": ["--thumbnail-bridge-json", "--image-bridge (youtubesum json-stdin)",
                       "--action/--project-dir (youtubesum cli)"],
    }


def _status_payload(request: ThumbnailBridgeRequest) -> dict[str, Any]:
    from .thumbnail_bridge_ai import MEMORY_PROFILES, select_memory_profile

    env, models_dir, model_id = _environment(request)
    writable = _project_writable(request.project_path)
    profile = select_memory_profile(env.get("vram_bytes"), str(request.options.get("memory_profile") or ""))
    return {
        "bridge_protocol_version": 1,
        "project_dir_writable": writable,
        "environment": env,
        "models_dir": str(models_dir),
        "model_id": model_id,
        "runtime_profile": {
            "memory_profile": profile,
            "settings": {key: value for key, value in MEMORY_PROFILES[profile].items() if key != "working_size"},
            "generation_working_size": list(MEMORY_PROFILES[profile]["working_size"]),
            "final_size": list(FINAL_SIZE),
            "oom_policy": "one deterministic retry with the next lower-memory profile",
        },
        "capabilities": _capabilities(env, writable),
    }


def _require_generation_ready(env: dict[str, Any], *, reference: bool = False) -> None:
    from .thumbnail_bridge_ai import BridgeActionError

    status = env.get("status")
    if status == "gpu_unavailable":
        raise BridgeActionError("GPU_UNAVAILABLE", "CUDA GPU is unavailable; SDXL generation is disabled on CPU.")
    if status == "package_missing":
        raise BridgeActionError("PACKAGE_MISSING", "SDXL Python packages are not installed.")
    if status != "ready":
        reason = (env.get("model") or {}).get("failure_reason") or status
        raise BridgeActionError("MODEL_NOT_READY", f"SDXL model is not prepared: {reason}")
    if reference and env.get("reference_status") != "ready":
        reason = (env.get("ip_adapter") or {}).get("failure_reason") or env.get("reference_status")
        raise BridgeActionError("MODEL_NOT_READY", f"IP-Adapter reference model is not prepared: {reason}")


def _engine_factory(model_id: str):
    from .generation import SDXLTextToImageEngine
    from .thumbnail_bridge_ai import MEMORY_PROFILES

    def make(profile: str) -> SDXLTextToImageEngine:
        engine = SDXLTextToImageEngine(model_id=model_id, local_files_only=True)
        engine.memory_profile = {key: value for key, value in MEMORY_PROFILES[profile].items() if key != "working_size"}
        return engine

    return make


# ------------------------------------------------------------------ option helpers
def _canvas_options(request: ThumbnailBridgeRequest, warnings: list[str]) -> tuple[tuple[int, int], str, bool]:
    from .thumbnail_bridge_ai import BridgeActionError

    options = request.options
    ratio = str(options.get("ratio") or "16:9")
    if ratio != "16:9":
        raise BridgeActionError("UNSUPPORTED_OPTION", f"Only 16:9 thumbnails are supported (got {ratio}).")
    try:
        width, height = int(options.get("width") or FINAL_SIZE[0]), int(options.get("height") or FINAL_SIZE[1])
    except (TypeError, ValueError) as exc:
        raise BridgeActionError("INVALID_REQUEST", "width/height must be integers.") from exc
    if width * 9 != height * 16 or not 640 <= width <= 3840:
        raise BridgeActionError("UNSUPPORTED_OPTION", f"{width}x{height} is not a supported 16:9 size.")
    if options.get("textless") is False:
        warnings.append("textless=false ignored: the bridge always produces a textless canvas.")
    side = str(options.get("prefer_text_space") or "left").casefold()
    side = side if side in ("left", "right", "top", "bottom", "auto", "none") else "left"
    preserve_people = options.get("preserve_people", True) is not False
    return (width, height), side, preserve_people


def _int_option(options: dict[str, Any], key: str, default: int, low: int, high: int) -> int:
    try:
        return max(low, min(high, int(options.get(key) or default)))
    except (TypeError, ValueError):
        return default


def _float_option(options: dict[str, Any], key: str, default: float, low: float, high: float) -> float:
    try:
        value = options.get(key)
        return max(low, min(high, float(default if value in (None, "") else value)))
    except (TypeError, ValueError):
        return default


def _commit(project_dir: Path, canvas, sidecars, preview) -> list[str]:
    from .thumbnail_bridge_assets import commit_outputs, stage_outputs

    staging = stage_outputs(project_dir, canvas, sidecars, preview)
    return commit_outputs(project_dir, staging)


def _outputs_map(written: list[str]) -> dict[str, str]:
    names = {path.name: key for key, path in standard_output_paths(Path(".")).items()}
    return {names[name]: name for name in written if name in names}


# ------------------------------------------------------------------ generate
def _generate(request: ThumbnailBridgeRequest) -> ThumbnailBridgeResponse:
    from .thumbnail_bridge_ai import build_prompt_plan, generate_with_memory_fallback, select_memory_profile
    from .thumbnail_bridge_assets import analyze_canvas, build_sidecars, cleanup_stale_staging, fit_16x9, flag_missing_faces

    warnings: list[str] = []
    final_size, text_side, preserve_people = _canvas_options(request, warnings)
    project_dir = request.project_path
    if not _project_writable(project_dir):
        raise _action_error("PROJECT_NOT_WRITABLE", f"project_dir is not writable: {project_dir}")
    cleanup_stale_staging(project_dir)
    env, models_dir, model_id = _environment(request)
    _require_generation_ready(env)
    options = request.options
    profile = select_memory_profile(env.get("vram_bytes"), str(options.get("memory_profile") or ""))
    seed = _int_option(options, "seed", random.randint(1, 2**31 - 1), 0, 2**31 - 1)
    steps = _int_option(options, "steps", DEFAULT_STEPS, 8, 60)
    guidance = _float_option(options, "guidance_scale", DEFAULT_GUIDANCE, 1.0, 15.0)
    plan = build_prompt_plan(request.prompt, request.channel, text_side, preserve_people)
    if plan.untranslated_text:
        warnings.append("Part of the non-English prompt has no translation; SDXL understands English prompts best.")
    _log(f"generate: GPU={env.get('gpu')} VRAM={env.get('vram_bytes')} profile={profile} project={project_dir}")
    image, metrics = generate_with_memory_fallback(
        _engine_factory(model_id), profile, plan,
        {"model_id": model_id, "steps": steps, "guidance_scale": guidance, "local_files_only": True, "seed": seed},
        seed, warnings)
    canvas = fit_16x9(image, final_size)
    analysis = analyze_canvas(canvas, preferred_side=text_side, protagonist_side=str(options.get("protagonist_side") or ""),
                              expect_people=preserve_people)
    if warning := flag_missing_faces(analysis, _expected_people(plan.prompt_parts[0]) if preserve_people else 0):
        warnings.append(warning)
    if text_side in ("left", "right") and not any(region["name"].startswith(text_side) for region in analysis.preferred_regions):
        warnings.append(f"The generated composition has no clean {text_side} text area (people are there). "
                        f"Try another seed, or edit: move the people and widen the {text_side} text space.")
    generation = {
        "backend": "CoverMorph SDXLTextToImageEngine (SDXL base 1.0, text-to-image)",
        "gpu": env.get("gpu"),
        "vram_bytes": env.get("vram_bytes"),
        "torch": env.get("torch"),
        "diffusers": env.get("diffusers"),
        "model_id": model_id,
        "user_prompt": request.prompt,
        "translated_terms": plan.translated_terms,
        "final_size": list(final_size),
        "resize": "center crop to 16:9 + Lanczos (no stretch)",
        **metrics,
    }
    sidecars = build_sidecars(analysis, request=request, final_size=final_size, source_size=image.size,
                              scene_type=plan.scene_type, generation=generation)
    written = _commit(project_dir, canvas, sidecars, image)
    return ThumbnailBridgeResponse(
        request_id=request.request_id, action=request.action, project_dir=request.project_dir, ok=True,
        outputs=_outputs_map(written), warnings=warnings, message="Generated thumbnail bridge assets.",
        details={"generation": generation, "subjects": sidecars["subject_boxes"]["subjects"]})


def _expected_people(prompt: str) -> int:
    """People count implied by the prompt wording (used only to flag missed face detections)."""
    text = prompt.casefold()
    if any(word in text for word in ("two people", "couple", " and a ", "two ", "pair")):
        return 2
    return 1 if any(word in text for word in ("man", "woman", "person", "girl", "boy", "people")) else 0


def _action_error(code: str, message: str, details: dict[str, Any] | None = None):
    from .thumbnail_bridge_ai import BridgeActionError

    return BridgeActionError(code, message, details)


# ------------------------------------------------------------------ edit
def _reference_crop(canvas, subjects):
    """Crop the people from the current canvas as the IP-Adapter person reference."""
    width, height = canvas.size
    people = [subject for subject in subjects if subject.role in ("protagonist", "counterpart")] or subjects
    if not people:
        return canvas.copy()
    left = min(subject.box[0] for subject in people)
    top = min(subject.box[1] for subject in people)
    right = max(subject.box[0] + subject.box[2] for subject in people)
    bottom = max(subject.box[1] + subject.box[3] for subject in people)
    pad_x, pad_y = (right - left) * 0.08, (bottom - top) * 0.08
    box = (int(max(0, left - pad_x) * width), int(max(0, top - pad_y) * height),
           int(min(1, right + pad_x) * width), int(min(1, bottom + pad_y) * height))
    return canvas.crop(box) if box[2] - box[0] > 32 and box[3] - box[1] > 32 else canvas.copy()


def _edit(request: ThumbnailBridgeRequest) -> ThumbnailBridgeResponse:
    from PIL import Image

    from .generation import DEFAULT_IP_ADAPTER_REVISION
    from .project import utc_now
    from .thumbnail_bridge_ai import (
        EDIT_LEVELS,
        apply_recompose,
        build_prompt_plan,
        generate_with_memory_fallback,
        parse_edit_instruction,
        select_memory_profile,
    )
    from .thumbnail_bridge_assets import (
        analyze_canvas,
        build_sidecars,
        cleanup_stale_staging,
        detect_subjects,
        fit_16x9,
        flag_missing_faces,
        merge_subjects,
        read_json,
        subjects_from_known,
    )

    warnings: list[str] = []
    final_size, requested_side, _ = _canvas_options(request, warnings)
    project_dir = request.project_path
    paths = standard_output_paths(project_dir)
    options = request.options
    source = Path(str(options.get("source_image") or paths["canvas_clean"])).expanduser()
    if not source.is_absolute():
        source = project_dir / source
    if not source.is_file():
        raise _action_error("NO_SOURCE_CANVAS", f"No canvas to edit: {source}. Run generate first.")
    if not _project_writable(project_dir):
        raise _action_error("PROJECT_NOT_WRITABLE", f"project_dir is not writable: {project_dir}")
    cleanup_stale_staging(project_dir)
    forced = str(options.get("edit_mode") or "")
    if forced and forced not in EDIT_LEVELS:
        raise _action_error("INVALID_REQUEST", f"edit_mode must be one of {', '.join(EDIT_LEVELS)}.")
    plan = parse_edit_instruction(request.edit_instruction, forced)
    env, models_dir, model_id = _environment(request)
    capabilities = _capabilities(env, True)
    details = {"edit": plan.to_dict(), "available_edit_levels": capabilities["edit"],
               "edit_levels": capabilities["edit_levels"]}
    if plan.unsupported:
        raise _action_error("UNSUPPORTED_EDIT", "This edit needs capabilities the bridge does not have: "
                            + "; ".join(plan.unsupported), details)
    if not plan.level:
        raise _action_error("UNSUPPORTED_EDIT", "The edit instruction was not recognized as a supported "
                            "recompose/regenerate/reference_regenerate edit.", details)
    if plan.level not in capabilities["edit"]:
        raise _action_error("UNSUPPORTED_EDIT", f"Edit level '{plan.level}' is not available on this machine "
                            "(model/adapter not ready).", details)

    with Image.open(source) as opened:
        before = opened.convert("RGB")
    before = before if before.size == final_size else fit_16x9(before, final_size)
    manifest = read_json(paths["project_manifest"])
    composition = read_json(paths["composition"])
    known = subjects_from_known(read_json(paths["subject_boxes"]).get("subjects")) if source == paths["canvas_clean"] else []
    fresh, _ = detect_subjects(before, text_side=str(composition.get("recommended_text_side") or "left"), expect_people=not known)
    subjects = merge_subjects(known, fresh) if known else fresh
    text_side = plan.simplify_side or (requested_side if "prefer_text_space" in options else "") \
        or str(composition.get("recommended_text_side") or "left")
    generation = manifest.get("generation") if isinstance(manifest.get("generation"), dict) else None
    edit_record: dict[str, Any] = {"at": utc_now(), "instruction": request.edit_instruction, "level": plan.level,
                                   "recognized": plan.recognized}
    notes: list[str] = []
    if plan.level == "recompose":
        after, subjects, notes = apply_recompose(before, plan, subjects)
        source_size = final_size
        analysis = analyze_canvas(after, preferred_side=text_side, known_subjects=subjects)
    else:
        reference = plan.level == "reference_regenerate"
        _require_generation_ready(env, reference=reference)
        previous_prompt = str((generation or {}).get("user_prompt") or "")
        prompt = request.prompt or (", ".join(plan.scene_terms) if plan.scene_terms else previous_prompt)
        prompt_plan = build_prompt_plan(prompt, request.channel or str(manifest.get("channel") or ""), text_side, True)
        profile = select_memory_profile(env.get("vram_bytes"), str(options.get("memory_profile") or ""))
        seed = _int_option(options, "seed", random.randint(1, 2**31 - 1), 0, 2**31 - 1)
        config = {"model_id": model_id, "steps": _int_option(options, "steps", DEFAULT_STEPS, 8, 60),
                  "guidance_scale": _float_option(options, "guidance_scale", DEFAULT_GUIDANCE, 1.0, 15.0),
                  "local_files_only": True, "seed": seed}
        reference_image = None
        if reference:
            config.update({"reference_mode": "person", "ip_adapter_id": str(models_dir / "ip_adapter"),
                           "ip_adapter_revision": DEFAULT_IP_ADAPTER_REVISION,
                           "reference_strength": _float_option(options, "reference_strength", DEFAULT_REFERENCE_STRENGTH, 0.0, 1.0)})
            reference_image = _reference_crop(before, subjects)
            warnings.append("reference_regenerate keeps people similar via IP-Adapter; exact pixels/poses are not guaranteed.")
        image, metrics = generate_with_memory_fallback(_engine_factory(model_id), profile, prompt_plan, config, seed,
                                                       warnings, reference_image=reference_image)
        after = fit_16x9(image, final_size)
        if plan.has_recompose_ops:
            fresh, _ = detect_subjects(after, text_side=text_side)
            after, _, notes = apply_recompose(after, plan, fresh)
        source_size = image.size
        analysis = analyze_canvas(after, preferred_side=text_side)
        expected = sum(1 for subject in subjects if subject.role in ("protagonist", "counterpart")) if reference else 0
        if warning := flag_missing_faces(analysis, max(expected, _expected_people(prompt))):
            warnings.append(warning)
        generation = {"backend": "CoverMorph SDXLTextToImageEngine" + (" + IP-Adapter Plus SDXL" if reference else ""),
                      "gpu": env.get("gpu"), "vram_bytes": env.get("vram_bytes"),
                      "model_id": model_id, "user_prompt": prompt, "edit_of": (manifest.get("request_id") or ""),
                      "final_size": list(final_size), **metrics}
        edit_record["generation_seconds"] = metrics.get("generation_time_seconds")
    analysis.notes.extend(notes)
    if plan.simplify_side and not any(region["name"].startswith(plan.simplify_side) for region in analysis.preferred_regions):
        warnings.append(f"The {plan.simplify_side} side could not be fully cleared without cutting into people; "
                        "background there was simplified around them. A regenerate edit can recompose the scene.")
    history = list(manifest.get("edit_history") or []) + [edit_record]
    sidecars = build_sidecars(analysis, request=_with_manifest_defaults(request, manifest), final_size=final_size,
                              source_size=source_size, scene_type=str(composition.get("scene_type") or "story"),
                              generation=generation, history=history)
    written = _commit(project_dir, after, sidecars, before)
    details.update({"generation": generation if plan.level != "recompose" else None, "notes": notes})
    return ThumbnailBridgeResponse(
        request_id=request.request_id, action=request.action, project_dir=request.project_dir, ok=True,
        outputs=_outputs_map(written), warnings=warnings,
        message=f"Edited canvas ({plan.level}: {', '.join(plan.recognized) or 'regenerated'}).", details=details)


def _with_manifest_defaults(request: ThumbnailBridgeRequest, manifest: dict[str, Any]) -> ThumbnailBridgeRequest:
    """Edits without metadata keep the project's existing channel/title fields."""
    for name in ("channel", "story_type", "episode", "title", "subtitle", "preferred_typography"):
        if not getattr(request, name) and manifest.get(name):
            setattr(request, name, str(manifest[name]))
    return request


# ------------------------------------------------------------------ dispatch
def handle_request(request: ThumbnailBridgeRequest) -> ThumbnailBridgeResponse:
    from .thumbnail_bridge_ai import BridgeActionError
    from .thumbnail_bridge_assets import BridgeOutputError

    base = {"request_id": request.request_id, "action": request.action, "project_dir": request.project_dir}
    if request.action == "status":
        payload = _status_payload(request)
        ready = bool(payload["project_dir_writable"])
        return ThumbnailBridgeResponse(
            **base, ok=ready, message="CoverMorph thumbnail bridge status.", outputs={"status": payload},
            warnings=[] if ready else ["project_dir is not writable"], error_code="" if ready else "PROJECT_NOT_WRITABLE")
    try:
        return _generate(request) if request.action == "generate" else _edit(request)
    except BridgeActionError as exc:
        return ThumbnailBridgeResponse(**base, ok=False, message=str(exc), error_code=exc.code, details=exc.details)
    except BridgeOutputError as exc:
        return ThumbnailBridgeResponse(**base, ok=False, message=f"Outputs were not committed: {exc}",
                                       error_code="OUTPUT_COMMIT_FAILED")
    except Exception as exc:
        from .generation import GenerationError

        _write_exception(f"thumbnail bridge {request.action}", exc)
        code = "GENERATION_FAILED" if isinstance(exc, GenerationError) else "INTERNAL_ERROR"
        return ThumbnailBridgeResponse(**base, ok=False, message=f"{type(exc).__name__}: {exc}"[:800],
                                       error_code=code, warnings=["Previous bridge outputs were left unchanged."])


def _write_exception(prefix: str, exc: BaseException) -> None:
    try:
        from .logger import write_exception

        write_exception(_app_root(), prefix, exc)
    except Exception:
        pass
    _log(f"{prefix}: {type(exc).__name__}: {exc}")


def request_from_youtubesum(data: dict[str, Any]) -> ThumbnailBridgeRequest:
    """Map youtubesum's ``youtubesum-image-bridge/1`` payload onto the contract request."""
    if not isinstance(data, dict):
        raise ThumbnailBridgeError("Bridge request must be a JSON object.")
    options = dict(data.get("options") or {})
    mapped = {
        "protocol_version": 1,
        "request_id": str(data.get("request_id") or options.pop("request_id", "") or f"youtubesum-{uuid.uuid4().hex[:12]}"),
        "action": data.get("action") or "status",
        "project_dir": data.get("project_dir") or "",
        "prompt": data.get("prompt") or "",
        "edit_instruction": data.get("edit_instruction") or "",
        "options": options,
    }
    for name in ("channel", "story_type", "episode", "title", "subtitle", "preferred_typography"):
        mapped[name] = data.get(name) or options.pop(name, "") or ""
    for name in ("channel", "story_type", "episode", "title", "subtitle", "preferred_typography"):
        options.pop(name, None)
    return ThumbnailBridgeRequest.from_dict(mapped)


def parse_argv_request(argv: list[str]) -> ThumbnailBridgeRequest:
    parser = argparse.ArgumentParser(prog="CoverMorphStudio", add_help=False, exit_on_error=False)
    parser.add_argument("--action", required=True)
    parser.add_argument("--project-dir", required=True)
    for name in ("--channel", "--story-type", "--title", "--subtitle", "--episode", "--preferred-typography",
                 "--prompt", "--edit-instruction", "--request-id"):
        parser.add_argument(name, default="")
    parser.add_argument("--options-json", default="{}")
    try:
        args, _unknown = parser.parse_known_args(argv)
        options = json.loads(args.options_json or "{}")
    except (argparse.ArgumentError, json.JSONDecodeError, SystemExit) as exc:
        raise ThumbnailBridgeError(f"Invalid bridge arguments: {exc}") from exc
    return request_from_youtubesum({
        "action": args.action, "project_dir": args.project_dir, "channel": args.channel,
        "story_type": args.story_type, "title": args.title, "subtitle": args.subtitle, "episode": args.episode,
        "preferred_typography": args.preferred_typography, "prompt": args.prompt,
        "edit_instruction": args.edit_instruction, "request_id": args.request_id,
        "options": options if isinstance(options, dict) else {},
    })


def _read_stdin_text() -> str:
    stream = getattr(sys.stdin, "buffer", None)
    data = stream.read() if stream is not None else b""
    if not data and sys.stdin is None:
        chunks = []
        try:
            while chunk := os.read(0, 65536):
                chunks.append(chunk)
        except OSError:
            pass
        data = b"".join(chunks)
    return data.decode("utf-8-sig")


def parse_request(argv: list[str], stdin_text: str | None = None) -> ThumbnailBridgeRequest:
    if "--thumbnail-bridge-json" in argv:
        return read_request_json(_read_stdin_text() if stdin_text is None else stdin_text)
    if "--image-bridge" in argv:
        text = _read_stdin_text() if stdin_text is None else stdin_text
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ThumbnailBridgeError(f"Invalid request JSON: {exc}") from exc
        if isinstance(data, dict) and "protocol_version" in data:
            return ThumbnailBridgeRequest.from_dict(data)
        return request_from_youtubesum(data)
    return parse_argv_request(argv)


@contextmanager
def _stdout_reserved() -> Iterator[int | None]:
    """Route Python and native stdout writes to stderr; yield a dup of the real stdout fd."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except Exception:
            pass
    if sys.stderr is None:
        sys.stderr = open(os.devnull, "w", encoding="utf-8")
    saved_fd: int | None = None
    try:
        saved_fd = os.dup(1)
        os.dup2(2, 1)
    except OSError:
        saved_fd = None
    previous = sys.stdout
    sys.stdout = sys.stderr
    try:
        yield saved_fd
    finally:
        try:
            sys.stderr.flush()
        except Exception:
            pass
        sys.stdout = previous
        if saved_fd is not None:
            os.dup2(saved_fd, 1)
            os.close(saved_fd)


def _emit(text: str) -> None:
    data = text.encode("utf-8")
    try:
        while data:
            written = os.write(1, data)
            data = data[written:]
    except OSError:
        if sys.stdout is not None:
            sys.stdout.buffer.write(data)
            sys.stdout.flush()


def run_bridge_cli(stdin_text: str | None = None, argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not any(flag in argv for flag in BRIDGE_FLAGS) and "--action" not in argv:
        argv.append("--thumbnail-bridge-json")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    with _stdout_reserved():
        request: ThumbnailBridgeRequest | None = None
        try:
            request = parse_request(argv, stdin_text)
            _log(f"request {request.request_id or '-'} action={request.action} project={request.project_dir}")
            response = handle_request(request)
        except ThumbnailBridgeError as exc:
            response = ThumbnailBridgeResponse(request_id="", action="status", project_dir="", ok=False,
                                               message=str(exc), error_code="INVALID_REQUEST")
        except Exception as exc:  # structured boundary: never leak a traceback to stdout
            _write_exception("thumbnail bridge", exc)
            response = ThumbnailBridgeResponse(
                request_id=request.request_id if request else "", action=request.action if request else "status",
                project_dir=request.project_dir if request else "", ok=False, message=str(exc),
                error_code="INTERNAL_ERROR")
        _log(f"result ok={response.ok} code={response.error_code or '-'} {response.message}")
    _emit(write_response_json(response))
    return 0 if response.ok else 2

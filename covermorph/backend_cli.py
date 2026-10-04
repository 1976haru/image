"""Headless backend commands used by YouTube Dynamic Thumbnail Studio (the single user-facing app).

    CoverMorphStudio.exe --studio-job <payload.json> --result <result.json>
        Runs one creator job (YouTube/Shopify, references, product modes, candidates). Prints JSON progress lines
        on stdout ({"progress": 0.4, "message": "..."}) and writes the result file. Exit codes:
        0 done · 2 failed (result.error = {code, message, action}) · 3 waiting (resources short; nothing loaded)
    CoverMorphStudio.exe --editor-project <image> --out <folder> [--meta <meta.json>]
        Turns a chosen candidate into an editor project folder (canvas + subject/safe-zone/palette/composition/
        manifest sidecars) using the same analysis as the thumbnail bridge.
    CoverMorphStudio.exe --backend-status --result <status.json>
        Readiness for the app's 설정/Home: engines, translator, GPU, models folder, version.

The engine processes started by a job exit with it, so memory returns when the command ends.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from threading import Event
from typing import Any

from . import __version__

EXIT_DONE, EXIT_FAILED, EXIT_WAITING = 0, 2, 3


def _write(path: Path, data: dict[str, Any]) -> None:
    from .app_paths import atomic_write_json
    atomic_write_json(Path(path), data)


def _emit(data: dict[str, Any]) -> None:
    try:
        sys.stdout.write(json.dumps(data, ensure_ascii=False) + "\n")
        sys.stdout.flush()
    except (OSError, ValueError):
        pass


def _settings(payload: dict[str, Any]) -> tuple[Path, Path, dict[str, Any]]:
    from .creator_settings import load_creator_settings, resolve_models_dir, resolve_output_dir
    root = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parents[1]
    settings = load_creator_settings(root)
    if payload.get("models_dir"):
        settings["models_dir"] = str(payload["models_dir"])
    models = resolve_models_dir(root, settings)
    output = Path(payload["output_root"]) if payload.get("output_root") else resolve_output_dir(root, settings)
    return models, output, settings


def run_studio_job(payload_path: Path, result_path: Path) -> int:
    from .creator_jobs import run_creator_job
    from .errors import classify, preflight, remember_last_error, write_details
    from .quality_engines import check_resources, resource_snapshot

    started = time.perf_counter()
    try:
        payload = json.loads(Path(payload_path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        _write(result_path, {"status": "failed", "error": {"code": "PROJECT_CORRUPT", "message": f"작업 정보를 읽지 못했습니다: {exc}",
                                                         "action": "작업을 다시 만들어 주세요."}})
        return EXIT_FAILED
    models, output, _ = _settings(payload)
    if not payload.get("skip_resource_check"):
        ok, reasons = check_resources(payload.get("memory") or "interactive_low_memory", resource_snapshot())
        if not ok:
            _write(result_path, {"status": "waiting", "reasons": reasons, "version": __version__})
            return EXIT_WAITING

    def progress(event: dict[str, Any]) -> None:
        if event.get("phase") == "stage":
            _emit({"progress": float(event.get("fraction", 0)), "message": event.get("message", "")})
        elif event.get("phase") == "inference" and event.get("steps"):
            _emit({"step": event.get("step"), "steps": event.get("steps")})
        elif event.get("phase") == "waiting":
            _emit({"message": event.get("message", "")})

    try:
        preflight(payload, models, output)
        result = run_creator_job(payload, Event(), models_dir=models, output_root=output, progress=progress)
    except Exception as exc:
        error = classify(exc)
        details = write_details(error, str(payload.get("job_id") or "job"))
        remember_last_error(error, details)
        _write(result_path, {"status": "failed", "error": {"code": error.code, "title": error.title,
                                                          "message": error.message, "action": error.action,
                                                          "details_file": str(details or "")},
                             "seconds": round(time.perf_counter() - started, 1), "version": __version__})
        return EXIT_FAILED
    _write(result_path, {"status": "done", "version": __version__, **result})
    return EXIT_DONE


def make_editor_project(image_path: Path, out_dir: Path, meta: dict[str, Any] | None = None) -> int:
    """Chosen candidate -> editor project folder with the bridge's sidecars (subjects, safe zones, palette...)."""
    from PIL import Image

    from .thumbnail_bridge import ThumbnailBridgeRequest
    from .thumbnail_bridge_assets import analyze_canvas, build_sidecars, commit_outputs, stage_outputs

    meta = dict(meta or {})
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with Image.open(image_path) as opened:
        canvas = opened.convert("RGB")
    people = int(meta.get("people") or 0)
    analysis = analyze_canvas(canvas, preferred_side=str(meta.get("text_side") or "left"), expect_people=people > 0)
    request = ThumbnailBridgeRequest.from_dict({
        "protocol_version": 1, "request_id": str(meta.get("job_id") or "studio"), "action": "generate",
        "project_dir": str(out_dir), "channel": str(meta.get("channel") or ""), "title": str(meta.get("title") or ""),
        "subtitle": str(meta.get("subtitle") or ""), "episode": str(meta.get("episode") or ""),
        "story_type": str(meta.get("story_type") or ""), "prompt": str(meta.get("prompt") or ""),
        "preferred_typography": str(meta.get("preferred_typography") or "")})
    generation = {"backend": meta.get("engine", ""), "seed": meta.get("seed"), "user_prompt": meta.get("prompt", ""),
                  "purpose": meta.get("purpose", ""), "candidate": str(image_path)}
    extra = {"purpose": meta.get("purpose", ""), "cta": meta.get("cta", ""), "studio_job": meta.get("job_id", "")}
    sidecars = build_sidecars(analysis, request=request, final_size=canvas.size, source_size=canvas.size,
                              scene_type="story", generation=generation, extra_manifest=extra)
    staging = stage_outputs(out_dir, canvas, sidecars, canvas)
    commit_outputs(out_dir, staging)
    return EXIT_DONE


def backend_status(result_path: Path, models_dir: str = "") -> int:
    from .creator_settings import engine_readiness
    from .diagnostics import _gpu
    from .quality_engines import resource_snapshot

    models, _, settings = _settings({"models_dir": models_dir} if models_dir else {})
    ready = engine_readiness(models) if models.is_dir() else {"engines": {}, "translator": False}
    snap = resource_snapshot()
    engines = ready["engines"]
    _write(result_path, {
        "version": __version__, "models_dir": str(models), "models_dir_exists": models.is_dir(),
        "ready": bool(engines.get("zimage_turbo", {}).get("ready") and engines.get("flux2_klein_4b", {}).get("ready")),
        "engines": {k: v["ready"] for k, v in engines.items()}, "missing": {k: v["missing"] for k, v in engines.items()},
        "translator": ready["translator"], "gpu": _gpu(), "gpu_free_mib": snap.gpu_free_mib,
        "ram_available_mib": snap.ram_available_mib, "product_mode_default": settings.get("product_mode")})
    return EXIT_DONE


def main(argv: list[str]) -> int:
    def arg(flag: str) -> str | None:
        return argv[argv.index(flag) + 1] if flag in argv and argv.index(flag) + 1 < len(argv) else None

    if argv[0] == "--studio-job":
        return run_studio_job(Path(argv[1]), Path(arg("--result") or Path(argv[1]).with_suffix(".result.json")))
    if argv[0] == "--editor-project":
        meta = json.loads(Path(arg("--meta")).read_text(encoding="utf-8")) if arg("--meta") else {}
        return make_editor_project(Path(argv[1]), Path(arg("--out")), meta)
    if argv[0] == "--backend-status":
        return backend_status(Path(arg("--result")), arg("--models-dir") or "")
    return 64

"""Creator settings saved from the UI (no environment variables needed for normal use).

``config/creator_settings.json`` next to the EXE (or repo root). Resolution order for the models folder:
saved setting > COVERMORPH_MODELS_DIR > <app>/models. The bridge reads the same file, so a path chosen in
the UI also works for youtubesum calls.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

DEFAULTS: dict[str, Any] = {
    "models_dir": "",
    "sdcpp_dir": "",            # blank = <models>/quality_v2/sdcpp
    "output_dir": "",           # blank = <app>/creator_output
    "quality_label": "일반",
    "memory_label": "작업 중 PC 우선",
    "jpg_quality": 92,
    "auto_start_when_free": True,   # queue starts by itself when resources allow
    "require_idle_minutes": 0,      # >0: in 자리 비움 mode only start after this much keyboard/mouse idle
    "recheck_seconds": 20,
}


def settings_file(app_root: Path) -> Path:
    return Path(app_root) / "config" / "creator_settings.json"


def load_creator_settings(app_root: Path) -> dict[str, Any]:
    data: dict[str, Any] = {}
    try:
        data = json.loads(settings_file(app_root).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        data = {}
    return {**DEFAULTS, **{key: data[key] for key in DEFAULTS if key in data}}


def save_creator_settings(app_root: Path, settings: dict[str, Any]) -> None:
    path = settings_file(app_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    clean = {**DEFAULTS, **{key: settings[key] for key in DEFAULTS if key in settings}}
    fd, tmp = tempfile.mkstemp(prefix=".creator_", suffix=".json", dir=path.parent)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(clean, handle, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def resolve_models_dir(app_root: Path, settings: dict[str, Any] | None = None) -> Path:
    settings = settings if settings is not None else load_creator_settings(app_root)
    configured = str(settings.get("models_dir") or os.environ.get("COVERMORPH_MODELS_DIR") or "")
    return Path(configured).expanduser() if configured else Path(app_root) / "models"


def resolve_output_dir(app_root: Path, settings: dict[str, Any] | None = None) -> Path:
    settings = settings if settings is not None else load_creator_settings(app_root)
    configured = str(settings.get("output_dir") or "")
    return Path(configured).expanduser() if configured else Path(app_root) / "creator_output"


def apply_backend_paths(settings: dict[str, Any]) -> None:
    """Make a UI-chosen sd.cpp folder visible to quality_engines (process-local, nothing persisted)."""
    if settings.get("sdcpp_dir"):
        os.environ["COVERMORPH_SDCPP_DIR"] = str(settings["sdcpp_dir"])


def engine_readiness(models_dir: Path) -> dict[str, Any]:
    from .prompt_translate import translator_ready
    from .quality_engines import make_backend

    engines = {}
    for name in ("zimage_turbo", "flux2_klein_4b", "realvisxl_v5"):
        status = make_backend(name, models_dir).status()
        engines[name] = {"ready": status["ready"],
                         "missing": [f"{k}: {v}" for k, v in status["files"].items() if not Path(v).exists()]}
    return {"engines": engines, "translator": translator_ready(models_dir)}

"""Studio settings, saved per user in %LOCALAPPDATA%\\CoverMorphStudio\\settings.json ("studio" section).

They survive EXE rebuilds/updates (see app_paths). Resolution order for the models folder: saved setting >
COVERMORPH_MODELS_DIR > <app>/models. The youtubesum bridge reads the same file, so a path chosen in the UI also
works for bridge calls. ``app_root`` is only used to migrate settings that older builds kept in the EXE folder.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from . import app_paths

DEFAULTS: dict[str, Any] = {
    "models_dir": "",
    "sdcpp_dir": "",            # blank = <models>/quality_v2/sdcpp
    "output_dir": "",           # blank = Documents\\CoverMorphStudio
    "quality_label": "일반",
    "memory_label": "작업 중 PC 우선",
    "jpg_quality": 92,
    "auto_start_when_free": True,   # queue starts by itself when resources allow
    "require_idle_minutes": 0,      # >0: in 자리 비움 mode only start after this much keyboard/mouse idle
    "recheck_seconds": 20,
    # remembered between sessions
    "usage": "both",                # youtube | shopify | both (first-run wizard)
    "setup_completed": False,
    "last_purpose": "youtube_thumbnail",
    "custom_sizes": {},             # purpose key -> [w, h] last used
    # natural | strict | ai. Default chosen from the RC1 visual review: natural-light composite keeps the product's
    # pixels and fits the scene best; strict is the absolute-preservation option; ai can change label/shape.
    "product_mode": "natural",
    "natural_strength": "default",  # weak | default | strong (약하게 / 기본 / 강하게)
    "last_export_dir": "",
    "last_project_dir": "",
    "last_reference_dir": "",
}
_migrated: set[str] = set()


def _migrate_once(app_root: Path) -> None:
    key = str(Path(app_root).resolve())
    if key not in _migrated:
        _migrated.add(key)
        app_paths.migrate_from_app_folder(Path(app_root))


def settings_file(app_root: Path | None = None) -> Path:
    return app_paths.settings_file()


def load_creator_settings(app_root: Path) -> dict[str, Any]:
    _migrate_once(app_root)
    data = app_paths.section("studio")
    settings = {**DEFAULTS, **{key: data[key] for key in DEFAULTS if key in data}}
    if not isinstance(settings["custom_sizes"], dict):
        settings["custom_sizes"] = {}
    try:
        settings["jpg_quality"] = max(50, min(100, int(settings["jpg_quality"])))
    except (TypeError, ValueError):
        settings["jpg_quality"] = DEFAULTS["jpg_quality"]
    return settings


def save_creator_settings(app_root: Path, settings: dict[str, Any]) -> None:
    app_paths.update_section("studio", {key: settings[key] for key in DEFAULTS if key in settings})


def resolve_models_dir(app_root: Path, settings: dict[str, Any] | None = None) -> Path:
    settings = settings if settings is not None else load_creator_settings(app_root)
    configured = str(settings.get("models_dir") or os.environ.get("COVERMORPH_MODELS_DIR") or "")
    return Path(configured).expanduser() if configured else Path(app_root) / "models"


def resolve_output_dir(app_root: Path, settings: dict[str, Any] | None = None) -> Path:
    settings = settings if settings is not None else load_creator_settings(app_root)
    configured = str(settings.get("output_dir") or "")
    return Path(configured).expanduser() if configured else app_paths.default_output_dir()


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


def setup_needed(app_root: Path, settings: dict[str, Any] | None = None) -> bool:
    """First-run wizard: not completed yet, or the default engines cannot be found."""
    settings = settings if settings is not None else load_creator_settings(app_root)
    if not settings.get("setup_completed"):
        return True
    ready = engine_readiness(resolve_models_dir(app_root, settings))["engines"]
    return not (ready["zimage_turbo"]["ready"] and ready["flux2_klein_4b"]["ready"])

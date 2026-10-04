"""Per-user data that must survive EXE rebuilds/updates: settings, queue, logs, app state.

Everything lives under ``%LOCALAPPDATA%\\CoverMorphStudio`` (override with ``COVERMORPH_DATA_DIR`` for a portable
install or tests), never inside the EXE/dist folder that PyInstaller replaces on every build.

settings.json::

    {"schema_version": 1, "studio": {...}, "main_app": {...}, "state": {...}, "migrated_from": [...]}

Writes are atomic (temp file + os.replace). A corrupt file is kept as ``settings.corrupt-<time>.json`` and the
app starts from defaults instead of crashing; the recovery is reported once through ``recovery_notes()``.
"""
from __future__ import annotations

import ctypes
import json
import os
import shutil
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
APP_DIR_NAME = "CoverMorphStudio"
_lock = threading.RLock()
_notes: list[str] = []


def data_dir() -> Path:
    override = os.environ.get("COVERMORPH_DATA_DIR")
    base = Path(override) if override else Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local") / APP_DIR_NAME
    base.mkdir(parents=True, exist_ok=True)
    return base


def settings_file() -> Path:
    return data_dir() / "settings.json"


def queue_file() -> Path:
    return data_dir() / "studio_queue.json"


def logs_dir() -> Path:
    path = data_dir() / "logs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def documents_dir() -> Path:
    """The user's real Documents folder (follows OneDrive/known-folder redirection)."""
    if os.name == "nt":
        buffer = ctypes.create_unicode_buffer(260)
        if ctypes.windll.shell32.SHGetFolderPathW(None, 5, None, 0, buffer) == 0 and buffer.value:  # CSIDL_PERSONAL
            return Path(buffer.value)
    return Path.home() / "Documents"


def default_output_dir() -> Path:
    return documents_dir() / APP_DIR_NAME


def recovery_notes() -> list[str]:
    """Messages about recovered/migrated data, shown once by the UI."""
    notes = list(_notes)
    _notes.clear()
    return notes


def atomic_write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.stem}_", suffix=".json", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def quarantine(path: Path, reason: str) -> Path:
    """Keep a damaged file next to the original for inspection; never delete user data."""
    target = path.with_name(f"{path.stem}.corrupt-{time.strftime('%Y%m%d_%H%M%S')}{path.suffix}")
    try:
        shutil.move(str(path), str(target))
    except OSError:
        target = path
    _notes.append(f"{path.name} 파일이 손상되어 기본값으로 시작했습니다 ({reason}). 원본은 {target.name}로 보관했습니다.")
    return target


def _empty() -> dict[str, Any]:
    return {"schema_version": SCHEMA_VERSION, "studio": {}, "main_app": {}, "state": {}, "migrated_from": []}


def _read() -> dict[str, Any]:
    path = settings_file()
    if not path.exists():
        return _empty()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("not a JSON object")
    except (OSError, ValueError) as exc:
        quarantine(path, str(exc)[:80])
        return _empty()
    version = int(data.get("schema_version") or 0)
    if version > SCHEMA_VERSION:
        _notes.append("설정 파일이 더 새 버전에서 저장되었습니다. 알 수 없는 항목은 그대로 둡니다.")
    for key in ("studio", "main_app", "state"):
        if not isinstance(data.get(key), dict):
            data[key] = {}
    data.setdefault("migrated_from", [])
    data["schema_version"] = max(version, SCHEMA_VERSION)
    return data


def load_all() -> dict[str, Any]:
    with _lock:
        return _read()


def section(name: str) -> dict[str, Any]:
    return dict(load_all().get(name) or {})


def update_section(name: str, values: dict[str, Any], replace: bool = False) -> None:
    with _lock:
        data = _read()
        data[name] = dict(values) if replace else {**(data.get(name) or {}), **values}
        atomic_write_json(settings_file(), data)


def migrate_from_app_folder(app_root: Path) -> list[str]:
    """One-time import of settings/queue that older builds kept inside the EXE folder (left in place, not deleted)."""
    app_root = Path(app_root)
    imported: list[str] = []
    with _lock:
        data = _read()
        done = set(data.get("migrated_from") or [])
        sources = {"studio": app_root / "config" / "creator_settings.json",
                   "main_app": app_root / "config" / "settings.json"}
        for name, source in sources.items():
            key = str(source.resolve()) if source.exists() else ""
            if not key or key in done or data.get(name):
                continue
            try:
                values = json.loads(source.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if isinstance(values, dict):
                data[name] = values
                done.add(key)
                imported.append(str(source))
        if imported:
            data["migrated_from"] = sorted(done)
            atomic_write_json(settings_file(), data)
    target = queue_file()
    if not target.exists():
        studio = data.get("studio") or {}
        candidates = [Path(studio["output_dir"]) / "studio_queue.json"] if studio.get("output_dir") else []
        candidates.append(app_root / "creator_output" / "studio_queue.json")
        for candidate in candidates:
            if candidate.exists():
                shutil.copy2(candidate, target)
                imported.append(str(candidate))
                break
    if imported:
        _notes.append("이전 버전의 설정/대기열을 사용자 폴더로 옮겼습니다: " + ", ".join(Path(p).name for p in imported))
    return imported


def sanitize_path(text: str) -> str:
    """Hide the user name in paths for copied diagnostics."""
    out = str(text)
    local = os.environ.get("LOCALAPPDATA")
    if local:  # inside the home folder: replace it first
        out = out.replace(local, "%LOCALAPPDATA%")
    return out.replace(str(Path.home()), "%USERPROFILE%")

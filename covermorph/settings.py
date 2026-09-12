from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

DEFAULT_SETTINGS: dict[str, Any] = {
    "output_directory": "",
    "output_square": True,
    "output_thumbnail": True,
    "output_shorts": True,
    "thumbnail_resolution": "1920x1080",
    "duplicate_policy": "new_number",
    "last_preset": "OldPopLounge",
    "last_ocr_language": "영어",
}

THUMBNAIL_RESOLUTIONS: dict[str, tuple[int, int]] = {
    "1920x1080": (1920, 1080),
    "1280x720": (1280, 720),
}
DUPLICATE_POLICIES = {
    "overwrite": "덮어쓰기",
    "new_number": "새 번호 붙이기",
    "skip": "건너뛰기",
}


class OutputDirectoryError(ValueError):
    pass


def settings_path(app_root: Path) -> Path:
    return app_root / "config" / "settings.json"


def normalize_settings(data: dict[str, Any] | None) -> dict[str, Any]:
    settings = DEFAULT_SETTINGS.copy()
    if isinstance(data, dict):
        settings.update({key: data[key] for key in settings if key in data})
    if settings["thumbnail_resolution"] not in THUMBNAIL_RESOLUTIONS:
        settings["thumbnail_resolution"] = DEFAULT_SETTINGS["thumbnail_resolution"]
    if settings["duplicate_policy"] not in DUPLICATE_POLICIES:
        settings["duplicate_policy"] = DEFAULT_SETTINGS["duplicate_policy"]
    for key in ("output_square", "output_thumbnail", "output_shorts"):
        settings[key] = bool(settings[key])
    settings["output_directory"] = str(settings.get("output_directory") or "")
    return settings


def load_settings(app_root: Path) -> dict[str, Any]:
    path = settings_path(app_root)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        settings = DEFAULT_SETTINGS.copy()
        save_settings(app_root, settings)
        return settings

    settings = normalize_settings(data)
    if settings != data:
        save_settings(app_root, settings)
    return settings


def save_settings(app_root: Path, settings: dict[str, Any]) -> None:
    normalized = normalize_settings(settings)
    path = settings_path(app_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(normalized, ensure_ascii=False, indent=2), encoding="utf-8")


def default_output_dir_for_source(source: Path) -> Path:
    return source.parent / "CoverMorph_Output"


def thumbnail_size(resolution: str) -> tuple[int, int]:
    return THUMBNAIL_RESOLUTIONS.get(resolution, THUMBNAIL_RESOLUTIONS["1920x1080"])


def has_selected_outputs(square: bool, thumbnail: bool, shorts: bool) -> bool:
    return square or thumbnail or shorts


def selected_output_labels(square: bool, thumbnail: bool, shorts: bool) -> list[str]:
    labels: list[str] = []
    if thumbnail:
        labels.append("16:9 썸네일")
    if shorts:
        labels.append("9:16 숏츠")
    if square:
        labels.append("1:1 클린 커버")
    return labels


def expected_output_count(file_count: int, square: bool, thumbnail: bool, shorts: bool) -> int:
    return file_count * len(selected_output_labels(square, thumbnail, shorts))


def is_valid_windows_path_text(path_text: str) -> bool:
    if not path_text.strip():
        return False
    invalid = '<>"|?*\0'
    return not any(char in path_text for char in invalid)


def ensure_output_directory(path_text: str, create: bool = False) -> Path:
    if not is_valid_windows_path_text(path_text):
        raise OutputDirectoryError("잘못된 출력 폴더 경로입니다.")

    path = Path(path_text).expanduser()
    if path.exists() and not path.is_dir():
        raise OutputDirectoryError("출력 경로가 폴더가 아닙니다.")
    if not path.exists():
        if not create:
            raise FileNotFoundError(str(path))
        path.mkdir(parents=True, exist_ok=True)

    test_file = path / ".covermorph_write_test.tmp"
    try:
        test_file.write_text("ok", encoding="utf-8")
    except OSError as exc:
        raise PermissionError(f"출력 폴더에 쓸 수 없습니다: {path}") from exc
    finally:
        try:
            test_file.unlink()
        except FileNotFoundError:
            pass
    return path


def next_numbered_path(path: Path) -> Path:
    if not path.exists():
        return path
    for number in range(2, 10000):
        candidate = path.with_name(f"{path.stem}_{number:02d}{path.suffix}")
        if not candidate.exists():
            return candidate
    raise FileExistsError(f"사용 가능한 새 파일명을 만들 수 없습니다: {path}")


def resolve_duplicate_path(path: Path, policy: str) -> Path | None:
    if policy == "overwrite" or not path.exists():
        return path
    if policy == "skip":
        return None
    return next_numbered_path(path)


def open_folder(path: Path) -> bool:
    try:
        if sys.platform.startswith("win") and hasattr(os, "startfile"):
            os.startfile(str(path))  # type: ignore[attr-defined]
            return True
        if sys.platform == "darwin":
            subprocess.Popen(["open", str(path)])
            return True
        subprocess.Popen(["xdg-open", str(path)])
        return True
    except (OSError, ValueError):
        return False

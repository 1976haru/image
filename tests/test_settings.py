from __future__ import annotations

import json
from pathlib import Path

from covermorph import settings as settings_module
from covermorph.settings import (
    DEFAULT_SETTINGS,
    default_output_dir_for_source,
    ensure_output_directory,
    expected_output_count,
    has_selected_outputs,
    load_settings,
    open_folder,
    save_settings,
    selected_output_labels,
)


def test_no_output_format_selected_is_blocked() -> None:
    assert has_selected_outputs(False, False, False) is False
    assert selected_output_labels(False, False, False) == []
    assert expected_output_count(3, False, False, False) == 0


def test_output_folder_settings_save(tmp_path: Path) -> None:
    output_dir = tmp_path / "CoverMorph_Output"
    data = DEFAULT_SETTINGS | {"output_directory": str(output_dir), "output_thumbnail": False}

    save_settings(tmp_path, data)
    loaded = load_settings(tmp_path)

    assert loaded["output_directory"] == str(output_dir)
    assert loaded["output_thumbnail"] is False


def test_output_folder_restored_after_restart(tmp_path: Path) -> None:
    output_dir = tmp_path / "saved output"
    save_settings(tmp_path, DEFAULT_SETTINGS | {"output_directory": str(output_dir)})

    first = load_settings(tmp_path)
    second = load_settings(tmp_path)

    assert first["output_directory"] == second["output_directory"] == str(output_dir)


def test_korean_space_path_supported(tmp_path: Path) -> None:
    output_dir = tmp_path / "한글 폴더 이름"
    resolved = ensure_output_directory(str(output_dir), create=True)

    assert resolved == output_dir
    assert output_dir.is_dir()


def test_missing_output_folder_can_be_created(tmp_path: Path) -> None:
    output_dir = tmp_path / "missing" / "CoverMorph_Output"
    assert not output_dir.exists()

    ensure_output_directory(str(output_dir), create=True)

    assert output_dir.is_dir()


def test_open_folder_uses_windows_startfile(monkeypatch, tmp_path: Path) -> None:
    calls: list[str] = []

    def fake_startfile(path: str) -> None:
        calls.append(path)

    monkeypatch.setattr(settings_module.sys, "platform", "win32")
    monkeypatch.setattr(settings_module.os, "startfile", fake_startfile, raising=False)

    assert open_folder(tmp_path) is True
    assert calls == [str(tmp_path)]


def test_corrupt_settings_json_recovers_defaults(tmp_path: Path) -> None:
    config = tmp_path / "config"
    config.mkdir()
    path = config / "settings.json"
    path.write_text("{broken json", encoding="utf-8")

    loaded = load_settings(tmp_path)
    repaired = json.loads(path.read_text(encoding="utf-8"))

    assert loaded["output_square"] is True
    assert repaired["thumbnail_resolution"] == "1920x1080"


def test_default_output_dir_uses_source_folder() -> None:
    source = Path("D:/음원커버/19 음악커버 1.png")
    assert default_output_dir_for_source(source) == Path("D:/음원커버/CoverMorph_Output")

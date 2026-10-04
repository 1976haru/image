from __future__ import annotations

import json
import shutil
from pathlib import Path

from covermorph import app_paths
from covermorph.creator_settings import load_creator_settings, resolve_output_dir, save_creator_settings
from covermorph.job_queue import JobQueue
from covermorph.settings import load_settings, save_settings


def _dist(tmp_path: Path, name: str = "dist") -> Path:
    folder = tmp_path / name / "CoverMorphStudio"
    folder.mkdir(parents=True)
    return folder


def test_settings_survive_exe_rebuild(tmp_path):
    exe_folder = _dist(tmp_path)
    settings = load_creator_settings(exe_folder)
    settings.update(models_dir=r"D:\models 모델", quality_label="최고 품질", memory_label="균형",
                    custom_sizes={"shopify_hero": [2000, 800]}, last_export_dir=str(tmp_path / "내보내기"))
    save_creator_settings(exe_folder, settings)
    main = load_settings(exe_folder)
    main["thumbnail_resolution"] = "1280x720"
    save_settings(exe_folder, main)
    JobQueue(app_paths.queue_file()).add({"prompt": "남은 작업"})

    shutil.rmtree(exe_folder)                 # PyInstaller --noconfirm replaces the whole folder
    rebuilt = _dist(tmp_path, "dist2")
    again = load_creator_settings(rebuilt)
    assert again["models_dir"] == r"D:\models 모델" and again["quality_label"] == "최고 품질"
    assert again["custom_sizes"] == {"shopify_hero": [2000, 800]}
    assert load_settings(rebuilt)["thumbnail_resolution"] == "1280x720"
    assert [j.payload["prompt"] for j in JobQueue(app_paths.queue_file()).jobs] == ["남은 작업"]
    assert not str(app_paths.settings_file()).startswith(str(tmp_path / "dist"))


def test_corrupt_settings_file_is_quarantined_not_fatal(tmp_path):
    app_paths.settings_file().write_text("{broken json", encoding="utf-8")
    settings = load_creator_settings(tmp_path)
    assert settings["quality_label"] == "일반"
    notes = app_paths.recovery_notes()
    assert notes and "손상" in notes[0]
    assert list(app_paths.data_dir().glob("settings.corrupt-*.json"))
    save_creator_settings(tmp_path, {**settings, "models_dir": "X"})
    assert json.loads(app_paths.settings_file().read_text(encoding="utf-8"))["schema_version"] == app_paths.SCHEMA_VERSION


def test_wrong_types_and_newer_schema_are_tolerated(tmp_path):
    app_paths.settings_file().write_text(json.dumps({"schema_version": 99, "studio": "oops", "future": {"x": 1}}),
                                         encoding="utf-8")
    assert load_creator_settings(tmp_path)["memory_label"] == "작업 중 PC 우선"
    save_creator_settings(tmp_path, {"models_dir": "Y"})
    data = json.loads(app_paths.settings_file().read_text(encoding="utf-8"))
    assert data["future"] == {"x": 1} and data["studio"]["models_dir"] == "Y"


def test_unicode_data_dir_and_paths(tmp_path, monkeypatch):
    monkeypatch.setenv("COVERMORPH_DATA_DIR", str(tmp_path / "사용자 データ"))
    save_creator_settings(tmp_path, {"output_dir": str(tmp_path / "출력 出力"), "models_dir": "모델"})
    assert load_creator_settings(tmp_path)["models_dir"] == "모델"
    assert resolve_output_dir(tmp_path) == tmp_path / "출력 出力"
    assert (tmp_path / "사용자 データ" / "settings.json").exists()


def test_migrates_old_dist_local_settings_and_queue_once(tmp_path):
    old = _dist(tmp_path, "old_build")
    (old / "config").mkdir()
    (old / "config" / "creator_settings.json").write_text(json.dumps({"models_dir": r"D:\old models"}), encoding="utf-8")
    (old / "config" / "settings.json").write_text(json.dumps({"thumbnail_resolution": "1280x720"}), encoding="utf-8")
    (old / "creator_output").mkdir()
    JobQueue(old / "creator_output" / "studio_queue.json").add({"prompt": "old job"})

    assert load_creator_settings(old)["models_dir"] == r"D:\old models"
    assert load_settings(old)["thumbnail_resolution"] == "1280x720"
    assert [j.payload["prompt"] for j in JobQueue(app_paths.queue_file()).jobs] == ["old job"]
    assert (old / "config" / "creator_settings.json").exists()      # source left in place
    notes = app_paths.recovery_notes()
    assert any("옮겼습니다" in note for note in notes)

    save_creator_settings(old, {"models_dir": r"D:\new models"})     # a later save is not overwritten by migration
    app_paths.migrate_from_app_folder(old)
    assert load_creator_settings(old)["models_dir"] == r"D:\new models"


def test_default_output_is_outside_the_exe_folder(tmp_path):
    output = resolve_output_dir(tmp_path, {"output_dir": ""})
    assert output.name == "CoverMorphStudio" and tmp_path not in output.parents


def test_corrupt_queue_file_is_quarantined(tmp_path):
    app_paths.queue_file().write_text("{not json", encoding="utf-8")
    queue = JobQueue(app_paths.queue_file())
    assert queue.jobs == []
    assert list(app_paths.data_dir().glob("studio_queue.corrupt-*.json"))
    queue.add({"prompt": "new"})
    assert len(JobQueue(app_paths.queue_file()).jobs) == 1


def test_sanitize_path_hides_user_name(monkeypatch, tmp_path):
    monkeypatch.setenv("LOCALAPPDATA", str(Path.home() / "AppData" / "Local"))
    text = app_paths.sanitize_path(str(Path.home() / "AppData" / "Local" / "CoverMorphStudio" / "settings.json"))
    assert text.startswith("%LOCALAPPDATA%") and Path.home().name not in text
    assert app_paths.sanitize_path(str(Path.home() / "Documents")) == "%USERPROFILE%\\Documents"

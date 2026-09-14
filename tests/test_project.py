from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from PIL import Image

from covermorph.project import (
    INPUT_TYPE_EXISTING_COVER,
    INPUT_TYPE_TEXTLESS,
    CandidateSettings,
    ChannelGenerationPreset,
    InputRecord,
    ProjectAssetError,
    ProjectLoadError,
    SceneCard,
    add_candidate_from_file,
    add_input_records,
    add_person,
    add_person_reference,
    adopt_removal_preview,
    configure_scene_prompt,
    create_project,
    load_project,
    parse_input_file,
    resolve_project_path,
    save_project_atomic,
    save_removal_preview,
    validate_project_assets,
)


def make_image(path: Path, color: tuple[int, int, int], size: tuple[int, int] = (96, 80)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path, "PNG")
    return path


def pixel(path: Path) -> tuple[int, int, int]:
    with Image.open(path) as opened:
        return opened.convert("RGB").getpixel((0, 0))


def test_project_save_load_restores_candidates_selection_and_settings(tmp_path: Path) -> None:
    project = create_project(tmp_path / "한글 프로젝트 공백", "가사 시리즈")
    project.channel_name = "채널 A"
    project.series_name = "밤 산책"
    project.lyric_mood_text = "몽환적이고 따뜻한 이별감"
    project.song_count = 3
    source = make_image(tmp_path / "입력 이미지" / "same.png", (20, 40, 60))
    settings = CandidateSettings(
        out_square=False,
        out_thumb=True,
        out_shorts=False,
        manual_boxes=((1, 2, 30, 40),),
        ocr_boxes=((4, 5, 20, 25),),
        extension_mode="natural",
        subject_offset_x=0.12,
        subject_offset_y=-0.08,
        subject_scale=1.2,
    )

    candidate = add_candidate_from_file(project, source, INPUT_TYPE_TEXTLESS, settings)
    candidate.selected = False
    project.selected_candidate_ids = []
    save_project_atomic(project)

    loaded = load_project(project.project_file)
    restored = loaded.candidates[0]

    assert loaded.schema_version == 3
    assert loaded.project_id == project.project_id
    assert loaded.channel_name == "채널 A"
    assert loaded.series_name == "밤 산책"
    assert loaded.lyric_mood_text == "몽환적이고 따뜻한 이별감"
    assert loaded.song_count == 3
    assert restored.candidate_id == candidate.candidate_id
    assert restored.selected is False
    assert restored.settings.out_square is False
    assert restored.settings.out_thumb is True
    assert restored.settings.out_shorts is False
    assert restored.settings.manual_boxes == ((1, 2, 30, 40),)
    assert restored.settings.ocr_boxes == ((4, 5, 20, 25),)
    assert not Path(restored.original_path).is_absolute()
    assert not Path(restored.working_source_path).is_absolute()
    assert resolve_project_path(loaded, restored.original_path).is_file()
    assert resolve_project_path(loaded, restored.working_source_path).is_file()


def test_project_folder_can_move_and_relative_images_still_load(tmp_path: Path) -> None:
    original_dir = tmp_path / "원래 프로젝트"
    moved_dir = tmp_path / "옮긴 프로젝트 공백"
    project = create_project(original_dir, "move test")
    source = make_image(tmp_path / "source" / "cover.png", (90, 20, 10))
    add_candidate_from_file(project, source, INPUT_TYPE_TEXTLESS)
    save_project_atomic(project)

    shutil.move(str(original_dir), str(moved_dir))
    loaded = load_project(moved_dir)
    candidate = loaded.candidates[0]

    assert pixel(resolve_project_path(loaded, candidate.original_path)) == (90, 20, 10)
    assert pixel(resolve_project_path(loaded, candidate.working_source_path)) == (90, 20, 10)


def test_same_filename_images_do_not_collide_and_inputs_are_preserved(tmp_path: Path) -> None:
    project = create_project(tmp_path / "project", "collision")
    first = make_image(tmp_path / "a" / "same.png", (255, 0, 0))
    second = make_image(tmp_path / "b" / "same.png", (0, 255, 0))
    first_bytes = first.read_bytes()
    second_bytes = second.read_bytes()

    cand1 = add_candidate_from_file(project, first, INPUT_TYPE_TEXTLESS)
    cand2 = add_candidate_from_file(project, second, INPUT_TYPE_TEXTLESS)

    assert cand1.candidate_id != cand2.candidate_id
    assert cand1.original_path != cand2.original_path
    assert pixel(resolve_project_path(project, cand1.original_path)) == (255, 0, 0)
    assert pixel(resolve_project_path(project, cand2.original_path)) == (0, 255, 0)
    assert first.read_bytes() == first_bytes
    assert second.read_bytes() == second_bytes


def test_existing_cover_preview_is_not_adopted_until_user_approval(tmp_path: Path) -> None:
    project = create_project(tmp_path / "project", "cover")
    source = make_image(tmp_path / "cover.png", (10, 20, 30))
    candidate = add_candidate_from_file(project, source, INPUT_TYPE_EXISTING_COVER)

    assert candidate.working_source_path == ""
    preview = Image.new("RGB", (96, 80), (30, 40, 50))
    quality = save_removal_preview(project, candidate, preview, "Fake LaMa", expected_size=(96, 80))

    assert quality.startswith("basic_passed")
    assert candidate.removal_preview_path
    assert candidate.working_source_path == ""
    assert candidate.working_source_approved is False

    adopted = adopt_removal_preview(project, candidate)

    assert adopted.suffix == ".png"
    assert candidate.working_source_approved is True
    assert candidate.user_approval_status == "approved_by_user"
    assert pixel(adopted) == (30, 40, 50)


def test_failed_removal_preview_cannot_be_auto_adopted(tmp_path: Path) -> None:
    project = create_project(tmp_path / "project", "cover")
    source = make_image(tmp_path / "cover.png", (10, 20, 30), size=(96, 80))
    candidate = add_candidate_from_file(project, source, INPUT_TYPE_EXISTING_COVER)
    bad_preview = Image.new("RGB", (32, 32), (255, 255, 255))

    quality = save_removal_preview(project, candidate, bad_preview, "Fake LaMa", expected_size=(96, 80))

    assert quality == "failed_size_changed"
    with pytest.raises(ProjectAssetError):
        adopt_removal_preview(project, candidate)
    assert candidate.working_source_path == ""
    assert candidate.working_source_approved is False


def test_corrupt_project_and_missing_internal_image_are_reported(tmp_path: Path) -> None:
    broken = tmp_path / "broken.json"
    broken.write_text("{broken json", encoding="utf-8")
    with pytest.raises(ProjectLoadError):
        load_project(broken)

    project = create_project(tmp_path / "project", "missing")
    source = make_image(tmp_path / "cover.png", (1, 2, 3))
    candidate = add_candidate_from_file(project, source, INPUT_TYPE_TEXTLESS)
    resolve_project_path(project, candidate.original_path).unlink()

    issues = validate_project_assets(project)

    assert issues
    assert issues[0].kind == "original"


def test_atomic_save_writes_valid_json(tmp_path: Path) -> None:
    project = create_project(tmp_path / "project", "atomic")
    add_candidate_from_file(project, make_image(tmp_path / "cover.png", (7, 8, 9)), INPUT_TYPE_TEXTLESS)

    save_project_atomic(project)
    data = json.loads(project.project_file.read_text(encoding="utf-8"))

    assert data["schema_version"] == 3
    assert data["candidates"][0]["working_source_approved"] is True


def test_schema_v1_project_migrates_without_candidate_loss(tmp_path: Path) -> None:
    source = make_image(tmp_path / "project" / "assets" / "originals" / "old.png", (4, 5, 6))
    working = make_image(tmp_path / "project" / "assets" / "textless" / "old.png", (4, 5, 6))
    project_file = tmp_path / "project" / "covermorph_project.json"
    project_file.write_text(json.dumps({"schema_version": 1, "project_id": "old", "name": "old", "candidates": [{"candidate_id": "c1", "original_path": "assets/originals/old.png", "working_source_path": "assets/textless/old.png", "input_type": "textless"}]}), encoding="utf-8")
    loaded = load_project(project_file)
    assert loaded.schema_version == 3
    assert loaded.candidates[0].candidate_id == "c1"
    save_project_atomic(loaded)
    assert json.loads(project_file.read_text(encoding="utf-8"))["schema_version"] == 3
    assert source.exists() and working.exists()


def test_people_references_scenes_and_prompt_snapshot_round_trip(tmp_path: Path) -> None:
    project = create_project(tmp_path / "project", "prompt")
    person = add_person(project, "Shared Person")
    reference = add_person_reference(project, person, make_image(tmp_path / "ref.png", (10, 20, 30)), "person", "face")
    preset = ChannelGenerationPreset("p1", "Tokyo draft", version=3, master_prompt="cinematic", negative_prompt="blur")
    project.channel_preset_id = preset.preset_id
    project.channel_preset = preset.to_dict()
    scene = SceneCard("s1", person_ids=[person.person_id], reference_image_ids=[reference.image_id], user_description="밤의 카페 東京 café")
    configure_scene_prompt(project, scene, preset)
    scene.prompt_confirmed = True
    project.scenes.append(scene)
    save_project_atomic(project)
    moved = tmp_path / "moved"
    shutil.move(str(project.project_dir), str(moved))
    loaded = load_project(moved)
    assert loaded.people[0].reference_images[0].path.startswith("assets/")
    assert resolve_project_path(loaded, loaded.people[0].reference_images[0].path).is_file()
    assert loaded.scenes[0].person_ids == [person.person_id]
    assert "東京" in loaded.scenes[0].prompt_user
    assert loaded.scenes[0].prompt_confirmed is True


def test_input_records_are_copied_inside_project(tmp_path: Path) -> None:
    project = create_project(tmp_path / "project", "inputs")
    source = tmp_path / "외부" / "가사.json"
    source.parent.mkdir()
    source.write_text(json.dumps({"title": "東京", "lyrics": "café 프랑스"}, ensure_ascii=False), encoding="utf-8")
    records = parse_input_file(source)
    add_input_records(project, records)
    save_project_atomic(project)
    loaded = load_project(project.project_file)
    assert not Path(loaded.inputs[0].source_path).is_absolute()
    assert resolve_project_path(loaded, loaded.inputs[0].source_path).is_file()
    assert source.read_text(encoding="utf-8").find("café") >= 0


def test_scene_prompt_does_not_force_lyrics_or_cover_text_and_refresh_resets_confirmation(tmp_path: Path) -> None:
    project = create_project(tmp_path / "project", "prompt")
    project.inputs.append(InputRecord("i1", "lyrics", lyrics="가사 전체는 프롬프트에 자동 삽입하지 않음", theme_mood="French café"))
    first = ChannelGenerationPreset("p1", "first", master_prompt="base", negative_prompt="bad")
    second = ChannelGenerationPreset("p2", "second", version=2, master_prompt="changed")
    scene = SceneCard("s1", user_description="사용자가 쓴 장면")
    configure_scene_prompt(project, scene, first)
    scene.prompt_confirmed = True
    configure_scene_prompt(project, scene, second)
    assert scene.prompt_source_preset_id == "p1"
    configure_scene_prompt(project, scene, second, refresh=True)
    assert scene.prompt_source_preset_id == "p2"
    assert scene.prompt_confirmed is False
    assert "가사 전체" not in scene.prompt_user


def test_utf8_bom_input_and_unknown_json_are_reported(tmp_path: Path) -> None:
    txt = tmp_path / "lyrics.txt"
    txt.write_text("日本語 café 프랑스", encoding="utf-8-sig")
    assert parse_input_file(txt)[0].lyrics == "日本語 café 프랑스"
    unknown = tmp_path / "unknown.json"
    unknown.write_text(json.dumps({"mystery": 1}), encoding="utf-8")
    with pytest.raises(ProjectLoadError, match="Unknown JSON"):
        parse_input_file(unknown)

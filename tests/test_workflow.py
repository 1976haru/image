from __future__ import annotations

import json
from pathlib import Path

import pytest

from covermorph.project import (
    ImagePlanningBrief,
    ProjectError,
    add_input_records,
    create_project,
    load_project,
    parse_input_file_detailed,
    rule_based_image_planning,
    save_project_atomic,
    validate_candidate_count,
)


def test_json_bom_fields_and_album_selection_are_explicit(tmp_path: Path) -> None:
    source = tmp_path / "songs.json"
    source.write_text(json.dumps({"songs": [{"title": "첫눈", "lyrics": "가사", "music_prompt": "slow piano", "series_description": "quiet cafe"}, {"title": "두번째", "lyrics": "lyrics 2"}], "unknown": "preserved outside records"}, ensure_ascii=False), encoding="utf-8-sig")
    parsed = parse_input_file_detailed(source)
    assert parsed["needs_mapping"] is False
    assert [record.title for record in parsed["records"]] == ["첫눈", "두번째"]
    assert parsed["records"][0].music_prompt == "slow piano"
    project = create_project(tmp_path / "project", "album")
    records = parsed["records"]
    records[1].selected = False
    add_input_records(project, records)
    project.input_selection_scope = "album"
    project.selected_input_ids = [records[0].input_id]
    project.image_planning_brief = rule_based_image_planning(records, "첫눈")
    project.cover_text = {"album_title": "첫눈"}
    save_project_atomic(project)
    loaded = load_project(project.project_file)
    assert loaded.input_selection_scope == "album"
    assert loaded.selected_input_ids == [records[0].input_id]
    assert loaded.cover_text["album_title"] == "첫눈"
    assert loaded.image_planning_brief.confirmed is False


def test_unknown_json_and_bad_candidate_count_are_not_guessed(tmp_path: Path) -> None:
    source = tmp_path / "unknown.json"
    source.write_text(json.dumps({"mystery": 1}), encoding="utf-8")
    parsed = parse_input_file_detailed(source)
    assert parsed["needs_mapping"] is True
    mapped = parse_input_file_detailed(source, {"title": "mystery"})
    assert mapped["records"][0].title == "1"
    with pytest.raises(ProjectError):
        validate_candidate_count(3)
    assert validate_candidate_count(4) == 4


def test_image_planning_brief_is_reviewable_and_round_trips() -> None:
    brief = ImagePlanningBrief(core_subject="첫눈", extraction_method="rule_based_draft_user_confirmation_required")
    restored = ImagePlanningBrief.from_dict(brief.to_dict())
    assert restored.core_subject == "첫눈"
    assert restored.confirmed is False
    assert "rule_based" in restored.extraction_method

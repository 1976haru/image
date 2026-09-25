from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from covermorph.cover_studio import (
    CANVAS_SIZE,
    CoverEdit,
    CoverRenderError,
    TextLayer,
    apply_template,
    candidate_edit,
    export_cover,
    font_supports_text,
    render_cover,
    save_candidate_edit,
    system_font_paths,
)
from covermorph.project import (
    SceneCard,
    add_generated_candidate,
    create_project,
    load_project,
    parse_input_file_detailed,
)


def supporting_font(text: str) -> Path:
    return next(path for path in system_font_paths() if font_supports_text(path, text)[0])


def generated_candidate(tmp_path: Path):
    project = create_project(tmp_path / "project", "quick")
    source = tmp_path / "generated.png"
    Image.new("RGB", (1024, 1024), "#345678").save(source)
    scene = SceneCard("scene", prompt_confirmed=True)
    candidate = add_generated_candidate(project, scene, source, {"seed": 42})
    return project, candidate


def multilingual_edit() -> CoverEdit:
    text = "한글 日本語 English"
    font = str(supporting_font(text))
    return CoverEdit(
        title=TextLayer("한글 제목", font_path=font, size=118, y=0.14),
        subtitle=TextLayer("日本語サブタイトル", font_path=font, size=58, y=0.31),
        label=TextLayer("English Label", font_path=font, size=42, y=0.91),
    )


def test_multilingual_render_is_exact_rgb_square_and_uses_same_renderer(tmp_path: Path) -> None:
    background = Image.new("RGB", (1024, 1024), "#345678")
    rendered, warnings = render_cover(background, multilingual_edit())
    assert rendered.size == (CANVAS_SIZE, CANVAS_SIZE)
    assert rendered.mode == "RGB"
    assert warnings == []
    assert rendered.getbbox() == (0, 0, CANVAS_SIZE, CANVAS_SIZE)


def test_export_preserves_original_and_records_plain_resize_metadata(tmp_path: Path) -> None:
    project, candidate = generated_candidate(tmp_path)
    source = project.project_dir / candidate.original_path
    before = source.read_bytes()
    paths = export_cover(project, candidate, multilingual_edit(), tmp_path / "output", "album")
    assert source.read_bytes() == before
    with Image.open(paths["textless_png"]) as original:
        assert original.size == (1024, 1024) and original.format == "PNG"
    with Image.open(paths["cover_jpg"]) as cover:
        assert cover.size == (1400, 1400) and cover.mode == "RGB" and cover.format == "JPEG"
    metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
    assert metadata["candidate_id"] == candidate.candidate_id
    assert metadata["source_size"] == [1024, 1024]
    assert "AI upscaling" in metadata["resize_method"]
    second = export_cover(project, candidate, multilingual_edit(), tmp_path / "output", "album")
    assert second["cover_jpg"] != paths["cover_jpg"]
    assert paths["cover_jpg"].is_file()


def test_candidate_specific_edits_survive_project_reload(tmp_path: Path) -> None:
    project, first = generated_candidate(tmp_path)
    other_source = tmp_path / "other.png"
    Image.new("RGB", (1024, 1024), "#654321").save(other_source)
    second = add_generated_candidate(
        project, SceneCard("other", prompt_confirmed=True), other_source, {"seed": 43}
    )
    first_edit = multilingual_edit()
    second_edit = multilingual_edit()
    first_edit.title.text = "첫 번째"
    second_edit.title.text = "두 번째"
    save_candidate_edit(project, first, first_edit)
    save_candidate_edit(project, second, second_edit)
    loaded = load_project(project.project_file)
    assert candidate_edit(loaded.candidates[0]).title.text == "첫 번째"
    assert candidate_edit(loaded.candidates[1]).title.text == "두 번째"
    assert loaded.selected_candidate_ids == [second.candidate_id]


def test_templates_are_bounded_and_distinct() -> None:
    edit = multilingual_edit()
    positions = set()
    for template in ("상단 제목", "중앙 제목", "하단 제목"):
        apply_template(edit, template)
        positions.add((edit.title.y, edit.subtitle.y, edit.label.y))
        render_cover(Image.new("RGB", (1024, 1024)), edit)
    assert len(positions) == 3


def test_overlap_and_unfit_text_are_reported_not_silently_cut() -> None:
    edit = multilingual_edit()
    edit.subtitle.y = edit.title.y
    with pytest.raises(CoverRenderError, match="겹칩니다"):
        render_cover(Image.new("RGB", (1024, 1024)), edit)
    edit = multilingual_edit()
    edit.title.text = "긴문구" * 1000
    with pytest.raises(CoverRenderError, match="수정해 주세요"):
        render_cover(Image.new("RGB", (1024, 1024)), edit)


def test_album_title_is_distinct_from_song_title(tmp_path: Path) -> None:
    no_album = tmp_path / "songs.json"
    no_album.write_text(
        json.dumps({"songs": [{"title": "첫 곡", "lyrics": "가사"}]}, ensure_ascii=False),
        encoding="utf-8",
    )
    parsed = parse_input_file_detailed(no_album)
    assert parsed["records"][0].title == "첫 곡"
    assert parsed["cover_text"]["title"] == ""

    album = tmp_path / "album.json"
    album.write_text(
        json.dumps(
            {"album_title": "앨범 제목", "songs": [{"title": "첫 곡", "lyrics": "가사"}]},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    assert parse_input_file_detailed(album)["cover_text"]["title"] == "앨범 제목"

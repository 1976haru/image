from __future__ import annotations

import json
from pathlib import Path

import pytest

from covermorph.planning import (
    PlanningError,
    generate_plans,
    make_input_snapshot,
    replace_one_plan,
)
from covermorph.project import (
    InputRecord,
    create_project,
    default_generation_presets,
    load_project,
    save_project_atomic,
)


class FakeBackend:
    def __init__(self, payload: dict) -> None:
        self.payload = payload
        self.prompt = ""

    def generate(self, prompt: str, schema: dict, cancel=None) -> dict:
        self.prompt = prompt
        return self.payload


class LongAlbumBackend(FakeBackend):
    def __init__(self, final_payload: dict) -> None:
        super().__init__(final_payload)
        self.calls: list[str] = []

    def generate(self, prompt: str, schema: dict, cancel=None) -> dict:
        self.calls.append(prompt)
        if "다음 한 곡" in prompt:
            return {"summary": "요약", "facts": "사실", "emotion_flow": "흐름", "motifs": "상징", "viewpoint": "시점", "season_place_actions": "장소"}
        self.prompt = prompt
        return self.payload


def plan(index: int, source_id: str, brightness: str, title: dict | None = None) -> dict:
    return {
        "name_ko": f"기획 {index}",
        "scene_ko": f"서로 다른 장면 {index}",
        "connection_reason": "가사에 나온 비와 창문을 사용하고, 카페 장면은 새 제안이다.",
        "related_input_ids": [source_id],
        "characters_action": "주인공이 창밖을 본다",
        "place_time_weather_season": "카페, 저녁, 비, 여름",
        "background_props": f"소품 {index}",
        "composition_distance": f"구도 {index}",
        "brightness_color": brightness,
        "text_safe_area": "왼쪽 위",
        "image_prompt_en": f"cinematic scene {index}, no text",
        "negative_prompt": "text, logo, watermark",
        "title": title or {"main": "비의 기억", "subtitle": "", "label": "Tokyo ChillRap", "series": "003", "language": "일본어", "source_type": "proposed", "source_input_id": "", "source_text": ""},
    }


def payload(source_id: str, count: int = 4) -> dict:
    return {
        "interpretation": {"album_theme": "관계", "emotion_flow": "그리움", "motifs_symbols": "비", "viewpoint_protagonist": "여자", "season_place_actions": "여름 카페", "channel_visualization": "도시 실사", "conflicts": "없음"},
        "processed_inputs": [{"input_id": source_id, "status": "ok", "summary": "비 오는 날의 이별"}],
        "plans": [plan(i, source_id, "밝은 장면" if i == 1 else "중간 밝기" if i == 2 else "어두운 장면") for i in range(1, count + 1)],
    }


def snapshot(count: int = 4):
    record = InputRecord("song_1", "lyrics", "비", lyrics="창문에 비가 내려")
    return make_input_snapshot(default_generation_presets()[4], [record], "여자 이야기 003, 4장", {"label": "Tokyo ChillRap"}, count)


@pytest.mark.parametrize("count", [4, 5])
def test_real_backend_contract_count_priority_and_prompt_separation(count: int) -> None:
    snap = snapshot(count)
    backend = FakeBackend(payload("song_1", count))
    result = generate_plans(snap, backend)
    assert len(result["plans"]) == count
    assert result["status"] == "complete"
    assert '"candidate_count": ' + str(count) in backend.prompt
    assert "BPM/악기" in backend.prompt
    assert all(item["title"]["main"] not in item["image_prompt_en"] for item in result["plans"])


def test_selected_songs_only_and_all_selected_are_accounted_for() -> None:
    records = [InputRecord("one", "lyrics", "one", lyrics="front"), InputRecord("two", "lyrics", "two", lyrics="後ろの歌詞"), InputRecord("skip", "lyrics", "skip", lyrics="omit", selected=False)]
    snap = make_input_snapshot(default_generation_presets()[0], records, "", {}, 4)
    data = payload("one")
    data["processed_inputs"].append({"input_id": "two", "status": "failed", "summary": "실패"})
    result = generate_plans(snap, FakeBackend(data))
    assert {item["input_id"] for item in snap.records} == {"one", "two"}
    assert result["status"] == "partial"
    assert result["missing_input_ids"] == ["two"]


def test_lyric_excerpt_must_match_exact_source() -> None:
    data = payload("song_1")
    data["plans"][0]["title"] = {"main": "없는 문장", "subtitle": "", "label": "x", "series": "", "language": "ko", "source_type": "lyric_excerpt", "source_input_id": "song_1", "source_text": "없는 문장"}
    with pytest.raises(PlanningError, match="원문 발췌"):
        generate_plans(snapshot(), FakeBackend(data), retries=0)


def test_format_failure_is_bounded_and_old_cards_can_be_preserved() -> None:
    backend = FakeBackend({"plans": []})
    with pytest.raises(PlanningError, match="재시도"):
        generate_plans(snapshot(), backend, retries=1)
    existing = payload("song_1")
    for index, item in enumerate(existing["plans"]):
        item["plan_id"] = f"stable-{index}"
        item["user_edited"] = index == 1
    replacement = payload("song_1")
    updated = replace_one_plan(existing, replacement, "stable-0")
    assert updated["plans"][0]["plan_id"] == "stable-0"
    assert updated["plans"][1] == existing["plans"][1]


def test_planning_round_trip_and_old_project_compatibility(tmp_path: Path) -> None:
    project = create_project(tmp_path / "p", "p")
    project.cover_planning = {"request_text": "요청", "requested_count": 4, "result": payload("song_1")}
    save_project_atomic(project)
    loaded = load_project(project.project_file)
    assert loaded.cover_planning["request_text"] == "요청"
    old = project.to_dict()
    old["schema_version"] = 3
    old.pop("cover_planning")
    project.project_file.write_text(json.dumps(old, ensure_ascii=False), encoding="utf-8")
    assert load_project(project.project_file).cover_planning == {}


def test_long_album_summarizes_every_selected_song_before_reduce() -> None:
    records = [InputRecord(f"song_{i}", "lyrics", str(i), lyrics=("가사" * 10_000)) for i in range(3)]
    snap = make_input_snapshot(default_generation_presets()[0], records, "앨범", {}, 4)
    data = payload("song_0")
    data["processed_inputs"] = [{"input_id": f"song_{i}", "status": "ok", "summary": "완료"} for i in range(3)]
    for item in data["plans"]:
        item["related_input_ids"] = ["song_0", "song_1", "song_2"]
    backend = LongAlbumBackend(data)
    result = generate_plans(snap, backend)
    assert result["status"] == "complete"
    assert sum("다음 한 곡" in call for call in backend.calls) == 3
    assert "[곡별 LLM 요약]" in backend.prompt

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
        self.plan_call = 0

    def generate(self, prompt: str, schema: dict, cancel=None) -> dict:
        self.prompt = prompt
        outlines_schema = (schema.get("properties") or {}).get("outlines") or {}
        if outlines_schema:
            count = outlines_schema.get("maxItems", 4)
            source_id = self.payload.get("processed_inputs", [{"input_id": "song_1"}])[0]["input_id"]
            return {"outlines": [
                {"related_input_ids": [source_id], "lyric_grounding": f"가사 근거 {i}", "emotion": "그리움", "action": f"행동 {i}", "place": f"장소 {i}", "composition": f"구도 {i}", "props": f"소품 {i}", "lighting": f"조명 {i}"}
                for i in range(count)
            ]}
        plans_schema = (schema.get("properties") or {}).get("plans") or {}
        if plans_schema.get("maxItems") == 1 and self.payload.get("plans"):
            item = self.payload["plans"][self.plan_call % len(self.payload["plans"])]
            self.plan_call += 1
            return {"plans": [json.loads(json.dumps(item))]}
        return self.payload


class LongAlbumBackend(FakeBackend):
    def __init__(self, final_payload: dict) -> None:
        super().__init__(final_payload)
        self.calls: list[str] = []

    def generate(self, prompt: str, schema: dict, cancel=None) -> dict:
        self.calls.append(prompt)
        if "다음 한 곡" in prompt:
            return {"summary": "요약", "facts": "사실", "emotion_flow": "흐름", "motifs": "상징", "viewpoint": "시점", "season_place_actions": "장소"}
        return super().generate(prompt, schema, cancel)


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
        "image_prompt_en": f"cinematic scene {index}, no text, " + ("bright daylight" if index == 1 else "balanced medium light" if index == 2 else "dark ambient light"),
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

class CountingBackend(FakeBackend):
    def __init__(self, payload: dict) -> None:
        super().__init__(payload)
        self.count = 0

    def generate(self, prompt: str, schema: dict, cancel=None) -> dict:
        self.count += 1
        return super().generate(prompt, schema, cancel)


def test_user_title_and_fixed_metadata_are_program_owned() -> None:
    snap = snapshot()
    snap.explicit_settings.update({"album_title": "사용자 제목", "label": "고정 채널", "series": "EP 07", "cover_text_language": "한국어"})
    data = payload("song_1")
    data["plans"][0]["title"] = {"main": "가짜 발췌", "subtitle": "", "label": "변경", "series": "변경", "language": "중국어", "source_type": "lyric_excerpt", "source_input_id": "song_1", "source_text": "없는 문장"}
    result = generate_plans(snap, FakeBackend(data), retries=0)
    for item in result["plans"]:
        assert item["title"]["main"] == "사용자 제목"
        assert item["title"]["label"] == "고정 채널"
        assert item["title"]["series"] == "EP 07"
        assert item["title"]["language"] == "한국어"
        assert item["title"]["source_type"] == "user"
        assert item["title"]["source_start"] == -1


def test_title_source_validation_and_excerpt_offsets() -> None:
    proposed = payload("song_1")
    proposed["plans"][0]["title"].update({"main": "가사에 없는 AI 제목", "source_type": "proposed", "source_text": "", "source_input_id": ""})
    assert generate_plans(snapshot(), FakeBackend(proposed), retries=0)["plans"][0]["title"]["main"] == "가사에 없는 AI 제목"
    excerpt = payload("song_1")
    excerpt["plans"][0]["title"].update({"main": "창문에 비가 내려", "source_type": "lyric_excerpt", "source_text": "창문에 비가 내려", "source_input_id": "song_1"})
    title = generate_plans(snapshot(), FakeBackend(excerpt), retries=0)["plans"][0]["title"]
    assert title["source_start"] == 0 and title["source_end"] == len("창문에 비가 내려")


def test_language_roles_and_title_prompt_separation() -> None:
    snap = snapshot()
    snap.explicit_settings["album_title"] = "雨の距離"
    result = generate_plans(snap, FakeBackend(payload("song_1")), retries=0)
    assert result["quality_passed"] is True
    assert all("雨の距離" not in item["image_prompt_en"] for item in result["plans"])
    assert all(any("가" <= char <= "힣" for char in item["scene_ko"]) for item in result["plans"])
    assert all(any("a" <= char.lower() <= "z" for char in item["image_prompt_en"]) for item in result["plans"])


def test_structural_failure_retry_is_bounded() -> None:
    backend = CountingBackend({"plans": []})
    with pytest.raises(PlanningError, match="재시도 상한"):
        generate_plans(snapshot(), backend, retries=99)
    assert backend.count == 2


def test_repeated_candidate_only_is_replanned_and_passed_cards_keep_ids() -> None:
    first = payload("song_1")
    for i, item in enumerate(first["plans"]):
        item["plan_id"] = f"keep-{i}"
    first["plans"][3] = dict(first["plans"][2])
    first["plans"][3]["plan_id"] = "replace-me"
    replacement = payload("song_1", 1)
    replacement["plans"][0].update({"characters_action": "주인공이 우산을 접고 계단을 오른다", "place_time_weather_season": "역 출구, 새벽, 비", "background_props": "젖은 우산과 계단 난간", "composition_distance": "높은 각도의 전신 원경", "brightness_color": "차분한 새벽빛", "image_prompt_en": "wide high-angle station exit at dawn, woman folding an umbrella, cool ambient light, no text"})
    class SequenceBackend:
        def __init__(self): self.calls = 0
        def generate(self, prompt, schema, cancel=None):
            self.calls += 1
            if self.calls == 1:
                return {"interpretation": first["interpretation"], "processed_inputs": first["processed_inputs"]}
            if self.calls == 2:
                return {"outlines": [
                    {"related_input_ids": ["song_1"], "lyric_grounding": f"근거 {i}", "emotion": "그리움", "action": f"행동 {i}", "place": f"장소 {i}", "composition": f"구도 {i}", "props": f"소품 {i}", "lighting": f"조명 {i}"}
                    for i in range(4)
                ]}
            if 3 <= self.calls <= 6:
                return {"plans": [first["plans"][self.calls - 3]]}
            return replacement
    result = generate_plans(snapshot(), SequenceBackend(), retries=1)
    assert result["plans"][0]["plan_id"] == "keep-0"
    assert result["plans"][1]["plan_id"] == "keep-1"
    assert result["plans"][2]["plan_id"] == "keep-2"
    assert result["plans"][3]["plan_id"] != "replace-me"


def test_truncated_server_response_is_rejected(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from covermorph.planning import LlamaCppCliBackend
    backend = LlamaCppCliBackend(tmp_path)
    monkeypatch.setattr(backend, "_ensure_server", lambda cancel=None: None)
    replies = iter(({"prompt": "formatted"}, {"content": "{\"x\":1", "truncated": True, "stop_type": "limit"}))
    monkeypatch.setattr(backend, "_post", lambda *args, **kwargs: next(replies))
    with pytest.raises(PlanningError, match="잘렸"):
        backend.generate("prompt", {"type": "object"})

def test_audience_is_not_silently_promoted_to_character_and_sources_are_recorded() -> None:
    record = InputRecord("song_1", "lyrics", "눈", lyrics="첫눈이 내린다")
    snap = make_input_snapshot(default_generation_presets()[0], [record], "잔잔한 발라드", {"visual_style": "실사"}, 4)
    constraints = snap.intent_constraints
    assert constraints["channel_audience"]["value"] == "한국 시니어"
    assert constraints["channel_audience"]["visual_character_constraint"] is False
    assert not any(item["field"] == "character" for item in constraints["fixed"])
    assert constraints["visual_style"]["source"] == "사용자가 명시적으로 고정한 설정"


def test_conflicting_explicit_conditions_are_shown_instead_of_overwritten() -> None:
    record = InputRecord("song_1", "lyrics", "눈", lyrics="첫눈")
    with pytest.raises(PlanningError, match="충돌.*등장인물"):
        make_input_snapshot(default_generation_presets()[0], [record], "", {"character": "여성 1명", "characters": "남성 2명"}, 4)


def test_scene_outlines_precede_cards_and_require_two_meaningful_differences() -> None:
    result = generate_plans(snapshot(), FakeBackend(payload("song_1")), retries=0)
    assert len(result["scene_outlines"]) == 4
    assert result["attempts"] == 6
    assert result["validation_state"] == {
        "execution": "success",
        "structure_and_fixed_conditions": "pass",
        "semantic_diversity_language_review": "automatic_pass_human_review_required",
        "user_adoption": "unreviewed",
    }

    class DuplicateOutlineBackend(FakeBackend):
        def generate(self, prompt, schema, cancel=None):
            response = super().generate(prompt, schema, cancel)
            if "outlines" in response:
                response["outlines"][1] = json.loads(json.dumps(response["outlines"][0]))
            return response

    duplicate_result = generate_plans(snapshot(), DuplicateOutlineBackend(payload("song_1")), retries=0)
    assert duplicate_result["scene_outlines"][1]["outline_quality_status"] == "review"
    assert duplicate_result["quality_passed"] is False


def test_fixed_conditions_are_kept_as_structured_prompt_components() -> None:
    snap = snapshot()
    snap.explicit_settings["character"] = "20대 일본 여성 1명"
    snap.intent_constraints["fixed"].append({"field": "character", "value": "20대 일본 여성 1명", "source": "사용자가 명시적으로 고정한 설정"})
    data = payload("song_1")
    for item in data["plans"]:
        item["scene_ko"] += " 20대 일본 여성 1명"
    result = generate_plans(snap, FakeBackend(data), retries=0)
    components = result["plans"][0]["reference"]["image_prompt_components"]
    assert components["ai_scene_en"] == result["plans"][0]["image_prompt_en"]
    assert components["fixed_conditions"][0]["value"] == "20대 일본 여성 1명"
    assert components["assembly_status"] == "structured_for_stage2_not_rendered"
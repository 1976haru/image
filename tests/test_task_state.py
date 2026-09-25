from __future__ import annotations

import time
from types import SimpleNamespace

from covermorph import planning
from covermorph.quick_planning import generate_plans_for_quick_cover
from covermorph.task_state import TaskState


def test_stale_job_events_cannot_overwrite_new_job() -> None:
    state = TaskState()
    old = state.start(4, {"source": "old"})
    new = state.start(1, {"source": "new"})
    assert state.update(old, "이미지 생성", "늦은 이벤트", step=28) is False
    assert state.job_id == new
    assert state.state == "자료 읽기"
    assert state.step == 0


def test_terminal_state_stops_elapsed_timer_and_restores_interrupted_as_cancelled() -> None:
    state = TaskState()
    job = state.start(5, {"source": "snapshot"})
    state.started_at = time.monotonic() - 10
    state.finish(job, "중단됨", "안전하게 중단", failed_indices=[2, 3, 4], completed=2)
    elapsed = state.elapsed()
    time.sleep(0.02)
    assert state.elapsed() == elapsed
    assert state.request_cancel() is False

    stored = state.to_dict()
    stored["state"] = "이미지 생성"
    restored = TaskState.restore(stored)
    assert restored.state == "중단됨"
    assert restored.failed_indices == [2, 3, 4]
    assert restored.input_snapshot == {"source": "snapshot"}


def test_invalid_ai_lyric_title_is_cleared_without_hiding_image_plan_failure(monkeypatch) -> None:
    title = {
        "main": "가사에 없는 제목",
        "source_type": "lyric_excerpt",
        "source_input_id": "song",
        "source_text": "가사에 없는 제목",
    }
    records = [{"input_id": "song", "lyrics": "실제 가사"}]

    def fake_generate(_snapshot, _backend, _cancel):
        planning._validate_excerpt(title, records)
        return {"plans": [{"image_prompt_en": "valid visual prompt", "title": title}]}

    original_generate = planning.generate_plans
    original_validate = planning._validate_excerpt
    monkeypatch.setattr(planning, "generate_plans", fake_generate)
    result = generate_plans_for_quick_cover(SimpleNamespace(), object())
    assert result["plans"][0]["image_prompt_en"] == "valid visual prompt"
    assert title["main"] == ""
    assert title["validation_warning"].startswith("AI 제안 제목")
    assert planning._validate_excerpt is original_validate
    monkeypatch.setattr(planning, "generate_plans", original_generate)

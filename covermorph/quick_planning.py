from __future__ import annotations

import threading
from typing import Any

from . import planning

_VALIDATION_LOCK = threading.Lock()


def generate_plans_for_quick_cover(
    snapshot: planning.PlanningInput,
    backend: planning.PlanningBackend,
    cancel: Any = None,
) -> dict[str, Any]:
    """Keep a valid image plan when only its AI lyric-title citation is invalid."""
    original = planning._validate_excerpt

    def tolerate_invalid_ai_title(title: dict[str, Any], records: list[dict[str, Any]]) -> None:
        try:
            original(title, records)
        except planning.PlanningError:
            if title.get("source_type") != "lyric_excerpt":
                raise
            title.update(
                {
                    "main": "",
                    "source_type": "invalid_ai_suggestion_removed",
                    "source_input_id": "",
                    "source_text": "",
                    "source_start": -1,
                    "source_end": -1,
                    "validation_warning": "AI 제안 제목이 선택한 가사 원문과 일치하지 않아 비웠습니다.",
                }
            )

    with _VALIDATION_LOCK:
        planning._validate_excerpt = tolerate_invalid_ai_title
        try:
            return planning.generate_plans(snapshot, backend, cancel)
        finally:
            planning._validate_excerpt = original

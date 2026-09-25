from __future__ import annotations

import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any

TERMINAL_STATES = {"중단됨", "실패", "완료"}
ACTIVE_STATES = {"자료 읽기", "가사 해석", "모델 로딩", "이미지 생성", "저장", "중단 요청"}


@dataclass(slots=True)
class TaskState:
    job_id: str = ""
    state: str = "대기"
    description: str = "작업을 시작할 수 있습니다."
    requested: int = 0
    completed: int = 0
    current_candidate: int = 0
    step: int = 0
    steps: int = 0
    selected_songs: int = 0
    analyzed_songs: int = 0
    started_at: float = 0.0
    last_event_at: float = 0.0
    finished_at: float = 0.0
    error_summary: str = ""
    error_detail: str = ""
    log_path: str = ""
    failed_indices: list[int] = field(default_factory=list)
    input_snapshot: dict[str, Any] = field(default_factory=dict)

    @property
    def active(self) -> bool:
        return self.state in ACTIVE_STATES

    @property
    def terminal(self) -> bool:
        return self.state in TERMINAL_STATES

    def start(self, requested: int, snapshot: dict[str, Any], selected_songs: int = 0) -> str:
        now = time.monotonic()
        self.job_id = f"job_{uuid.uuid4().hex}"
        self.state = "자료 읽기"
        self.description = "입력 스냅샷을 확인하고 있습니다."
        self.requested = requested
        self.completed = 0
        self.current_candidate = 0
        self.step = 0
        self.steps = 0
        self.selected_songs = selected_songs
        self.analyzed_songs = 0
        self.started_at = now
        self.last_event_at = now
        self.finished_at = 0.0
        self.error_summary = ""
        self.error_detail = ""
        self.failed_indices = []
        self.input_snapshot = dict(snapshot)
        return self.job_id

    def update(
        self, job_id: str, state: str | None = None, description: str | None = None, **values: Any
    ) -> bool:
        if job_id != self.job_id or self.terminal:
            return False
        if state:
            self.state = state
        if description is not None:
            self.description = description
        for key, value in values.items():
            if key in self.__dataclass_fields__:
                setattr(self, key, value)
        self.last_event_at = time.monotonic()
        return True

    def finish(self, job_id: str, state: str, description: str, **values: Any) -> bool:
        if job_id != self.job_id:
            return False
        if state not in TERMINAL_STATES:
            raise ValueError(f"종료 상태가 아닙니다: {state}")
        for key, value in values.items():
            if key in self.__dataclass_fields__:
                setattr(self, key, value)
        self.state = state
        self.description = description
        self.last_event_at = time.monotonic()
        self.finished_at = self.last_event_at
        return True

    def request_cancel(self) -> bool:
        if not self.active or self.state == "중단 요청":
            return False
        previous = self.state
        self.state = "중단 요청"
        if previous == "모델 로딩":
            self.description = "모델 로딩이 끝나는 안전 지점에서 중단합니다."
        else:
            self.description = "현재 안전 지점에서 작업을 중단하고 있습니다."
        self.last_event_at = time.monotonic()
        return True

    def elapsed(self, now: float | None = None) -> int:
        if not self.started_at:
            return 0
        end = self.finished_at or (now if now is not None else time.monotonic())
        return max(0, int(end - self.started_at))

    def idle_seconds(self, now: float | None = None) -> int:
        if not self.last_event_at:
            return 0
        end = self.finished_at or (now if now is not None else time.monotonic())
        return max(0, int(end - self.last_event_at))

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        # monotonic timestamps are process-local and never restored as active.
        data.update({"started_at": 0.0, "last_event_at": 0.0, "finished_at": 0.0})
        return data

    @classmethod
    def restore(cls, data: Any) -> "TaskState":
        values = dict(data or {})
        state = str(values.get("state") or "대기")
        if state in ACTIVE_STATES:
            state = "중단됨"
            values["description"] = "앱 종료로 이전 작업이 중단되었습니다. 남은 작업을 재시작할 수 있습니다."
        values["state"] = state
        allowed = {key: values[key] for key in cls.__dataclass_fields__ if key in values}
        return cls(**allowed)

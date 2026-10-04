"""Background runner for the studio queue: one job at a time, resource-gated, pause-after-current.

The queue file is ``<output root>/studio_queue.json`` (survives app restarts; a job interrupted by a
crash returns to pending). Before each job the runner checks GPU/RAM/commit for that job's memory mode
and, if configured, keyboard/mouse idle time; when short it leaves the job pending with the reason
"PC 사용 중 — 자원 대기" and re-checks every ``recheck_seconds``. It never kills other programs.
"""
from __future__ import annotations

import ctypes
import os
import threading
import time
from pathlib import Path
from typing import Any, Callable

from .job_queue import JobQueue, QueueJob, QueueRunner
from .quality_engines import check_resources, resource_snapshot

WAITING_LABEL = "PC 사용 중 — 자원 대기"


def idle_seconds() -> float | None:
    """Seconds since the last keyboard/mouse input (Windows GetLastInputInfo); None elsewhere."""
    if os.name != "nt":
        return None

    class LASTINPUTINFO(ctypes.Structure):
        _fields_ = [("cbSize", ctypes.c_uint), ("dwTime", ctypes.c_uint)]

    info = LASTINPUTINFO()
    info.cbSize = ctypes.sizeof(info)
    if not ctypes.windll.user32.GetLastInputInfo(ctypes.byref(info)):
        return None
    return max(0.0, (ctypes.windll.kernel32.GetTickCount() - info.dwTime) / 1000.0)


class StudioRunner:
    def __init__(self, queue: JobQueue, executor: Callable[[dict[str, Any], threading.Event, Callable], dict[str, Any]],
                 settings: dict[str, Any], on_change: Callable[[], None] | None = None):
        self.queue = queue
        self.settings = settings
        self.on_change = on_change or (lambda: None)
        self.progress: dict[str, dict[str, Any]] = {}   # job id -> {"fraction", "message"} (not persisted)
        self.status = "대기열 정지"
        self._executor = executor
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._thread: threading.Thread | None = None
        self.runner = QueueRunner(queue, self._execute, self._gate, self._event)

    # -------------------------------------------------------------- control
    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        self.queue.resume()
        if not self.running:
            self._stop.clear()
            self._thread = threading.Thread(target=self._loop, name="studio-queue", daemon=True)
            self._thread.start()
        self._wake.set()

    def pause_after_current(self) -> None:
        self.queue.pause_after_current()
        self.status = "현재 작업 후 일시정지"
        self.on_change()

    def cancel_current(self) -> None:
        self.runner.cancel_current()

    def shutdown(self) -> None:
        """App closing: stop taking new jobs; the running job is cancelled (it returns to pending on restart)."""
        self._stop.set()
        self._wake.set()
        self.runner.cancel_current()

    def poke(self) -> None:
        self._wake.set()

    # -------------------------------------------------------------- internals
    def _gate(self, payload: dict[str, Any]) -> tuple[bool, list[str]]:
        policy = payload.get("memory") or "interactive_low_memory"
        ok, reasons = check_resources(policy, resource_snapshot())
        idle_needed = float(self.settings.get("require_idle_minutes") or 0) * 60
        if ok and policy == "night_best" and idle_needed > 0:
            idle = idle_seconds()
            if idle is not None and idle < idle_needed:
                ok, reasons = False, [f"자리 비움 대기: 입력 없음 {int(idle // 60)}분 / 필요 {int(idle_needed // 60)}분"]
        return ok, reasons

    def _event(self, event: str, job: QueueJob) -> None:
        if event == "waiting":
            self.status = f"{WAITING_LABEL} ({job.waiting_reason})"
        elif event == "started":
            self.status = "실행 중"
            self.progress[job.id] = {"fraction": 0.0, "message": "시작"}
        else:
            if self._stop.is_set() and event == "cancelled":  # app closing: run it again after restart
                job.state, job.started, job.finished, job.error = "pending", None, None, ""
                self.queue.save()
            self.progress.setdefault(job.id, {})["fraction"] = 1.0
        self.on_change()

    def _execute(self, payload: dict[str, Any], cancel: threading.Event) -> dict[str, Any]:
        job = self.runner.current

        def progress(event: dict[str, Any]) -> None:
            if job is None:
                return
            state = self.progress.setdefault(job.id, {"fraction": 0.0, "message": ""})
            if event.get("phase") == "stage":
                state.update(fraction=float(event.get("fraction", state["fraction"])), message=event.get("message", ""))
            elif event.get("phase") == "inference" and event.get("steps"):
                state["message"] = f"{state.get('message', '').split(' · ')[0]} · {event['step']}/{event['steps']}"
            elif event.get("phase") == "waiting":
                state["message"] = event.get("message", "")
            self.on_change()

        return self._executor(payload, cancel, progress)

    def _loop(self) -> None:
        while not self._stop.is_set():
            status = self.runner.run_next()
            if status == "ran":
                continue
            if status == "paused":
                self.status = "일시정지됨"
            elif status == "empty":
                self.status = "대기 작업 없음"
            self.on_change()
            self._wake.clear()
            self._wake.wait(float(self.settings.get("recheck_seconds") or 20) if status == "waiting" else 3600)
        self.status = "대기열 정지"
        self.on_change()


def queue_path(output_root: Path | None = None) -> Path:
    """The queue lives with the per-user settings (survives rebuilds and output-folder changes)."""
    from .app_paths import queue_file
    return queue_file()

"""Persistent job queue with pause-after-current, resume and cancel-pending.

Pause/resume is job-level: the running image always finishes, the queue stops before the next job.
No mid-denoise checkpointing. State lives in one JSON file written atomically (temp file + os.replace),
so an app restart never loses jobs; a job left "running" by a crash goes back to "pending" on load.
Before each job the runner checks free GPU/commit/RAM; if too low the job stays queued.
"""
from __future__ import annotations

import json
import os
import tempfile
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

PENDING, RUNNING, DONE, FAILED, CANCELLED = "pending", "running", "done", "failed", "cancelled"
TERMINAL = (DONE, FAILED, CANCELLED)


@dataclass
class QueueJob:
    payload: dict[str, Any]
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    state: str = PENDING
    created: float = field(default_factory=time.time)
    started: float | None = None
    finished: float | None = None
    result: dict[str, Any] | None = None
    error: str = ""
    waiting_reason: str = ""


class JobQueue:
    def __init__(self, path: Path):
        self.path = Path(path)
        self._lock = threading.RLock()
        self.jobs: list[QueueJob] = []
        self.paused = False
        self.load()

    # -------------------------------------------------------------- persistence
    def load(self) -> None:
        with self._lock:
            if not self.path.exists():
                return
            data = json.loads(self.path.read_text(encoding="utf-8"))
            self.paused = bool(data.get("paused", False))
            self.jobs = [QueueJob(**job) for job in data.get("jobs", [])]
            for job in self.jobs:
                if job.state == RUNNING:  # interrupted by a crash/exit: run it again
                    job.state, job.started = PENDING, None
            self.save()

    def save(self) -> None:
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            text = json.dumps({"version": 1, "paused": self.paused, "jobs": [asdict(j) for j in self.jobs]},
                              ensure_ascii=False, indent=2)
            fd, tmp = tempfile.mkstemp(prefix=".queue_", suffix=".json", dir=self.path.parent)
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    handle.write(text)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(tmp, self.path)
            except BaseException:
                Path(tmp).unlink(missing_ok=True)
                raise

    # -------------------------------------------------------------- operations
    def add(self, payload: dict[str, Any]) -> QueueJob:
        with self._lock:
            job = QueueJob(payload=payload)
            self.jobs.append(job)
            self.save()
            return job

    def get(self, job_id: str) -> QueueJob | None:
        return next((job for job in self.jobs if job.id == job_id), None)

    def pending(self) -> list[QueueJob]:
        return [job for job in self.jobs if job.state == PENDING]

    def cancel(self, job_id: str) -> bool:
        """Cancel a pending job. The running job is not interrupted here (see Runner.cancel_current)."""
        with self._lock:
            job = self.get(job_id)
            if job is None or job.state != PENDING:
                return False
            job.state, job.finished = CANCELLED, time.time()
            self.save()
            return True

    def cancel_pending(self) -> int:
        with self._lock:
            count = 0
            for job in self.pending():
                job.state, job.finished = CANCELLED, time.time()
                count += 1
            self.save()
            return count

    def remove(self, job_ids: list[str]) -> int:
        """Delete jobs that are not running (pending/cancelled/failed/done)."""
        with self._lock:
            before = len(self.jobs)
            self.jobs = [job for job in self.jobs if job.id not in job_ids or job.state == RUNNING]
            self.save()
            return before - len(self.jobs)

    def retry(self, job_id: str) -> bool:
        """Put a failed/cancelled job back in the queue."""
        with self._lock:
            job = self.get(job_id)
            if job is None or job.state not in (FAILED, CANCELLED):
                return False
            job.state, job.error, job.started, job.finished, job.result = PENDING, "", None, None, None
            self.save()
            return True

    def pause_after_current(self) -> None:
        with self._lock:
            self.paused = True
            self.save()

    def resume(self) -> None:
        with self._lock:
            self.paused = False
            self.save()

    def summary(self) -> dict[str, Any]:
        counts: dict[str, int] = {}
        for job in self.jobs:
            counts[job.state] = counts.get(job.state, 0) + 1
        return {"paused": self.paused, "counts": counts,
                "waiting": [{"id": j.id, "reason": j.waiting_reason} for j in self.pending() if j.waiting_reason]}


ResourceGate = Callable[[dict[str, Any]], tuple[bool, list[str]]]
Executor = Callable[[dict[str, Any], threading.Event], dict[str, Any]]


class QueueRunner:
    """Runs jobs one at a time. ``executor`` must leave no model resident when it returns in low-memory mode."""

    def __init__(self, queue: JobQueue, executor: Executor, resource_gate: ResourceGate | None = None,
                 on_event: Callable[[str, QueueJob], None] | None = None):
        self.queue = queue
        self.executor = executor
        self.resource_gate = resource_gate
        self.on_event = on_event or (lambda event, job: None)
        self.cancel_event = threading.Event()
        self.current: QueueJob | None = None

    def cancel_current(self) -> None:
        self.cancel_event.set()

    def run_next(self) -> str:
        """Run at most one job. Returns 'ran', 'paused', 'empty' or 'waiting' (resources too low)."""
        queue = self.queue
        with queue._lock:
            if queue.paused:
                return "paused"
            pending = queue.pending()
            if not pending:
                return "empty"
            job = pending[0]
            if self.resource_gate is not None:
                ok, reasons = self.resource_gate(job.payload)
                if not ok:
                    job.waiting_reason = "; ".join(reasons)
                    queue.save()
                    self.on_event("waiting", job)
                    return "waiting"
            job.state, job.started, job.waiting_reason = RUNNING, time.time(), ""
            queue.save()
        self.current = job
        self.cancel_event.clear()
        self.on_event("started", job)
        try:
            job.result = self.executor(job.payload, self.cancel_event)
            job.state = CANCELLED if self.cancel_event.is_set() else DONE
        except Exception as exc:  # recorded on the job; the queue keeps going
            job.state = CANCELLED if self.cancel_event.is_set() else FAILED
            job.error = f"{type(exc).__name__}: {exc}"
        finally:
            job.finished = time.time()
            self.current = None
            queue.save()
        self.on_event(job.state, job)
        return "ran"

    def run(self, stop: threading.Event | None = None, wait_seconds: float = 30.0,
            max_waits: int | None = None) -> str:
        """Drain the queue until empty or paused; while resources are low, wait and re-check."""
        stop = stop or threading.Event()
        waits = 0
        while not stop.is_set():
            status = self.run_next()
            if status in ("empty", "paused"):
                return status
            if status == "waiting":
                waits += 1
                if max_waits is not None and waits >= max_waits:
                    return "waiting"
                if stop.wait(wait_seconds):
                    break
            else:
                waits = 0
        return "stopped"

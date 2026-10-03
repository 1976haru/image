"""Real-GPU queue / low-memory validation for Quality Engine V2.

    python scripts/validate_quality_queue.py --models-dir D:\\models --output-dir validation_results\\quality_queue

1. queues 4 real jobs (Korean/Japanese prompts, Unicode output folder), pauses after the first,
2. checks GPU memory returned to its baseline after each job (model process exited),
3. restarts the queue from disk (new JobQueue object), cancels one pending job, resumes, drains,
4. forces an impossible resource threshold and checks the job stays queued instead of starting.
Writes report.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from threading import Event

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from covermorph.job_queue import CANCELLED, DONE, PENDING, JobQueue, QueueRunner  # noqa: E402
from covermorph.quality_engines import check_resources, gpu_used_mib, resource_snapshot  # noqa: E402
from covermorph.quality_modes import run_quality_job  # noqa: E402

JOBS = [
    ("도쿄 비 오는 밤 거리, 사람 없음", "Tokyo Chill", 0, "preview"),
    ("東京の駅のホームに立つ若い男性", "Tokyo Chill", 1, "balanced"),
    ("a mature couple in their fifties in an autumn park", "OLD POP LOUNGE", 2, "preview"),
    ("a quiet seaside town at sunset, no people", "", 0, "preview"),
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir / "큐 キュー 출력"
    out.mkdir(parents=True, exist_ok=True)
    queue_path = out / "queue.json"
    queue_path.unlink(missing_ok=True)
    report: dict = {"baseline_gpu_mib": gpu_used_mib(), "jobs": [], "checks": {}}
    after_job_gpu: list[int | None] = []

    def executor(payload: dict, cancel: Event) -> dict:
        result = run_quality_job({**payload, "models_dir": str(args.models_dir), "max_candidates_per_engine": 1},
                                 cancel)
        time.sleep(1.0)  # let the driver report the freed memory
        after_job_gpu.append(gpu_used_mib())
        saved = []
        for index, candidate in enumerate(result["candidates"]):
            path = out / f"{payload['name']}_{index}_{candidate['manifest']['backend']}.png"
            candidate["image"].save(path)
            saved.append({"path": str(path), **{k: candidate["manifest"][k] for k in
                          ("backend", "timing", "peak_vram_mib", "vram_delta_mib",
                           "system_commit_before_mib", "system_commit_after_mib")}})
        return {"saved": saved, "warnings": result["warnings"]}

    gate = lambda payload: check_resources("interactive_low_memory", resource_snapshot())  # noqa: E731
    queue = JobQueue(queue_path)
    for index, (prompt, channel, people, mode) in enumerate(JOBS):
        queue.add({"name": f"job{index}", "prompt": prompt, "channel": channel, "people": people, "mode": mode,
                   "seed": 4100 + index, "composition": "SOLO_MEDIUM" if people == 1 else
                   ("COUPLE_MEDIUM" if people == 2 else "")})

    runner = QueueRunner(queue, executor, gate)
    first = queue.jobs[0]
    runner.on_event = lambda event, job: queue.pause_after_current() if event == "started" and job is first else None
    started = time.perf_counter()
    status = runner.run(wait_seconds=10, max_waits=6)
    report["checks"]["pause_after_current"] = {
        "status": status, "states": [j.state for j in queue.jobs],
        "pass": status == "paused" and [j.state for j in queue.jobs] == [DONE, PENDING, PENDING, PENDING]}

    reopened = JobQueue(queue_path)  # simulated app restart
    report["checks"]["persisted_across_restart"] = {
        "paused": reopened.paused, "states": [j.state for j in reopened.jobs],
        "pass": reopened.paused and [j.state for j in reopened.jobs] == [DONE, PENDING, PENDING, PENDING]}
    cancelled = reopened.cancel(reopened.jobs[2].id)
    reopened.resume()
    runner = QueueRunner(reopened, executor, gate)
    status = runner.run(wait_seconds=10, max_waits=6)
    states = [j.state for j in reopened.jobs]
    report["checks"]["cancel_pending_and_resume"] = {
        "status": status, "states": states, "pass": cancelled and status == "empty"
        and states == [DONE, DONE, CANCELLED, DONE]}

    baseline = report["baseline_gpu_mib"] or 0
    report["after_job_gpu_mib"] = after_job_gpu
    report["checks"]["model_unloaded_between_jobs"] = {
        "max_after_job_minus_baseline_mib": max((g or 0) - baseline for g in after_job_gpu),
        "pass": all(g is not None and g - baseline < 600 for g in after_job_gpu)}

    impossible = JobQueue(out / "queue_low_memory.json")
    impossible.jobs.clear()
    job = impossible.add({"name": "never", "prompt": "x", "mode": "preview"})
    ran: list = []
    tight = lambda payload: (False, ["simulated: GPU free below threshold"])  # noqa: E731
    status = QueueRunner(impossible, lambda p, c: ran.append(p) or {}, tight).run(wait_seconds=0.1, max_waits=3)
    real_gate = check_resources("interactive_low_memory", resource_snapshot(), engine_vram_mib=10 ** 6)
    report["checks"]["low_memory_keeps_job_queued"] = {
        "status": status, "state": job.state, "reason": job.waiting_reason, "real_gate_with_huge_need": real_gate,
        "pass": status == "waiting" and job.state == PENDING and not ran and real_gate[0] is False}

    report["jobs"] = [{"id": j.id, "state": j.state, "payload": j.payload, "result": j.result, "error": j.error,
                       "seconds": round((j.finished or 0) - (j.started or 0), 1)} for j in reopened.jobs]
    report["total_seconds"] = round(time.perf_counter() - started, 1)
    report["pass"] = all(check["pass"] for check in report["checks"].values())
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"pass": report["pass"], "checks": {k: v["pass"] for k, v in report["checks"].items()},
                      "after_job_gpu_mib": after_job_gpu, "baseline": baseline,
                      "total_seconds": report["total_seconds"]}, ensure_ascii=False))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

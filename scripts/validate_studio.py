"""Real-GPU validation of the AI 이미지 스튜디오 (same code path as the GUI: run_creator_job + StudioRunner).

    python scripts/validate_studio.py --models-dir D:\\models --output-dir validation_results\\studio [--only jobs|queue]

jobs : YouTube 4 (Tokyo Chill woman/man/couple, OLD POP couple with a Korean prompt) +
       Shopify 5 (hero, collection lifestyle, product-reference lifestyle, promo tile, mobile banner).
       The top candidate of each job is adopted and exported (textless + composed when text is set).
queue: mixed 5-job queue, pause-after-current, "restart" (new queue/runner objects from disk), resume,
       GPU memory back near baseline after every heavy job. Writes report.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from covermorph.creator_jobs import export_candidate, run_creator_job  # noqa: E402
from covermorph.creator_presets import BUILTIN_PROMPT_PRESETS as P, BUILTIN_PURPOSES  # noqa: E402
from covermorph.creator_runner import StudioRunner  # noqa: E402
from covermorph.job_queue import DONE, PENDING, JobQueue  # noqa: E402
from covermorph.quality_engines import gpu_used_mib  # noqa: E402

PRODUCT = REPOSITORY_ROOT / "validation_results" / "quality_v2" / "_refs" / "product_mug.png"
LOW = "interactive_low_memory"


def preset(key: str, **extra) -> dict:
    p = P[key]
    return {"purpose": p.purpose, "prompt": p.prompt, "prompt_preset": key, "channel": p.channel, "people": p.people,
            "composition": p.composition, **extra}


JOBS = [
    ("yt1_tokyo_woman", {**preset("tc_solo_woman"), "quality": "balanced", "memory": LOW, "seed": 6101, "candidates": 2}),
    ("yt2_tokyo_man", {**preset("tc_solo_man"), "quality": "balanced", "memory": LOW, "seed": 6102, "candidates": 2}),
    ("yt3_tokyo_couple", {**preset("tc_couple"), "quality": "balanced", "memory": LOW, "seed": 6103, "candidates": 2}),
    ("yt4_oldpop_couple", {**preset("op_mature_couple"), "prompt": "가을 공원 벤치에 나란히 앉은 50대 일본인 부부",
                           "quality": "balanced", "memory": LOW, "seed": 6104, "candidates": 2}),
    ("sh5_hero_no_product", {**preset("sh_editorial"), "quality": "balanced", "memory": LOW, "seed": 6105,
                             "candidates": 2, "title": "New Season Edit", "subtitle": "Linen & natural textures", "cta": "Shop now"}),
    ("sh6_collection_lifestyle", {**preset("sh_daylight"), "quality": "balanced", "memory": LOW, "seed": 6106,
                                  "candidates": 2, "title": "Home Collection"}),
    ("sh7_product_reference", {"purpose": "shopify_product_lifestyle", "prompt": "the product on a wooden cafe table in soft morning sunlight",
                               "references": [{"path": str(PRODUCT), "role": "PRODUCT"}], "product_preserve": True,
                               "product_scale": 0.55, "quality": "best", "memory": "night_best", "seed": 6107, "candidates": 2,
                               "title": "Teal Ceramic Mug"}),
    ("sh8_promo_tile", {**preset("sh_seasonal"), "references": [{"path": str(PRODUCT), "role": "PRODUCT"}],
                        "product_preserve": True, "product_scale": 0.5, "quality": "balanced", "memory": LOW, "seed": 6108,
                        "candidates": 2, "title": "가을 한정 20% 할인", "cta": "지금 구매"}),
    ("sh9_mobile_banner", {"purpose": "shopify_mobile", "prompt": "a cozy reading corner with a knit blanket, warm lamp light and autumn leaves outside the window",
                           "quality": "balanced", "memory": LOW, "seed": 6109, "candidates": 2,
                           "title": "가을 홈 컬렉션", "subtitle": "따뜻한 집을 위한 셀렉션", "cta": "둘러보기"}),
]


def run_jobs(models_dir: Path, out: Path, names: list[str] | None = None) -> list[dict]:
    rows = []
    for name, payload in JOBS:
        if names and name not in names:
            continue
        payload = {"kind": "generate", **payload}
        purpose = BUILTIN_PURPOSES[payload["purpose"]]
        payload.setdefault("canvas", [purpose.width, purpose.height])
        baseline = gpu_used_mib()
        started = time.perf_counter()
        print(f"== {name}", flush=True)
        result = run_creator_job(payload, None, models_dir=models_dir, output_root=out / "jobs", app_root=REPOSITORY_ROOT)
        top = result["candidates"][0]
        exports = export_candidate(top, purpose, export_dir=Path(result["job_dir"]) / "export", name=name,
                                   title=payload.get("title", ""), subtitle=payload.get("subtitle", ""),
                                   cta=payload.get("cta", ""), canvas=tuple(payload["canvas"]))
        row = {"name": name, "purpose": purpose.key, "canvas": payload["canvas"], "quality": payload["quality"],
               "memory": payload["memory"], "seconds": round(time.perf_counter() - started, 1), "job_dir": result["job_dir"],
               "gpu_baseline_mib": baseline, "gpu_after_mib": gpu_used_mib(),
               "commit_before_mib": result["memory_before"]["commit_total_mib"],
               "commit_after_mib": result["memory_after"]["commit_total_mib"], "warnings": result["warnings"],
               "exports": exports,
               "candidates": [{k: c.get(k) for k in ("kind", "engine", "seed", "seconds", "peak_vram_mib", "score",
                                                     "warnings", "translated_prompt", "compiled_prompt")}
                              | {"full": c["files"]["full"], "product_check": c.get("product_check")}
                              for c in result["candidates"]]}
        rows.append(row)
        print(json.dumps({k: row[k] for k in ("name", "seconds", "gpu_baseline_mib", "gpu_after_mib")}
                         | {"cands": [(c["kind"], c["engine"], c["seconds"], c["peak_vram_mib"], len(c["warnings"]))
                                      for c in row["candidates"]]}, ensure_ascii=False), flush=True)
    return rows


def run_queue(models_dir: Path, out: Path) -> dict:
    folder = out / "queue 큐"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "studio_queue.json"
    path.unlink(missing_ok=True)
    settings = {"recheck_seconds": 5}
    after_job: list[tuple[str, int | None]] = []
    baseline = gpu_used_mib()

    def executor(payload, cancel, progress):
        result = run_creator_job(payload, cancel, models_dir=models_dir, output_root=folder / "jobs",
                                 progress=progress, app_root=REPOSITORY_ROOT)
        after_job.append((payload["name"], gpu_used_mib()))
        return result

    mixed = [
        {"name": "q1_youtube", **preset("tc_solo_woman"), "quality": "preview", "seed": 7101},
        {"name": "q2_hero", **preset("sh_editorial"), "quality": "preview", "seed": 7102},
        {"name": "q3_product", "purpose": "shopify_promo_tile", "prompt": "a festive table with warm lights",
         "references": [{"path": str(PRODUCT), "role": "PRODUCT"}], "product_preserve": True, "quality": "preview", "seed": 7103},
        {"name": "q4_mobile", "purpose": "shopify_mobile", "prompt": "a minimal bathroom shelf with towels and plants",
         "quality": "preview", "seed": 7104},
        {"name": "q5_oldpop", **preset("op_mature_solo"), "quality": "preview", "seed": 7105},
    ]
    queue = JobQueue(path)
    for payload in mixed:
        purpose = BUILTIN_PURPOSES[payload["purpose"]]
        queue.add({"kind": "generate", "memory": LOW, "candidates": 1, "canvas": [purpose.width, purpose.height], **payload})
    runner = StudioRunner(queue, executor, settings)
    report: dict = {"baseline_gpu_mib": baseline, "checks": {}}
    first = threading.Event()
    original_event = runner._event

    def watch(event, job):
        original_event(event, job)
        if event == "started" and not first.is_set():
            first.set()
            runner.pause_after_current()  # user presses 현재 작업 후 일시정지 during job 1
    runner.runner.on_event = watch
    t0 = time.perf_counter()
    runner.start()
    deadline = time.time() + 900
    while time.time() < deadline and not (queue.jobs[0].state == DONE and runner.status == "일시정지됨"):
        time.sleep(1)
    time.sleep(3)
    states = [j.state for j in queue.jobs]
    report["checks"]["pause_after_current"] = {"states": states, "pass": states == [DONE] + [PENDING] * 4}
    runner.shutdown()

    reopened = JobQueue(path)  # simulated app restart: new objects from the file on disk
    states = [j.state for j in reopened.jobs]
    report["checks"]["restart_keeps_queue"] = {"paused": reopened.paused, "states": states,
                                               "pass": reopened.paused and states == [DONE] + [PENDING] * 4}
    runner2 = StudioRunner(reopened, executor, settings)
    runner2.start()  # 재개
    deadline = time.time() + 1800
    while time.time() < deadline and any(j.state == PENDING or j.state == "running" for j in reopened.jobs):
        time.sleep(2)
    runner2.shutdown()
    states = [j.state for j in reopened.jobs]
    report["checks"]["resume_completes_all"] = {"states": states, "pass": states == [DONE] * 5}
    report["after_job_gpu_mib"] = after_job
    worst = max((g or 0) - (baseline or 0) for _, g in after_job) if after_job else None
    report["checks"]["memory_back_to_baseline"] = {"baseline": baseline, "worst_delta_mib": worst,
                                                   "pass": worst is not None and worst < 600}
    report["jobs"] = [{"name": j.payload["name"], "state": j.state, "seconds": round((j.finished or 0) - (j.started or 0), 1),
                       "job_dir": (j.result or {}).get("job_dir"), "memory_before": (j.result or {}).get("memory_before"),
                       "memory_after": (j.result or {}).get("memory_after"),
                       "candidates": [(c["kind"], c["engine"], c["seconds"], c["peak_vram_mib"]) for c in (j.result or {}).get("candidates", [])]}
                      for j in reopened.jobs]
    report["total_seconds"] = round(time.perf_counter() - t0, 1)
    report["pass"] = all(c["pass"] for c in report["checks"].values())
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--only", choices=["jobs", "queue"])
    parser.add_argument("--names", nargs="*", help="re-run only these jobs (report rows are replaced)")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
    if args.only in (None, "jobs"):
        fresh = run_jobs(args.models_dir, args.output_dir, args.names)
        kept = [row for row in report.get("jobs", []) if row["name"] not in {r["name"] for r in fresh}]
        order = [name for name, _ in JOBS]
        report["jobs"] = sorted(kept + fresh, key=lambda row: order.index(row["name"]))
    if args.only in (None, "queue"):
        report["queue"] = run_queue(args.models_dir, args.output_dir)
        print(json.dumps({k: v["pass"] for k, v in report["queue"]["checks"].items()}, ensure_ascii=False), flush=True)
        print("after_job_gpu", report["queue"]["after_job_gpu_mib"], "baseline", report["queue"]["baseline_gpu_mib"])
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0 if report.get("queue", {}).get("pass", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())

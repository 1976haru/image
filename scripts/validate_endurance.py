"""v1.0 queue endurance on the real app process (packaged EXE by default).

    python scripts/validate_endurance.py --exe dist\\CoverMorphStudio\\CoverMorphStudio.exe --models-dir D:\\models \
        --output-dir "validation_results\\v1 내구성 テスト"

10 mixed jobs in an isolated data folder (Unicode path). Three app runs:
  A  start -> 현재 작업 후 일시정지 during job 1 -> app closed
  B  reopen -> 재개 -> app closed while a later job runs (must return to pending, backend killed)
  C  a helper process holds ~7.5 GB of GPU memory before the app opens (queue must wait: "자원 대기"),
     released after the wait is seen; the harness kills sd-cli during the "crash" job (forced backend
     failure -> friendly error) -> 실패 재시도 -> all 10 done.
After every run: no sd-cli / llama-completion / app process left, GPU memory back near the baseline.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

PRODUCTS = REPOSITORY_ROOT / "validation_results" / "v1_products"
CRASH_JOB = "q07_collection_crash"
VRAM_HOG_GIB = 7.5


def jobs() -> list[dict]:
    from covermorph.creator_presets import BUILTIN_PURPOSES as P

    def job(name, purpose, prompt, **extra):
        return {"name": name, "kind": "generate", "purpose": purpose, "canvas": [P[purpose].width, P[purpose].height],
                "prompt": prompt, "quality": "preview", "memory": "interactive_low_memory", "candidates": 1,
                "seed": 9500 + int(name[1:3]), **extra}

    mug = {"path": str(PRODUCTS / "mug.png"), "role": "PRODUCT"}
    box = {"path": str(PRODUCTS / "box.png"), "role": "PRODUCT"}
    return [
        job("q01_youtube_woman", "youtube_thumbnail", "side profile of a young Japanese woman on a Tokyo street at dusk", people=1, composition="SOLO_MEDIUM"),
        job("q02_hero", "shopify_hero", "an airy editorial living space with linen textiles and ceramic vases"),
        job("q03_product_strict", "shopify_product_lifestyle", "the product on a wooden cafe table", references=[mug], product_mode="strict"),
        job("q04_oldpop_korean", "youtube_thumbnail", "가을 공원 벤치에 앉은 50대 일본인 부부", channel="OLD POP LOUNGE", people=2, composition="COUPLE_MEDIUM"),
        job("q05_mobile", "shopify_mobile", "a cozy reading corner with warm lamp light", title="겨울 컬렉션"),
        job("q06_youtube_couple", "youtube_thumbnail", "a young Japanese couple at a cafe window", people=2, composition="COUPLE_MEDIUM"),
        job(CRASH_JOB, "shopify_collection", "a bright living room with soft morning daylight and plants"),
        job("q08_product_natural", "shopify_product_lifestyle", "the product on a dark walnut desk", references=[box], product_mode="natural"),
        job("q09_rainy_night", "youtube_thumbnail", "a rainy Shinjuku street at night with neon reflections, no people"),
        job("q10_promo", "shopify_promo_tile", "a festive wooden tabletop with warm string lights", references=[mug], product_mode="strict", title="20% 할인", cta="지금 구매"),
    ]


def procs(names=("sd-cli.exe", "llama-completion.exe", "CoverMorphStudio.exe")) -> dict[str, int]:
    out = subprocess.run(["tasklist", "/FO", "CSV", "/NH"], capture_output=True, text=True, errors="replace").stdout
    counts = {name: 0 for name in names}
    for line in out.splitlines():
        image = line.split(",")[0].strip('"').lower()
        for name in names:
            if image == name.lower():
                counts[name] += 1
    return counts


def gpu_used() -> int:
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout
    return int(out.strip().splitlines()[0])


def queue_state(data: Path) -> dict:
    try:
        return json.loads((data / "studio_queue.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"jobs": []}


def run_app(command: list[str], env: dict, phase: str, out: Path, timeout: float) -> float:
    started = time.time()
    process = subprocess.Popen([*command, "--studio-endurance", phase, str(out)], env=env)
    process.wait(timeout=timeout)
    return round(time.time() - started, 1)


def after_run(report: dict, key: str, baseline: int) -> None:
    time.sleep(4)
    left = procs()
    used = gpu_used()
    report[key] = {"leftover_processes": left, "gpu_used_mib": used, "gpu_delta_mib": used - baseline,
                   "pass_no_zombies": not any(left.values()), "pass_memory": used - baseline < 600}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exe", type=Path)
    parser.add_argument("--models-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    data = out / "_data 데이터"
    data.mkdir(parents=True, exist_ok=True)
    for stale in ("studio_queue.json", "settings.json"):
        (data / stale).unlink(missing_ok=True)
    env = {k: v for k, v in os.environ.items() if k != "COVERMORPH_MODELS_DIR"}
    env["COVERMORPH_DATA_DIR"] = str(data)
    os.environ["COVERMORPH_DATA_DIR"] = str(data)
    from covermorph.creator_settings import load_creator_settings, save_creator_settings
    from covermorph.job_queue import JobQueue
    settings = load_creator_settings(REPOSITORY_ROOT)
    settings.update(models_dir=str(args.models_dir), output_dir=str(out / "출력 結果"), setup_completed=True,
                    auto_start_when_free=False, recheck_seconds=10)
    save_creator_settings(REPOSITORY_ROOT, settings)
    queue = JobQueue(data / "studio_queue.json")
    for payload in jobs():
        queue.add(payload)
    command = [str(args.exe)] if args.exe else [sys.executable, str(REPOSITORY_ROOT / "app.py")]
    report: dict = {"command": command[0], "data_dir": str(data)}
    baseline = gpu_used()
    report["gpu_baseline_mib"] = baseline

    # ---- A: pause after current, close
    report["A_seconds"] = run_app(command, env, "a", out, 1200)
    a = json.loads((out / "phase_a.json").read_text(encoding="utf-8"))
    report["A"] = {**a, "pass": a["end_states"] == ["done"] + ["pending"] * 9 and a["end_paused"]}
    after_run(report, "A_after", baseline)

    # ---- B: reopen, resume, close while running
    report["B_seconds"] = run_app(command, env, "b", out, 1800)
    b = json.loads((out / "phase_b.json").read_text(encoding="utf-8"))
    states = [j["state"] for j in queue_state(data)["jobs"]]
    closed = b.get("closed_during")
    closed_state = next((j["state"] for j in queue_state(data)["jobs"] if j["payload"]["name"] == closed), None)
    report["B"] = {**b, "states_on_disk": states, "closed_job_state": closed_state,
                   "pass": b["start_states"] == a["end_states"] and closed_state == "pending" and states.count("running") == 0}
    after_run(report, "B_after", baseline)

    # ---- C: GPU memory held before start, forced backend crash, retry, finish
    hog = subprocess.Popen([sys.executable, "-c",
                            "import sys,torch; x=torch.empty(int(%f*2**30),dtype=torch.uint8,device='cuda'); "
                            "print('READY',flush=True); sys.stdin.read()" % VRAM_HOG_GIB],
                           stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    assert hog.stdout.readline().strip() == "READY"
    report["C_hog_gpu_used_mib"] = gpu_used()
    events: list[str] = []
    stop = threading.Event()
    t0 = time.time()

    def watch() -> None:
        hog_released = False
        crashed = False
        while not stop.wait(1):
            jobs_now = queue_state(data).get("jobs", [])
            waiting = [j for j in jobs_now if j["state"] == "pending" and j.get("waiting_reason")]
            if not hog_released and (waiting or time.time() - t0 > 120):
                events.append(f"{time.time() - t0:.0f}s queue waiting: {waiting[0]['waiting_reason'] if waiting else 'none seen'}")
                stop.wait(20)                                        # keep the pressure a little longer
                hog.kill()
                hog_released = True
                events.append(f"{time.time() - t0:.0f}s GPU memory released")
            running = [j for j in jobs_now if j["state"] == "running"]
            if not crashed and running and running[0]["payload"]["name"] == CRASH_JOB and procs()["sd-cli.exe"]:
                stop.wait(3)
                subprocess.run(["taskkill", "/F", "/IM", "sd-cli.exe"], capture_output=True)
                crashed = True
                events.append(f"{time.time() - t0:.0f}s killed sd-cli during {CRASH_JOB}")

    watcher = threading.Thread(target=watch, daemon=True)
    watcher.start()
    report["C_seconds"] = run_app(command, env, "c", out, 3600)
    stop.set()
    if hog.poll() is None:
        hog.kill()
    c = json.loads((out / "phase_c.json").read_text(encoding="utf-8"))
    final = queue_state(data)["jobs"]
    report["C"] = {**c, "harness_events": events,
                   "final_states": [j["state"] for j in final],
                   "crash_error": next((e for e in c.get("failed_errors", []) if e), ""),
                   "pass": [j["state"] for j in final] == ["done"] * 10 and bool(c.get("saw_wait"))
                   and c.get("retried") == [CRASH_JOB] and any(e.startswith("BACKEND_CRASH|") for e in c.get("failed_errors", []))}
    after_run(report, "C_after", baseline)
    report["no_lost_jobs"] = len(final) == 10 and sorted(j["payload"]["name"] for j in final) == sorted(p["name"] for p in jobs())
    report["pass"] = all(report[k]["pass"] for k in ("A", "B", "C")) and report["no_lost_jobs"] and all(
        report[k]["pass_no_zombies"] and report[k]["pass_memory"] for k in ("A_after", "B_after", "C_after"))
    (out / "endurance_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k]["pass"] if isinstance(report.get(k), dict) and "pass" in report[k] else report.get(k)
                      for k in ("A", "B", "C", "no_lost_jobs", "pass")}, ensure_ascii=False))
    for key in ("A_after", "B_after", "C_after"):
        print(key, report[key])
    print("C events", events, "saw_wait", c.get("saw_wait"), "crash_error", report["C"]["crash_error"])
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

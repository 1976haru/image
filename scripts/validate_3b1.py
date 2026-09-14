from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from threading import Event
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from covermorph.generation import (  # noqa: E402
    DEFAULT_IP_ADAPTER,
    DEFAULT_IP_ADAPTER_REVISION,
    DEFAULT_IP_ADAPTER_WEIGHT,
    GENERATION_SIZES,
    GenerationConfig,
    SDXLTextToImageEngine,
    detect_generation_environment,
    generate_scene_candidates,
)
from covermorph.project import (  # noqa: E402
    SceneCard,
    add_person,
    add_person_reference,
    create_project,
    save_project_atomic,
)

PROMPT = "cinematic portrait of a person in a Tokyo night cafe, cover art, no text"
NEGATIVE_PROMPT = "text, typography, logo, watermark, deformed, distorted face, bad anatomy, low quality"
SEED = 20260914
STEPS = 28
GUIDANCE_SCALE = 7.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the local CoverMorph 3-B1 comparison matrix.")
    parser.add_argument("--person-reference", type=Path)
    parser.add_argument("--style-reference", type=Path)
    parser.add_argument("--model-path", type=Path, default=Path("models/sdxl_base_1.0"))
    parser.add_argument("--adapter-path", type=Path, default=Path("models/ip_adapter"))
    parser.add_argument("--output-dir", type=Path, default=Path("validation_results/3b1_rtx3060_validation"))
    return parser.parse_args()


def nvidia_smi() -> dict[str, Any]:
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        return {"available": False, "error": str(exc)}
    return {"available": completed.returncode == 0, "raw": completed.stdout.strip(), "error": completed.stderr.strip() or None}


def inference_state(environment: dict[str, Any]) -> tuple[str, str | None]:
    if not environment.get("cuda"):
        return "not_run", "gpu_unavailable"
    if environment.get("status") != "ready":
        return "not_run", "package_missing"
    return "run", None


def blocked_job(label: str, mode: str, strength: float, ratio: str, reason: str) -> dict[str, Any]:
    return {
        "job_id": label,
        "requested_reference_mode": mode,
        "reference_strength": strength if mode != "off" else None,
        "actual_reference_applied": None,
        "person_id": None,
        "reference_image_id": None,
        "source_sha256": None,
        "processed_sha256": None,
        "crop_box": None,
        "preprocessing": None,
        "adapter_id": DEFAULT_IP_ADAPTER,
        "adapter_revision": DEFAULT_IP_ADAPTER_REVISION,
        "adapter_weight": DEFAULT_IP_ADAPTER_WEIGHT,
        "image_encoder": "models/image_encoder",
        "image_encoder_revision": DEFAULT_IP_ADAPTER_REVISION,
        "generation_status": "not_run",
        "failure_reason": reason,
        "visual_quality_status": "unverified",
        "size": list(GENERATION_SIZES[ratio]),
        "candidate_count": 1,
        "seed": SEED,
        "steps": STEPS,
        "guidance_scale": GUIDANCE_SCALE,
        "metrics": {
            "generation_time_seconds": None,
            "peak_memory_allocated": None,
            "peak_memory_reserved": None,
            "cuda_out_of_memory": False,
        },
    }


def run_job(
    label: str,
    mode: str,
    strength: float,
    ratio: str,
    reference_path: Path | None,
    args: argparse.Namespace,
    output_dir: Path,
) -> dict[str, Any]:
    if mode != "off" and reference_path is None:
        return blocked_job(label, mode, strength, ratio, "reference path was not supplied")
    job_dir = output_dir / "projects" / label
    project = create_project(job_dir, label)
    person = add_person(project, "validation-person")
    reference_id = ""
    if reference_path is not None:
        reference = add_person_reference(project, person, reference_path, mode, "other")
        reference_id = reference.image_id
    scene = SceneCard(
        scene_id=f"scene_{label}",
        prompt_user=PROMPT,
        negative_prompt_user=NEGATIVE_PROMPT,
        prompt_confirmed=True,
        output_ratio=ratio,
    )
    project.scenes.append(scene)
    config = GenerationConfig(
        model_id=str(args.model_path),
        output_ratio=ratio,
        candidate_count=1,
        seed=SEED,
        steps=STEPS,
        guidance_scale=GUIDANCE_SCALE,
        local_files_only=True,
        reference_mode=mode,
        reference_image_id=reference_id,
        reference_strength=strength,
        ip_adapter_id=str(args.adapter_path),
        ip_adapter_revision=DEFAULT_IP_ADAPTER_REVISION,
        ip_adapter_weight=DEFAULT_IP_ADAPTER_WEIGHT,
    )
    started = time.perf_counter()
    engine = SDXLTextToImageEngine(str(args.model_path), local_files_only=True)
    try:
        result = generate_scene_candidates(project, scene, engine, config, Event())
        save_project_atomic(project)
    except Exception as exc:
        return blocked_job(label, mode, strength, ratio, str(exc))
    finally:
        engine.unload()
    run = project.generation_runs[-1]
    candidate_path = None
    if result.candidate_ids:
        candidate = next(item for item in project.candidates if item.candidate_id == result.candidate_ids[0])
        candidate_path = output_dir / "generated" / f"{label}.png"
        candidate_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(project.project_dir / candidate.original_path, candidate_path)
    metrics = run.get("candidate_results", [{}])[0].get("metrics", {}) if run.get("candidate_results") else {}
    return {
        "job_id": label,
        "requested_reference_mode": mode,
        "reference_strength": strength if mode != "off" else None,
        "actual_reference_applied": run.get("actual_reference_applied"),
        "person_id": run.get("person_id"),
        "reference_image_id": run.get("reference_image_id"),
        "source_sha256": run.get("source_sha256"),
        "processed_sha256": run.get("processed_sha256"),
        "crop_box": run.get("crop_box"),
        "preprocessing": run.get("preprocessing"),
        "adapter_id": run.get("adapter_id", DEFAULT_IP_ADAPTER),
        "adapter_revision": run.get("adapter_revision", DEFAULT_IP_ADAPTER_REVISION),
        "adapter_weight": run.get("adapter_weight", DEFAULT_IP_ADAPTER_WEIGHT),
        "image_encoder": run.get("image_encoder", "models/image_encoder"),
        "image_encoder_revision": run.get("image_encoder_revision", DEFAULT_IP_ADAPTER_REVISION),
        "generation_status": run.get("generation_status"),
        "failure_reason": run.get("failure_reason"),
        "visual_quality_status": "unverified",
        "size": list(GENERATION_SIZES[ratio]),
        "candidate_count": 1,
        "seed": SEED,
        "steps": STEPS,
        "guidance_scale": GUIDANCE_SCALE,
        "elapsed_seconds_wall": round(time.perf_counter() - started, 3),
        "metrics": metrics,
        "project_json": str(project.project_file),
        "generated_png": str(candidate_path) if candidate_path else None,
    }


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    environment = detect_generation_environment(Path.cwd(), str(args.model_path))
    actual_inference, inference_reason = inference_state(environment)
    report: dict[str, Any] = {
        "validation": "CoverMorph Studio 3-B1",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "actual_inference": actual_inference,
        "reason": inference_reason,
        "environment": {**environment, "nvidia_smi": nvidia_smi()},
        "comparison_conditions": {
            "prompt": PROMPT,
            "negative_prompt": NEGATIVE_PROMPT,
            "seed": SEED,
            "steps": STEPS,
            "guidance_scale": GUIDANCE_SCALE,
            "candidate_count": 1,
            "sizes": {ratio: list(size) for ratio, size in GENERATION_SIZES.items()},
            "note": "1344x768 and 768x1344 are generation sizes, not final 16:9/9:16 output sizes.",
        },
        "reproduction_command": (
            r".\.venv\Scripts\python.exe scripts\validate_3b1.py "
            r"--person-reference <person-reference.png> --style-reference <style-reference.png> "
            r"--model-path models\sdxl_base_1.0 --adapter-path models\ip_adapter "
            r"--output-dir validation_results\3b1_rtx3060_validation"
        ),
        "jobs": [],
        "generated_pngs": [],
        "visual_quality_status": "unverified",
    }
    if not environment.get("cuda"):
        reason = "CUDA unavailable; actual SDXL/IP-Adapter inference was not run."
        for mode, references in (("person", args.person_reference), ("style", args.style_reference)):
            for ratio in GENERATION_SIZES:
                for strength in (0.0, 0.5, 0.8):
                    actual_strength = strength if strength else 0.5
                    label = f"{ratio.replace(':', 'x')}_{mode}_{'off' if strength == 0.0 else str(strength).replace('.', '_')}"
                    report["jobs"].append(blocked_job(label, "off" if strength == 0.0 else mode, actual_strength, ratio, reason))
    elif environment.get("status") != "ready":
        reason = f"generation environment is not ready: {environment.get('status')}"
        for mode, _references in (("person", args.person_reference), ("style", args.style_reference)):
            for ratio in GENERATION_SIZES:
                for strength in (0.0, 0.5, 0.8):
                    actual_strength = strength if strength else 0.5
                    label = f"{ratio.replace(':', 'x')}_{mode}_{'off' if strength == 0.0 else str(strength).replace('.', '_')}"
                    report["jobs"].append(blocked_job(label, "off" if strength == 0.0 else mode, actual_strength, ratio, reason))
    else:
        for ratio in GENERATION_SIZES:
            for mode, reference_path in (("person", args.person_reference), ("style", args.style_reference)):
                for strength in (0.0, 0.5, 0.8):
                    actual_mode = "off" if strength == 0.0 else mode
                    actual_strength = strength if strength else 0.5
                    label = f"{ratio.replace(':', 'x')}_{mode}_{'off' if strength == 0.0 else str(strength).replace('.', '_')}"
                    report["jobs"].append(run_job(label, actual_mode, actual_strength, ratio, reference_path if actual_mode != "off" else None, args, output_dir))
    report["generated_pngs"] = [job["generated_png"] for job in report["jobs"] if job.get("generated_png")]
    (output_dir / "environment.json").write_text(json.dumps(report["environment"], ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "generation_comparison.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "project_blocked.json").write_text(json.dumps({"jobs": report["jobs"]}, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    blocked_jobs_dir = output_dir / "projects_blocked"
    blocked_jobs_dir.mkdir(parents=True, exist_ok=True)
    for job in report["jobs"]:
        (blocked_jobs_dir / f"{job['job_id']}.json").write_text(json.dumps(job, ensure_ascii=False, indent=2), encoding="utf-8")
    failures = [f"{job['job_id']}: {job['failure_reason']}" for job in report["jobs"] if job.get("failure_reason")]
    (output_dir / "errors.log").write_text("\n".join(failures) + ("\n" if failures else ""), encoding="utf-8")
    (output_dir / "visual_quality_notes.md").write_text(
        "# 3-B1 visual quality\n\n"
        "Status: `unverified`. No automatic result is treated as visual-quality approval.\n\n"
        "This run could not inspect face, proportions, hands, eyes, hair, background collisions, style-induced composition changes, text/logo artifacts, or PNG integrity because CUDA inference was not executed.\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

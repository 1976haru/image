"""Real-GPU person-quality validation: SDXL base vs photoreal FAST vs photoreal QUALITY (+face detail).

    python scripts/validate_person_quality.py --models-dir D:\\models --output-dir validation_results\\person_quality

Uses the same prompt/QA/face-detail/finish functions as the bridge. Each model is loaded once and
every scene keeps one fixed seed across variants. Writes full 1280x720 images, face crops, 340px and
180px previews, contact sheets, metrics.json and a blank reviewer sheet (reviewer_sheet.csv).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from threading import Event

from PIL import Image, ImageDraw

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from covermorph.generation import GENERATION_SIZES, GenerationConfig, SDXLTextToImageEngine  # noqa: E402
from covermorph.person_quality import (  # noqa: E402
    COMPOSITION_PROFILES,
    MODEL_PROFILES,
    QUALITY_MODES,
    YUNET_FILENAME,
    detect_faces_yunet,
    face_sharpness,
    finish_image,
    run_person_pass,
)
from covermorph.thumbnail_bridge_ai import MEMORY_PROFILES, build_prompt_plan, fit_prompt  # noqa: E402
from covermorph.thumbnail_bridge_assets import fit_16x9  # noqa: E402

SCENES = [
    ("tc1_solo_woman_street", "Tokyo Chill", "a young Japanese woman on an evening Tokyo street", 1, "SOLO_CLOSE", 1101),
    ("tc2_solo_man_platform", "Tokyo Chill", "a young Japanese man standing on a train platform", 1, "SOLO_MEDIUM", 1102),
    ("tc3_couple_cafe_window", "Tokyo Chill", "a young Japanese man and woman couple sitting by a cafe window", 2, "COUPLE_MEDIUM", 1103),
    ("tc4_couple_night_backlight", "Tokyo Chill", "a young Japanese man and woman couple on a night street with cinematic backlight", 2, "COUPLE_EMOTIONAL", 1104),
    ("tc5_profile_woman_rain", "Tokyo Chill", "side profile of a young woman on a rainy Tokyo street", 1, "SOLO_MEDIUM", 1105),
    ("op6_mature_couple_autumn", "OLD POP LOUNGE", "a mature Japanese man and woman couple in their fifties in an autumn park", 2, "COUPLE_MEDIUM", 1106),
    ("op7_mature_woman_snow", "OLD POP LOUNGE", "a mature woman in her fifties in the first snow", 1, "SOLO_CLOSE", 1107),
    ("op8_mature_man_cafe", "OLD POP LOUNGE", "a mature man in his fifties by a cafe window", 1, "SOLO_MEDIUM", 1108),
]
VARIANTS = ("sdxl_base", "photoreal_fast", "photoreal_quality_nofd", "photoreal_quality_fd")


def make_engine(models_dir: Path, profile_name: str) -> SDXLTextToImageEngine:
    profile = MODEL_PROFILES[profile_name]
    engine = SDXLTextToImageEngine(model_id=str(models_dir / profile["folder"]), local_files_only=True)
    engine.memory_profile = {k: v for k, v in MEMORY_PROFILES["balanced"].items() if k != "working_size"}
    engine.variant = profile["variant"]
    engine.scheduler_name = profile["scheduler"]
    started = time.perf_counter()
    engine.load()
    print(f"loaded {profile_name} in {time.perf_counter() - started:.1f}s", flush=True)
    return engine


def face_crop(image: Image.Image, yunet: Path, size: int = 320) -> tuple[Image.Image, dict]:
    faces = detect_faces_yunet(image, yunet, 0.5)
    if not faces:
        return Image.new("RGB", (size, size), (40, 40, 40)), {"faces": 0}
    face = faces[0]
    x, y, w, h = face["box"]
    width, height = image.size
    side = max(w * width, h * height) * 1.8
    cx, cy = (x + w / 2) * width, (y + h / 2) * height
    crop = image.crop((int(cx - side / 2), int(cy - side / 2), int(cx + side / 2), int(cy + side / 2)))
    return crop.resize((size, size), Image.Resampling.LANCZOS), {
        "faces": len(faces), "confidence": round(face["confidence"], 3), "face_height": round(h, 3),
        "sharpness": round(face_sharpness(image, face["box"]), 1)}


def save_outputs(out: Path, scene: str, variant: str, canvas: Image.Image, yunet: Path) -> dict:
    folder = out / scene
    folder.mkdir(parents=True, exist_ok=True)
    canvas.save(folder / f"{variant}_1280.png")
    crop, face = face_crop(canvas, yunet)
    crop.save(folder / f"{variant}_face.png")
    canvas.resize((340, 191), Image.Resampling.LANCZOS).save(folder / f"{variant}_340.png")
    canvas.resize((180, 101), Image.Resampling.LANCZOS).save(folder / f"{variant}_180.png")
    return face


def generate(engine, prompt_parts, negative_parts, steps, guidance, seed):
    warnings: list[str] = []
    prompt = fit_prompt(prompt_parts, engine.pipeline.tokenizer, warnings)
    negative = fit_prompt(negative_parts, engine.pipeline.tokenizer, [])
    config = GenerationConfig(model_id=engine.model_id, steps=steps, guidance_scale=guidance,
                              working_size=GENERATION_SIZES["16:9"])
    image = engine.generate_one(prompt, negative, config, seed, Event())
    return image, config, prompt, negative, dict(engine.last_generation_metrics)


def run(models_dir: Path, out: Path, matrix: bool) -> dict:
    yunet = models_dir / "face_detection" / YUNET_FILENAME
    results: dict = {"scenes": {}, "matrix": []}
    for name, *_ in SCENES:
        results["scenes"][name] = {}

    engine = make_engine(models_dir, "sdxl_base")
    for name, channel, prompt, people, composition, seed in SCENES:
        plan = build_prompt_plan(prompt, channel, "left", True)  # previous bridge prompt path
        image, *_, metrics = generate(engine, plan.prompt_parts, plan.negative_parts, 28, 7.0, seed)
        face = save_outputs(out, name, "sdxl_base", fit_16x9(image), yunet)
        results["scenes"][name]["sdxl_base"] = {**metrics, **face, "total_seconds": metrics["generation_time_seconds"]}
        print(name, "sdxl_base", results["scenes"][name]["sdxl_base"], flush=True)
    engine.unload()

    engine = make_engine(models_dir, "photoreal_sdxl")
    guidance = MODEL_PROFILES["photoreal_sdxl"]["guidance_scale"]
    for name, channel, prompt, people, composition, seed in SCENES:
        framing = COMPOSITION_PROFILES[composition]["framing"]
        plan = build_prompt_plan(prompt, channel, "left", True, person=True, framing=framing)
        # FAST: single pass.
        image, *_, metrics = generate(engine, plan.prompt_parts, plan.negative_parts, QUALITY_MODES["fast"]["steps"], guidance, seed)
        face = save_outputs(out, name, "photoreal_fast", fit_16x9(image), yunet)
        results["scenes"][name]["photoreal_fast"] = {**metrics, **face, "total_seconds": metrics["generation_time_seconds"]}
        # QUALITY: generation + QA (one regenerate) + face detail + finish.
        started = time.perf_counter()
        image, config, fitted, negative, metrics = generate(engine, plan.prompt_parts, plan.negative_parts,
                                                            QUALITY_MODES["quality"]["steps"], guidance, seed)
        final, info = run_person_pass(engine, image, config=config, mode="quality", prompt=fitted, negative_prompt=negative,
                                      seed=seed, expected_people=people, composition=composition, model_path=yunet,
                                      face_detail=True, keep_intermediate=True)  # opt-in pass, forced on to compare
        before = info.pop("_before_face_detail")
        nofd_face = save_outputs(out, name, "photoreal_quality_nofd", finish_image(fit_16x9(before)), yunet)
        fd_canvas = finish_image(fit_16x9(final))
        total = round(time.perf_counter() - started, 3)
        fd_face = save_outputs(out, name, "photoreal_quality_fd", fd_canvas, yunet)
        detail = info.get("face_detail") or {}
        common = {**metrics, "qa": info["qa"], "regenerated": info["regenerated"], "seed_used": info["seed_used"]}
        results["scenes"][name]["photoreal_quality_nofd"] = {**common, **nofd_face}
        results["scenes"][name]["photoreal_quality_fd"] = {
            **common, **fd_face, "total_seconds": total, "face_detail": detail,
            "face_detail_applied": info["face_detail_applied"],
            "peak_memory_allocated_overall": max(filter(None, [metrics.get("peak_memory_allocated"), detail.get("peak_memory_allocated")])),
            "peak_memory_reserved_overall": max(filter(None, [metrics.get("peak_memory_reserved"), detail.get("peak_memory_reserved")]))}
        print(name, "quality", {k: results["scenes"][name]["photoreal_quality_fd"][k] for k in
                                ("total_seconds", "face_detail_applied", "regenerated", "sharpness", "face_height")}, flush=True)
    if matrix:
        for name, channel, prompt, people, composition, seed in (SCENES[0], SCENES[2]):
            plan = build_prompt_plan(prompt, channel, "left", True, person=True,
                                     framing=COMPOSITION_PROFILES[composition]["framing"])
            for steps in (22, 30):
                for cfg in (4.0, 6.0):
                    image, *_, metrics = generate(engine, plan.prompt_parts, plan.negative_parts, steps, cfg, seed)
                    face = save_outputs(out / "matrix", name, f"s{steps}_cfg{cfg:g}", fit_16x9(image), yunet)
                    results["matrix"].append({"scene": name, "steps": steps, "cfg": cfg, **metrics, **face})
                    print("matrix", name, steps, cfg, face, metrics["generation_time_seconds"], flush=True)
    engine.unload()
    return results


def contact_sheets(out: Path) -> None:
    labels = {"sdxl_base": "SDXL base (before)", "photoreal_fast": "photoreal FAST",
              "photoreal_quality_nofd": "photoreal QUALITY", "photoreal_quality_fd": "photoreal QUALITY + face detail"}
    for kind, cell in (("face", (320, 320)), ("340", (340, 191)), ("180", (180, 101))):
        sheet = Image.new("RGB", (cell[0] * 4 + 50, (cell[1] + 6) * len(SCENES) + 24), "white")
        draw = ImageDraw.Draw(sheet)
        for column, variant in enumerate(VARIANTS):
            draw.text((column * (cell[0] + 12) + 4, 6), labels[variant], fill="black")
        for row, (name, *_rest) in enumerate(SCENES):
            for column, variant in enumerate(VARIANTS):
                with Image.open(out / name / f"{variant}_{kind}.png") as image:
                    sheet.paste(image, (column * (cell[0] + 12), 24 + row * (cell[1] + 6)))
        sheet.save(out / f"sheet_{kind}.png")


def reviewer_sheet(out: Path) -> None:
    with (out / "reviewer_sheet.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(["scene", "variant", "face realism 1-5", "eyes 1-5", "hair 1-5", "anatomy/hands 1-5",
                         "subject sharpness 1-5", "thumbnail readability 1-5", "overall 1-5", "notes"])
        for name, *_ in SCENES:
            for variant in VARIANTS:
                writer.writerow([name, variant, "", "", "", "", "", "", "", ""])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("validation_results/person_quality"))
    parser.add_argument("--no-matrix", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    results = run(args.models_dir, args.output_dir, not args.no_matrix)
    results["wall_seconds"] = round(time.perf_counter() - started, 1)
    import torch

    results["gpu"] = torch.cuda.get_device_name(0)
    results["vram_bytes"] = int(torch.cuda.get_device_properties(0).total_memory)
    (args.output_dir / "metrics.json").write_text(json.dumps(results, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    contact_sheets(args.output_dir)
    reviewer_sheet(args.output_dir)
    print(f"done in {results['wall_seconds']}s -> {args.output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Quality Engine V2 benchmark: same scenes/seeds across Z-Image-Turbo, FLUX.2-klein-4B and RealVisXL.

    python scripts/benchmark_quality_v2.py --models-dir D:\\models --output-dir validation_results\\quality_v2 \
        --engines zimage_turbo flux2_klein_4b realvisxl_v5 [--scenes tc1 tc2 ...]

Engines run one at a time (all scenes for one engine, unload, next engine); nothing is kept resident in
parallel. Per scene/engine it writes the 1280x720 (or Shopify-size) image, a face crop, 340/180 px
previews, and metrics.json with the original + compiled prompts, timing, GPU peak and commit delta.
Reference scenes read their reference images from <output-dir>/_refs (see --make-refs).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from PIL import Image, ImageDraw

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from covermorph.person_quality import YUNET_FILENAME, detect_faces_yunet, quality_check, COMPOSITION_PROFILES  # noqa: E402
from covermorph.prompt_compilers import PURPOSES, compile_prompt  # noqa: E402
from covermorph.quality_engines import EngineJob, Reference, make_backend, resource_snapshot  # noqa: E402

# key, channel, prompt, people, composition, seed, purpose, references [(file, role)], edit_instruction
SCENES = [
    ("tc1_solo_woman_profile", "Tokyo Chill", "side profile of a young Japanese woman on a rainy Tokyo street at dusk",
     1, "SOLO_MEDIUM", 2101, "youtube_thumbnail_background", [], ""),
    ("tc2_solo_man", "Tokyo Chill", "a young Japanese man standing on a train platform in the evening",
     1, "SOLO_MEDIUM", 2102, "youtube_thumbnail_background", [], ""),
    ("tc3_couple", "Tokyo Chill", "a young Japanese man and woman couple sitting by a cafe window",
     2, "COUPLE_MEDIUM", 2103, "youtube_thumbnail_background", [], ""),
    ("tc4_rainy_night", "Tokyo Chill", "a quiet rainy night street in Shinjuku with neon reflections on wet asphalt, no people",
     0, "", 2104, "youtube_thumbnail_background", [], ""),
    ("op5_mature_couple", "OLD POP LOUNGE", "a mature Japanese man and woman couple in their fifties in an autumn park",
     2, "COUPLE_MEDIUM", 2105, "youtube_thumbnail_background", [], ""),
    ("op6_mature_solo", "OLD POP LOUNGE", "a mature woman in her fifties sitting by a retro coffee shop window",
     1, "SOLO_CLOSE", 2106, "youtube_thumbnail_background", [], ""),
    ("sh7_lifestyle_hero", "", "a bright minimalist living room with a linen sofa, plants and morning sunlight",
     0, "", 2107, "shopify_hero_banner", [], ""),
    ("sh8_product_ref_banner", "", "the ceramic mug on a wooden table in a sunny cafe with soft morning light",
     0, "", 2108, "shopify_hero_banner", [("product_mug.png", "PRODUCT")], ""),
    ("rf9_person_ref", "Tokyo Chill", "the same woman walking under cherry blossoms in a Tokyo park in spring",
     1, "SOLO_MEDIUM", 2109, "youtube_thumbnail_background", [("person_woman.png", "PERSON")], ""),
    ("rf10_composition_ref", "Tokyo Chill", "a young Japanese man on a rooftop at sunset",
     1, "SOLO_MEDIUM", 2110, "youtube_thumbnail_background", [("composition.png", "COMPOSITION")], ""),
    ("ed11_edit_background", "Tokyo Chill", "the same woman, now standing on a quiet snowy street at night",
     1, "SOLO_MEDIUM", 2111, "youtube_thumbnail_background", [("person_woman.png", "PERSON")],
     "Keep the woman exactly as she is and change the background to a quiet snowy street at night, with an empty dark area on the left for a title"),
]

# Reference images, generated once by --make-refs with Z-Image so every engine sees identical inputs.
REF_SPECS = {
    "product_mug.png": ("a matte teal ceramic coffee mug with a small white geometric triangle logo, studio product "
                        "packshot on a plain light grey background, soft even light", 3001, (1024, 1024)),
    "person_woman.png": ("a young Japanese woman with shoulder-length black hair wearing a cream knit sweater, "
                         "waist-up portrait on a quiet Tokyo street in the evening, natural skin texture", 3002, (1024, 1024)),
    "composition.png": ("a lone person standing small on the right side of a wide empty seaside pier at dusk, the left "
                        "two thirds of the frame is empty sky and sea", 3003, (1280, 720)),
}


def face_crop(image: Image.Image, yunet: Path, size: int = 320) -> Image.Image:
    faces = detect_faces_yunet(image, yunet, 0.5)
    if not faces:
        return image.resize((size, size * image.height // image.width))
    x, y, w, h = faces[0]["box"]
    side = max(w * image.width, h * image.height) * 1.8
    cx, cy = (x + w / 2) * image.width, (y + h / 2) * image.height
    crop = image.crop((int(cx - side / 2), int(cy - side / 2), int(cx + side / 2), int(cy + side / 2)))
    return crop.resize((size, size), Image.Resampling.LANCZOS)


def save_outputs(folder: Path, engine: str, image: Image.Image, yunet: Path, people: int, composition: str) -> dict:
    folder.mkdir(parents=True, exist_ok=True)
    image.save(folder / f"{engine}_full.png")
    face_crop(image, yunet).save(folder / f"{engine}_face.png")
    for width in (340, 180):
        image.resize((width, round(width * image.height / image.width)), Image.Resampling.LANCZOS).save(
            folder / f"{engine}_{width}.png")
    profile = COMPOSITION_PROFILES.get(composition, {})
    qa = quality_check(image, detect_faces_yunet(image, yunet, 0.5), people,
                       profile.get("min_face_height", 0.0), profile.get("max_face_height", 1.0))
    return qa


def make_refs(models_dir: Path, refs_dir: Path) -> None:
    refs_dir.mkdir(parents=True, exist_ok=True)
    backend = make_backend("zimage_turbo", models_dir)
    for name, (prompt, seed, (width, height)) in REF_SPECS.items():
        if (refs_dir / name).exists():
            continue
        result = backend.generate(EngineJob(prompt=prompt + ". No text.", width=width, height=height, seed=seed))
        result.image.save(refs_dir / name)
        print(f"ref {name} {result.seconds}s", flush=True)
    backend.unload()


def run_engine(engine: str, models_dir: Path, out: Path, scene_keys: set[str] | None, steps: int | None) -> list[dict]:
    yunet = models_dir / "face_detection" / YUNET_FILENAME
    backend = make_backend(engine, models_dir)
    rows: list[dict] = []
    for key, channel, prompt, people, composition, seed, purpose, refs, edit in SCENES:
        if scene_keys and not any(key.startswith(k) for k in scene_keys):
            continue
        references = [Reference(out / "_refs" / name, role) for name, role in refs]
        if references and not backend.info.supports_single_reference:
            continue
        if edit and not backend.info.supports_edit:
            continue
        if len(references) > 1 and not backend.info.supports_multi_reference:
            references = references[:1]
        compiled = compile_prompt(engine, prompt, channel, purpose, composition=composition, person=people > 0,
                                  references=references, edit_instruction=edit)
        width, height = PURPOSES[purpose]["size"]
        job = EngineJob(prompt=compiled.positive, negative_prompt=compiled.negative, width=width, height=height,
                        seed=seed, steps=steps, references=references, prompt_parts=compiled.prompt_parts,
                        negative_parts=compiled.negative_parts)
        if edit:  # the PERSON reference is the image being edited
            job.edit_image, job.references = references[0].path, references[1:]
        before = resource_snapshot()
        started = time.perf_counter()
        try:
            result = backend.edit(job) if edit else backend.generate(job)
        except Exception as exc:  # record and continue: one failed scene must not hide the others
            rows.append({"scene": key, "engine": engine, "error": f"{type(exc).__name__}: {exc}"})
            print(key, engine, "ERROR", exc, flush=True)
            continue
        image = result.image if result.image.size == (width, height) else result.image.resize((width, height), Image.Resampling.LANCZOS)
        qa = save_outputs(out / key, engine, image, yunet, people, composition)
        row = {"scene": key, "engine": engine, "model": backend.info.model, "quantization": backend.info.quantization,
               "license": backend.info.license_id, "commercial_ok": backend.info.commercial_ok, "seed": seed,
               "size": [width, height], "steps": steps or backend.info.default_steps,
               "seconds": result.seconds, "wall_seconds": round(time.perf_counter() - started, 2),
               "gpu_used_before_mib": before.gpu_used_mib, "gpu_peak_mib": result.peak_vram_mib,
               "gpu_delta_mib": result.vram_delta_mib, "commit_before_mib": result.commit_before_mib,
               "commit_after_mib": result.commit_after_mib, "qa": qa, "prompt": compiled.to_dict()}
        (out / key / f"{engine}_metrics.json").write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
        rows.append(row)
        print(key, engine, f"{result.seconds}s gpu_peak={result.peak_vram_mib} delta={result.vram_delta_mib} "
              f"faces={len(qa['faces'])} pass={qa['passed']}", flush=True)
    backend.unload()
    return rows


def contact_sheets(out: Path, engines: list[str]) -> None:
    for folder in sorted(p for p in out.iterdir() if p.is_dir() and not p.name.startswith("_")):
        for kind, width in (("full", 640), ("face", 240), ("340", 340), ("180", 180)):
            tiles = [(e, folder / f"{e}_{kind}.png") for e in engines if (folder / f"{e}_{kind}.png").exists()]
            if not tiles:
                continue
            images = [Image.open(p).convert("RGB") for _, p in tiles]
            images = [im.resize((width, round(width * im.height / im.width))) if kind in ("full", "face") else im for im in images]
            height = max(im.height for im in images) + 22
            sheet = Image.new("RGB", (sum(im.width + 8 for im in images), height), (24, 24, 24))
            x = 0
            draw = ImageDraw.Draw(sheet)
            for (engine, _), im in zip(tiles, images):
                sheet.paste(im, (x, 22))
                draw.text((x + 4, 4), engine, fill=(230, 230, 230))
                x += im.width + 8
            sheet.save(folder / f"compare_{kind}.png")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--engines", nargs="+", default=["zimage_turbo", "flux2_klein_4b", "realvisxl_v5"])
    parser.add_argument("--scenes", nargs="*")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--make-refs", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.make_refs:
        make_refs(args.models_dir, args.output_dir / "_refs")
    all_rows: list[dict] = []
    for engine in args.engines:
        all_rows += run_engine(engine, args.models_dir, args.output_dir, set(args.scenes or []), args.steps)
    summary = args.output_dir / "summary.json"
    previous = json.loads(summary.read_text(encoding="utf-8")) if summary.exists() else []
    done = {(r["scene"], r["engine"]) for r in all_rows}
    merged = [r for r in previous if (r["scene"], r["engine"]) not in done] + all_rows
    summary.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
    contact_sheets(args.output_dir, ["zimage_turbo", "flux2_klein_4b", "realvisxl_v5"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

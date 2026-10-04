"""v1.0 real-use validation: 20 jobs through the studio's queue executor (preflight + friendly errors).

    python scripts/validate_v1.py --models-dir D:\\models --output-dir validation_results\\v1 [--names ...]

Uses an isolated data folder (COVERMORPH_DATA_DIR=<output>/_data) so the user's real settings/queue are
untouched. Each job's top candidate is exported (textless + composed when text is set). Writes report.json
with output paths, engine/mode, runtime, GPU peak, commit delta and warnings; visual verdicts are added by hand
in docs/V1_RELEASE_VALIDATION.md after looking at every output.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from threading import Event

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

P = REPOSITORY_ROOT / "validation_results" / "v1_products"
LOW = "interactive_low_memory"


def yt(prompt, channel, people, composition, seed, **extra):
    return {"purpose": "youtube_thumbnail", "prompt": prompt, "channel": channel, "people": people,
            "composition": composition, "seed": seed, **extra}


def product(name, role="PRODUCT", mask=None):
    ref = {"path": str(P / f"{name}.png"), "role": role}
    if mask:
        ref["mask"] = str(P / mask)
    return ref


JOBS = [
    # ---- YouTube: Tokyo Chill 6
    ("tc01_woman", yt("side profile of a young woman on a quiet Tokyo street at dusk", "Tokyo Chill", 1, "SOLO_MEDIUM", 9101, prompt_preset="tc_solo_woman")),
    ("tc02_man", yt("a young man leaning on a railing by the Sumida river at night", "Tokyo Chill", 1, "SOLO_MEDIUM", 9102, prompt_preset="tc_solo_man")),
    ("tc03_couple", yt("a young man and woman couple walking under one umbrella in Shibuya", "Tokyo Chill", 2, "COUPLE_MEDIUM", 9103, prompt_preset="tc_couple")),
    ("tc04_rainy_night", yt("비 오는 신주쿠 밤거리, 네온이 젖은 아스팔트에 비치는 풍경, 사람 없음", "Tokyo Chill", 0, "", 9104)),
    ("tc05_cafe", yt("窓際のカフェで本を読む若い日本人女性", "Tokyo Chill", 1, "SOLO_CLOSE", 9105)),
    ("tc06_platform", yt("a young Japanese man waiting on an empty train platform at sunset", "Tokyo Chill", 1, "SOLO_MEDIUM", 9106)),
    # ---- YouTube: OLD POP 4
    ("op07_mature_woman", yt("a mature Japanese woman in her fifties listening to records in a retro living room", "OLD POP LOUNGE", 1, "SOLO_MEDIUM", 9107)),
    ("op08_mature_man", yt("눈 오는 날 재즈바 창가에 앉은 60대 일본인 남성", "OLD POP LOUNGE", 1, "SOLO_MEDIUM", 9108)),
    ("op09_mature_couple", yt("a mature Japanese couple in their fifties dancing slowly in a dim jazz lounge", "OLD POP LOUNGE", 2, "COUPLE_EMOTIONAL", 9109)),
    ("op10_seasonal", yt("cherry blossom petals falling over an old Showa-era shopping street in spring, no people", "OLD POP LOUNGE", 0, "", 9110)),
    # ---- Shopify: Hero 2
    ("sh11_hero_editorial", {"purpose": "shopify_hero", "prompt_preset": "sh_editorial", "seed": 9111,
                             "prompt": "an airy editorial living space with linen textiles, ceramic vases, light oak furniture and soft directional light",
                             "title": "New Season Edit", "subtitle": "Natural textures for calm homes", "cta": "Shop now"}),
    ("sh12_hero_product", {"purpose": "shopify_hero", "seed": 9112, "prompt": "a white marble bathroom counter in soft morning daylight",
                           "references": [product("bottle")], "product_mode": "strict", "product_scale": 0.55,
                           "title": "LUMA Serum", "cta": "지금 구매"}),
    # ---- Shopify: Collection 2
    ("sh13_collection_daylight", {"purpose": "shopify_collection", "prompt_preset": "sh_daylight", "seed": 9113,
                                  "prompt": "a bright home interior with soft morning daylight, linen textures and green plants",
                                  "title": "Home Collection"}),
    ("sh14_collection_square", {"purpose": "shopify_collection", "canvas": [1200, 1200], "seed": 9114,
                                "prompt": "가을 햇살이 드는 따뜻한 원목 거실, 니트 담요와 머그잔, 아늑한 분위기", "title": "가을 컬렉션"}),
    # ---- Shopify: Product Lifestyle 4
    ("sh15_strict_mug", {"purpose": "shopify_product_lifestyle", "seed": 9115, "prompt": "the product on a wooden cafe table in soft morning sunlight",
                         "references": [product("mug")], "product_mode": "strict", "product_scale": 0.5}),
    ("sh16_natural_box", {"purpose": "shopify_product_lifestyle", "seed": 9116, "prompt": "the product on a dark walnut desk with warm evening lamp light",
                          "references": [product("box")], "product_mode": "natural", "product_scale": 0.45}),
    ("sh17_strict_plant_masked", {"purpose": "shopify_product_lifestyle", "seed": 9117, "prompt": "the product on a bright white windowsill with sheer curtains",
                                  "references": [product("plant", mask="plant_mask_corrected.png")], "product_mode": "strict", "product_scale": 0.55}),
    ("sh18_ai_reference_bottle", {"purpose": "shopify_product_lifestyle", "seed": 9118, "quality": "best", "memory": "night_best",
                                  "prompt": "the product on a white marble bathroom counter in soft morning daylight",
                                  "references": [product("bottle")], "product_mode": "ai", "product_scale": 0.5}),
    # ---- Shopify: Promo 1, Mobile 1
    ("sh19_promo", {"purpose": "shopify_promo_tile", "seed": 9119, "prompt_preset": "sh_seasonal",
                    "prompt": "a festive wooden tabletop with warm string lights and tasteful seasonal decorations softly blurred in the background",
                    "references": [product("mug")], "product_mode": "strict", "product_scale": 0.5,
                    "title": "겨울 한정 20% 할인", "cta": "지금 구매"}),
    ("sh20_mobile", {"purpose": "shopify_mobile", "seed": 9120,
                     "prompt": "a cozy reading corner with a knit blanket, warm lamp light and snow falling outside the window",
                     "title": "겨울 홈 컬렉션", "subtitle": "따뜻한 집을 위한 셀렉션", "cta": "둘러보기"}),
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--names", nargs="*")
    args = parser.parse_args()
    out = args.output_dir.resolve()
    os.environ["COVERMORPH_DATA_DIR"] = str(out / "_data")
    os.environ.pop("COVERMORPH_MODELS_DIR", None)
    from covermorph.creator_gui import execute_job
    from covermorph.creator_jobs import export_candidate
    from covermorph.creator_presets import BUILTIN_PURPOSES
    from covermorph.creator_settings import load_creator_settings, save_creator_settings
    from covermorph.quality_engines import gpu_used_mib

    settings = load_creator_settings(REPOSITORY_ROOT)
    settings.update(models_dir=str(args.models_dir), output_dir=str(out / "jobs"), setup_completed=True)
    save_creator_settings(REPOSITORY_ROOT, settings)
    report_path = out / "report.json"
    rows = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else []
    for name, spec in JOBS:
        if args.names and name not in args.names:
            continue
        purpose = BUILTIN_PURPOSES[spec["purpose"]]
        payload = {"kind": "generate", "quality": "balanced", "memory": LOW, "candidates": 2,
                   "canvas": [purpose.width, purpose.height], **spec}
        baseline = gpu_used_mib()
        started = time.perf_counter()
        print(f"== {name}", flush=True)
        try:
            result = execute_job(REPOSITORY_ROOT, settings, payload, Event())
        except Exception as exc:  # recorded, the run continues like the queue would
            rows = [r for r in rows if r["name"] != name] + [{"name": name, "error": getattr(exc, "stored", lambda: str(exc))()}]
            print(name, "ERROR", exc, flush=True)
            continue
        top = result["candidates"][0]
        exports = export_candidate(top, purpose, export_dir=Path(result["job_dir"]) / "export", name=name,
                                   title=payload.get("title", ""), subtitle=payload.get("subtitle", ""),
                                   cta=payload.get("cta", ""), canvas=tuple(payload["canvas"]))
        row = {"name": name, "purpose": purpose.key, "canvas": payload["canvas"], "quality": payload["quality"],
               "memory": payload["memory"], "product_mode": payload.get("product_mode", ""),
               "seconds": round(time.perf_counter() - started, 1), "job_dir": result["job_dir"],
               "gpu_baseline_mib": baseline, "gpu_after_mib": gpu_used_mib(),
               "commit_delta_mib": (result["memory_after"]["commit_total_mib"] or 0) - (result["memory_before"]["commit_total_mib"] or 0),
               "warnings": result["warnings"], "exports": exports,
               "candidates": [{"kind": c["kind"], "engine": c["engine"], "seed": c["seed"], "seconds": c["seconds"],
                               "peak_vram_mib": c["peak_vram_mib"], "warnings": c["warnings"],
                               "translated_prompt": c.get("translated_prompt"), "full": c["files"]["full"]}
                              for c in result["candidates"]]}
        rows = [r for r in rows if r["name"] != name] + [row]
        report_path.write_text(json.dumps(sorted(rows, key=lambda r: r["name"][2:]), ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({k: row[k] for k in ("name", "seconds", "gpu_baseline_mib", "gpu_after_mib", "commit_delta_mib")}
                         | {"cands": [(c["kind"], c["engine"], c["seconds"], c["peak_vram_mib"], len(c["warnings"])) for c in row["candidates"]]},
                         ensure_ascii=False), flush=True)
    report_path.write_text(json.dumps(sorted(rows, key=lambda r: r["name"][2:]), ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

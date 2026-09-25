from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from threading import Event

from PIL import Image

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from covermorph.cover_studio import (  # noqa: E402
    CoverEdit,
    TextLayer,
    export_cover,
    font_supports_text,
    save_candidate_edit,
    system_font_paths,
)
from covermorph.generation import (  # noqa: E402
    GenerationConfig,
    SDXLTextToImageEngine,
    generate_scene_candidates,
    resolve_sdxl_model_path,
)
from covermorph.planning import LlamaCppCliBackend, generate_plans, make_input_snapshot  # noqa: E402
from covermorph.project import (  # noqa: E402
    InputRecord,
    SceneCard,
    add_input_records,
    create_project,
    load_generation_presets,
    load_project,
    new_id,
    parse_input_file_detailed,
    save_project_atomic,
)


def supported_font(text: str) -> Path:
    return next(path for path in system_font_paths() if font_supports_text(path, text)[0])


def progress(prefix: str):
    def report(data: dict) -> None:
        phase = data.get("phase")
        if phase in {"candidate_start", "candidate_done", "inference", "load"}:
            print(prefix, phase, data.get("candidate", ""), data.get("step", ""), flush=True)

    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", required=True, type=Path)
    parser.add_argument("--root", default=Path(__file__).resolve().parents[1], type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    output = root / "validation_results" / "quick_cover_actual"
    output.mkdir(parents=True, exist_ok=True)
    copied_json = output / "source_copy.json"
    shutil.copy2(args.json, copied_json)
    parsed = parse_input_file_detailed(copied_json)
    records: list[InputRecord] = list(parsed["records"])
    for index, record in enumerate(records):
        record.selected = index == 0

    project = create_project(output / "resume_project", "실제 간편 커버 검증")
    add_input_records(project, records)
    model = resolve_sdxl_model_path(root, "stabilityai/stable-diffusion-xl-base-1.0")
    engine = SDXLTextToImageEngine(str(model), local_files_only=True)
    report: dict = {
        "json_source": str(args.json),
        "json_copy": str(copied_json),
        "model": str(model),
        "runs": [],
    }

    direct_prompt = "A quiet blue-hour Seoul rooftop after rain, one empty chair, cinematic photography, no people, textless square album cover"
    direct_scene = SceneCard(
        new_id("direct"),
        order=1,
        prompt_user=direct_prompt,
        negative_prompt_user="text, letters, logo, watermark",
        output_ratio="1:1",
        candidate_count=1,
        prompt_confirmed=True,
        structured_request={
            "approval_source": "quick_cover_no_approval_required",
            "original_input": "비가 그친 서울 옥상, 빈 의자 하나",
            "actual_generation_prompt": direct_prompt,
            "prompt_mode": "이미지 프롬프트 직접 사용",
            "textless_background": True,
        },
    )
    project.scenes.append(direct_scene)
    direct_config = GenerationConfig(
        model_id=str(model), candidate_count=1, seed=26092501, steps=28, local_files_only=True
    )
    started = time.perf_counter()
    direct = generate_scene_candidates(
        project, direct_scene, engine, direct_config, Event(), progress("direct")
    )
    report["runs"].append(
        {
            "name": "direct_one",
            "completed": direct.completed,
            "failed": direct.failed,
            "seconds": round(time.perf_counter() - started, 2),
            "candidate_ids": direct.candidate_ids,
        }
    )

    presets = load_generation_presets(root / "config" / "channel_generation_presets.json")
    snapshot = make_input_snapshot(
        presets[0],
        records,
        "정사각 음원 커버용 글자 없는 장면",
        {"album_title": "검증용 앨범"},
        4,
    )
    planner = LlamaCppCliBackend(root)
    try:
        plans = generate_plans(snapshot, planner, Event())
    finally:
        planner.close()
    planned_prompt = str(plans["plans"][0]["image_prompt_en"])
    json_scene = SceneCard(
        new_id("json"),
        order=2,
        prompt_user=planned_prompt,
        negative_prompt_user=str(
            plans["plans"][0].get("negative_prompt") or "text, letters, logo, watermark"
        ),
        output_ratio="1:1",
        candidate_count=4,
        prompt_confirmed=True,
        structured_request={
            "approval_source": "quick_cover_no_approval_required",
            "source_input_ids": [records[0].input_id],
            "original_input": records[0].lyrics,
            "actual_generation_prompt": planned_prompt,
            "prompt_mode": "가사/주제로 만들기",
            "textless_background": True,
        },
    )
    project.scenes.append(json_scene)
    json_config = GenerationConfig(
        model_id=str(model), candidate_count=4, seed=26092510, steps=28, local_files_only=True
    )
    started = time.perf_counter()
    generated = generate_scene_candidates(project, json_scene, engine, json_config, Event(), progress("json"))
    report["runs"].append(
        {
            "name": "json_four_sequential",
            "completed": generated.completed,
            "failed": generated.failed,
            "seconds": round(time.perf_counter() - started, 2),
            "candidate_ids": generated.candidate_ids,
            "planned_prompt": planned_prompt,
        }
    )

    candidate = next(item for item in project.candidates if item.candidate_id == generated.candidate_ids[0])
    font = str(supported_font("한글 日本語 English"))
    edit = CoverEdit(
        title=TextLayer("비가 그친 뒤", font_path=font, size=118, y=0.14),
        subtitle=TextLayer("雨上がりの夜 · After the Rain", font_path=font, size=55, y=0.31),
        label=TextLayer("CoverMorph Records", font_path=font, size=42, y=0.91),
    )
    save_candidate_edit(project, candidate, edit)
    exported = export_cover(project, candidate, edit, output, "actual_json_cover")
    save_project_atomic(project)
    reloaded = load_project(project.project_file)
    restored = next(item for item in reloaded.candidates if item.candidate_id == candidate.candidate_id)
    report.update(
        {
            "font": font,
            "font_support": font_supports_text(font, "한글 日本語 English")[0],
            "exports": {key: str(value) for key, value in exported.items()},
            "resume": {
                "candidate_count": len(reloaded.candidates),
                "selected_candidate_ids": reloaded.selected_candidate_ids,
                "title": restored.generation_metadata["cover_edit"]["title"]["text"],
            },
        }
    )
    with Image.open(exported["cover_jpg"]) as image:
        report["jpg_reopen"] = {"size": list(image.size), "mode": image.mode, "format": image.format}
    (output / "validation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw

from covermorph import backend_cli as bc


def _payload(tmp_path, **extra):
    path = tmp_path / "작업 job.json"
    path.write_text(json.dumps({"kind": "generate", "purpose": "youtube_thumbnail", "prompt": "x", "quality": "preview",
                                "seed": 1, "models_dir": str(tmp_path / "models"), "output_root": str(tmp_path / "out"),
                                **extra}, ensure_ascii=False), encoding="utf-8")
    return path


def test_waiting_when_resources_short(tmp_path, monkeypatch):
    monkeypatch.setattr("covermorph.quality_engines.check_resources", lambda policy, snap=None, engine_vram_mib=None: (False, ["GPU 여유 2000 MiB"]))
    monkeypatch.setattr("covermorph.quality_engines.resource_snapshot", lambda: None)
    result = tmp_path / "r.json"
    assert bc.run_studio_job(_payload(tmp_path), result) == bc.EXIT_WAITING
    data = json.loads(result.read_text(encoding="utf-8"))
    assert data["status"] == "waiting" and "GPU" in data["reasons"][0]


def test_failed_job_reports_plain_error(tmp_path):
    result = tmp_path / "r.json"
    assert bc.run_studio_job(_payload(tmp_path, skip_resource_check=True), result) == bc.EXIT_FAILED
    error = json.loads(result.read_text(encoding="utf-8"))["error"]
    assert error["code"] == "MODELS_DIR_INVALID" and error["action"] and "Traceback" not in error["message"]


def test_done_job_writes_candidates(tmp_path, monkeypatch):
    (tmp_path / "models").mkdir()
    monkeypatch.setattr("covermorph.errors.preflight", lambda payload, models, output: None)
    monkeypatch.setattr("covermorph.creator_jobs.run_creator_job",
                        lambda payload, cancel, models_dir, output_root, progress=None, app_root=None:
                        (progress({"phase": "stage", "message": "생성 중", "fraction": 0.5}) or
                         {"job_dir": str(output_root / "j"), "candidates": [{"files": {"full": "a.png"}}], "warnings": []}))
    result = tmp_path / "r.json"
    assert bc.run_studio_job(_payload(tmp_path, skip_resource_check=True), result) == bc.EXIT_DONE
    data = json.loads(result.read_text(encoding="utf-8"))
    assert data["status"] == "done" and data["candidates"][0]["files"]["full"] == "a.png"


def test_editor_project_sidecars_for_shopify_size(tmp_path):
    image = Image.new("RGB", (1800, 700), (200, 190, 180))
    ImageDraw.Draw(image).rectangle((1100, 200, 1400, 650), fill=(60, 80, 120))
    source = tmp_path / "후보.png"
    image.save(source)
    out = tmp_path / "편집 project"
    assert bc.make_editor_project(source, out, {"purpose": "shopify_hero", "title": "Sale", "cta": "Shop"}) == 0
    for name in ("canvas_clean.png", "subject_boxes.json", "safe_zones.json", "palette.json", "composition.json",
                 "project_manifest.json"):
        assert (out / name).is_file(), name
    assert Image.open(out / "canvas_clean.png").size == (1800, 700)
    manifest = json.loads((out / "project_manifest.json").read_text(encoding="utf-8"))
    assert manifest["purpose"] == "shopify_hero" and manifest["title"] == "Sale"

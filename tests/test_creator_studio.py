from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from covermorph import creator_jobs as cj
from covermorph import creator_presets as cp
from covermorph import creator_settings as cs
from covermorph.job_queue import CANCELLED, DONE, FAILED, PENDING, JobQueue
from covermorph.product_checks import composite_product, extract_product


# ------------------------------------------------------------------ presets / labels
def test_ui_labels_map_to_engine_modes():
    assert cp.QUALITY_LABELS == {"빠른 미리보기": "preview", "일반": "balanced", "최고 품질": "best"}
    assert cp.MEMORY_LABELS[cp.DEFAULT_MEMORY_LABEL] == "interactive_low_memory"
    assert set(cp.ROLE_LABELS.values()) == {"PERSON", "PRODUCT", "STYLE", "COMPOSITION", "BACKGROUND"}


def test_builtin_shopify_presets_and_regions():
    purposes = cp.BUILTIN_PURPOSES
    sizes = {k: (p.width, p.height) for k, p in purposes.items()}
    assert sizes["shopify_hero"] == (1800, 700) and sizes["shopify_collection"] == (1600, 900)
    assert sizes["shopify_product_lifestyle"] == (1600, 1200) and sizes["shopify_promo_tile"] == (1080, 1080)
    assert sizes["shopify_mobile"] == (1080, 1350) and sizes["youtube_thumbnail"] == (1280, 720)
    assert purposes["shopify_collection"].alternates["정사각형"] == (1200, 1200)
    for purpose in purposes.values():
        for region in (purpose.text_region, purpose.subject_region, purpose.cta_region):
            if region:
                x, y, w, h = region
                assert 0 <= x and 0 <= y and x + w <= 1.0001 and y + h <= 1.0001, purpose.key


def test_user_preset_file_overrides_and_adds(tmp_path):
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "creator_presets.json").write_text(json.dumps({
        "purposes": {"shopify_hero": {"width": 2000, "height": 800},
                     "my_theme": {"label": "내 테마", "width": 1440, "height": 600, "compiler_purpose": "shopify_hero_banner",
                                  "text_region": [0.05, 0.2, 0.4, 0.6], "subject_region": [0.5, 0.1, 0.45, 0.8]}},
        "prompt_presets": {"op_mature_couple": {"appearance_hint": "Japanese"}}}, ensure_ascii=False), encoding="utf-8")
    purposes, prompts, problems = cp.load_presets(tmp_path)
    assert problems == []
    assert (purposes["shopify_hero"].width, purposes["shopify_hero"].height) == (2000, 800)
    assert purposes["shopify_hero"].label == "Shopify 히어로 배너"
    assert purposes["my_theme"].text_region == (0.05, 0.2, 0.4, 0.6)
    assert prompts["op_mature_couple"].appearance_hint == "Japanese"


def test_appearance_hint_only_when_configured():
    tc = cp.BUILTIN_PROMPT_PRESETS["tc_solo_woman"]
    op = cp.BUILTIN_PROMPT_PRESETS["op_mature_couple"]
    assert "Japanese young woman" in cp.apply_appearance_hint(tc.prompt, tc)
    assert cp.apply_appearance_hint(op.prompt, op) == op.prompt          # OLD POP: nothing forced
    assert cp.apply_appearance_hint("a Japanese woman", tc) == "a Japanese woman"
    assert cp.apply_appearance_hint("a rainy street", None) == "a rainy street"


def test_canvas_validation():
    assert cp.validate_canvas("1800", 700) == (1800, 700)
    with pytest.raises(ValueError):
        cp.validate_canvas(100, 700)


# ------------------------------------------------------------------ settings without env vars
def test_settings_persist_and_resolve_models_dir(tmp_path, monkeypatch):
    monkeypatch.delenv("COVERMORPH_MODELS_DIR", raising=False)
    assert cs.resolve_models_dir(tmp_path) == tmp_path / "models"
    settings = cs.load_creator_settings(tmp_path)
    settings["models_dir"] = str(tmp_path / "모델 モデル")
    cs.save_creator_settings(tmp_path, settings)
    assert cs.resolve_models_dir(tmp_path) == tmp_path / "모델 モデル"
    assert cs.load_creator_settings(tmp_path)["memory_label"] == "작업 중 PC 우선"


def test_bridge_reads_models_dir_saved_in_ui(tmp_path, monkeypatch):
    from covermorph import thumbnail_bridge_runtime as runtime
    from covermorph.thumbnail_bridge import ThumbnailBridgeRequest

    monkeypatch.delenv("COVERMORPH_MODELS_DIR", raising=False)
    monkeypatch.setattr(runtime, "_app_root", lambda: tmp_path)
    cs.save_creator_settings(tmp_path, {"models_dir": str(tmp_path / "saved")})
    request = ThumbnailBridgeRequest.from_dict({"protocol_version": 1, "request_id": "r", "action": "status",
                                                "project_dir": str(tmp_path)})
    assert runtime.resolve_models_dir(request) == tmp_path / "saved"


# ------------------------------------------------------------------ names / prompts
def test_safe_name_keeps_unicode():
    assert cj.safe_name('도쿄 "비" 夜:景/테스트?') == "도쿄_비_夜_景_테스트"
    assert cj.safe_name("...") == "image"


def test_background_prompt_removes_the_product():
    text = cj.background_prompt("the product on dark polished stone", (0.5, 0.3, 0.45, 0.6))
    assert text.startswith("on dark polished stone") and "product" not in text.casefold()
    assert "lower right" in text and "Nothing stands on the surface" in text
    for word in ("photograph", "plate", "camera"):
        assert word not in text.casefold()


# ------------------------------------------------------------------ product cutout / composite
def _studio_mug(path: Path) -> Path:
    image = Image.new("RGB", (400, 300), (200, 200, 200))
    draw = ImageDraw.Draw(image)
    draw.rectangle((150, 80, 270, 240), fill=(40, 140, 150))
    draw.ellipse((95, 110, 165, 200), outline=(40, 140, 150), width=14)       # handle with a see-through hole
    draw.polygon([(210, 120), (185, 170), (235, 170)], fill=(240, 240, 240))  # logo
    image.save(path)
    return path


def test_extract_product_keeps_logo_and_handle_hole(tmp_path):
    cutout = extract_product(Image.open(_studio_mug(tmp_path / "mug.png")))
    assert cutout.width > 150 and cutout.height > 140
    rgba = cutout.convert("RGBA")
    colours = [rgba.getpixel((x, y)) for x in range(cutout.width) for y in range(cutout.height)]
    assert any(c[3] > 200 and c[0] > 230 for c in colours), "white logo kept"
    assert sum(1 for c in colours if c[3] < 20) > 500, "background / handle hole removed"


def test_composite_keeps_reference_pixels(tmp_path):
    cutout = extract_product(Image.open(_studio_mug(tmp_path / "mug.png")))
    background = Image.new("RGB", (1280, 720), (90, 60, 40))
    composite, box = composite_product(background, cutout, (0.5, 0.1, 0.45, 0.8), 0.5)
    assert composite.size == (1280, 720) and 0.5 <= box[0] < 1 and box[2] > 0.05
    x0, y0 = int(box[0] * 1280), int(box[1] * 720)
    crop = composite.crop((x0, y0, x0 + int(box[2] * 1280), y0 + int(box[3] * 720)))
    assert (40, 140, 150) in {crop.getpixel((x, y))[:3] for x in range(0, crop.width, 3) for y in range(0, crop.height, 3)}


# ------------------------------------------------------------------ jobs with fake engines
class FakeRun:
    def __init__(self):
        self.calls = []

    def __call__(self, payload, cancel=None, progress=None, backend_factory=None):
        self.calls.append(payload)
        edit = bool(payload.get("edit_image"))
        engine = "flux2_klein_4b" if edit or payload.get("references") else "zimage_turbo"
        w, h = payload.get("canvas") or (1280, 720)
        candidates = []
        for i in range(int(payload.get("max_candidates_per_engine") or 1)):
            image = Image.new("RGB", (w, h), (80 + i * 20, 70, 60))
            candidates.append({"image": image, "score": 1.0 - i * 0.1, "qa": {"problems": [], "warnings": []},
                               "manifest": {"backend": engine, "model": engine, "seed": payload["seed"] + i,
                                            "timing": {"engine_seconds": 1.0}, "peak_vram_mib": 7000,
                                            "original_prompt": payload.get("original_prompt") or payload.get("prompt"),
                                            "translated_prompt": payload.get("prompt"), "translation_method": "none",
                                            "compiled_prompt": "compiled " + str(payload.get("prompt")),
                                            "model_license": "apache-2.0", "commercial_use_flag": True,
                                            "quality_mode": payload["mode"], "memory_profile": payload["memory_profile"]}})
        return {"candidates": candidates, "warnings": [], "plan": [], "mode": payload["mode"]}


@pytest.fixture
def fake_run(monkeypatch):
    run = FakeRun()
    monkeypatch.setattr(cj, "run_quality_job", run)
    monkeypatch.setattr(cj, "apply_ocr", lambda cutout, items, workdir: False)
    return run


def test_youtube_job_writes_candidates_and_previews(tmp_path, fake_run):
    result = cj.run_creator_job({"purpose": "youtube_thumbnail", "prompt": "rainy street", "prompt_preset": "tc_solo_woman",
                                 "people": 1, "quality": "balanced", "memory": "interactive_low_memory", "seed": 7,
                                 "candidates": 2}, None, models_dir=tmp_path, output_root=tmp_path / "출력 out")
    assert len(result["candidates"]) == 2
    files = result["candidates"][0]["files"]
    assert {"full", "preview_340", "preview_180"} <= set(files)
    assert Image.open(files["preview_180"]).width == 180
    assert fake_run.calls[0]["canvas"] == [1280, 720] and fake_run.calls[0]["max_candidates_per_engine"] == 2
    assert "Japanese" in fake_run.calls[0]["prompt"]  # explicit preset hint applied
    assert Path(result["job_dir"], "job.json").exists() and Path(result["job_dir"], "result.json").exists()
    assert "memory_before" in result and "memory_after" in result


def test_product_preserve_puts_exact_composites_first(tmp_path, fake_run):
    mug = _studio_mug(tmp_path / "상품.png")
    result = cj.run_creator_job({"purpose": "shopify_product_lifestyle", "prompt": "the product on a wooden table",
                                 "references": [{"path": str(mug), "role": "PRODUCT"}], "product_preserve": True,
                                 "quality": "best", "memory": "night_best", "seed": 3, "candidates": 2},
                                None, models_dir=tmp_path, output_root=tmp_path / "out")
    kinds = [c["kind"] for c in result["candidates"]]
    assert kinds[:2] == ["composite", "composite"] and "harmonized" in kinds and "regenerated" in kinds
    composite = result["candidates"][0]
    assert composite["product_check"]["exact"] is True and composite["files"]["product"]
    regenerated = [c for c in result["candidates"] if c["kind"] != "composite"]
    assert all(c["product_check"]["distorted"] for c in regenerated)
    assert all("AI가 다시 그린 상품" in c["warnings"][0] for c in regenerated)
    assert max(c["score"] for c in regenerated) < min(c["score"] for c in result["candidates"][:2])
    background_call = fake_run.calls[0]
    assert background_call["references"] == [] and "Nothing stands" in background_call["prompt"]
    assert background_call["canvas"] == [1600, 1200]
    roles = [r["role"] for r in json.loads(Path(result["job_dir"], "job.json").read_text(encoding="utf-8"))["references"]]
    assert roles == ["PRODUCT"]


def test_preview_product_job_is_composite_only(tmp_path, fake_run):
    mug = _studio_mug(tmp_path / "mug.png")
    result = cj.run_creator_job({"purpose": "shopify_promo_tile", "prompt": "festive table",
                                 "references": [{"path": str(mug), "role": "PRODUCT"}], "product_preserve": True,
                                 "quality": "preview", "seed": 1, "candidates": 1}, None,
                                models_dir=tmp_path, output_root=tmp_path / "out")
    assert [c["kind"] for c in result["candidates"]] == ["composite"]
    assert len(fake_run.calls) == 1


def test_export_textless_and_composed(tmp_path, fake_run):
    result = cj.run_creator_job({"purpose": "shopify_hero", "prompt": "x", "quality": "preview", "seed": 1,
                                 "candidates": 1}, None, models_dir=tmp_path, output_root=tmp_path / "out")
    purpose = cp.BUILTIN_PURPOSES["shopify_hero"]
    written = cj.export_candidate(result["candidates"][0], purpose, export_dir=tmp_path / "내보내기",
                                  name="여름 세일: 50%", title="여름 세일", subtitle="오늘만", cta="지금 구매", jpg_quality=85)
    names = sorted(Path(p).name for p in written)
    assert names == ["여름_세일_50%_composed.jpg", "여름_세일_50%_composed.png",
                     "여름_세일_50%_textless.jpg", "여름_세일_50%_textless.png"]
    assert Image.open(written[0]).size == (1800, 700)
    textless = cj.export_candidate(result["candidates"][0], purpose, export_dir=tmp_path / "e2", name="a", formats=("png",))
    assert [Path(p).name for p in textless] == ["a_textless.png"]


# ------------------------------------------------------------------ runner
def _runner(tmp_path, executor, settings=None):
    from covermorph.creator_runner import StudioRunner
    queue = JobQueue(tmp_path / "studio_queue.json")
    return queue, StudioRunner(queue, executor, {"recheck_seconds": 0.05, **(settings or {})})


def _wait(predicate, timeout=5.0):
    end = time.time() + timeout
    while time.time() < end:
        if predicate():
            return True
        time.sleep(0.02)
    return False


def test_runner_pause_after_current_and_resume(tmp_path, monkeypatch):
    from covermorph import creator_runner
    monkeypatch.setattr(creator_runner, "check_resources", lambda policy, snap=None: (True, []))
    monkeypatch.setattr(creator_runner, "resource_snapshot", lambda: None)
    release = threading.Event()
    done = []

    def executor(payload, cancel, progress):
        progress({"phase": "stage", "message": "생성 중", "fraction": 0.5})
        release.wait(5)
        done.append(payload["n"])
        return {"candidates": []}

    queue, runner = _runner(tmp_path, executor)
    for n in range(3):
        queue.add({"n": n})
    runner.start()
    assert _wait(lambda: runner.runner.current is not None)
    assert runner.progress[runner.runner.current.id]["message"] == "생성 중"
    runner.pause_after_current()
    release.set()
    assert _wait(lambda: done == [0] and runner.status == "일시정지됨")
    time.sleep(0.1)
    assert done == [0] and [j.state for j in JobQueue(tmp_path / "studio_queue.json").jobs] == [DONE, PENDING, PENDING]
    runner.start()  # 재개
    assert _wait(lambda: done == [0, 1, 2])
    runner.shutdown()


def test_runner_waits_for_resources_without_starting(tmp_path, monkeypatch):
    from covermorph import creator_runner
    monkeypatch.setattr(creator_runner, "check_resources", lambda policy, snap=None: (False, ["GPU free 2000 MiB < 6500 MiB"]))
    monkeypatch.setattr(creator_runner, "resource_snapshot", lambda: None)
    ran = []
    queue, runner = _runner(tmp_path, lambda p, c, g: ran.append(p) or {})
    job = queue.add({"memory": "interactive_low_memory"})
    runner.start()
    assert _wait(lambda: creator_runner.WAITING_LABEL in runner.status)
    assert ran == [] and job.state == PENDING and "GPU free" in job.waiting_reason
    runner.shutdown()


def test_runner_idle_requirement_for_away_mode(tmp_path, monkeypatch):
    from covermorph import creator_runner
    monkeypatch.setattr(creator_runner, "check_resources", lambda policy, snap=None: (True, []))
    monkeypatch.setattr(creator_runner, "resource_snapshot", lambda: None)
    monkeypatch.setattr(creator_runner, "idle_seconds", lambda: 30.0)
    queue, runner = _runner(tmp_path, lambda p, c, g: {}, {"require_idle_minutes": 5})
    ok, reasons = runner._gate({"memory": "night_best"})
    assert not ok and "자리 비움 대기" in reasons[0]
    assert runner._gate({"memory": "interactive_low_memory"})[0] is True


def test_closing_app_returns_running_job_to_pending(tmp_path, monkeypatch):
    from covermorph import creator_runner
    monkeypatch.setattr(creator_runner, "check_resources", lambda policy, snap=None: (True, []))
    monkeypatch.setattr(creator_runner, "resource_snapshot", lambda: None)
    started = threading.Event()

    def executor(payload, cancel, progress):
        started.set()
        cancel.wait(5)
        raise RuntimeError("process killed")

    queue, runner = _runner(tmp_path, executor)
    job = queue.add({})
    runner.start()
    assert started.wait(5)
    runner.shutdown()
    assert _wait(lambda: job.state == PENDING)
    assert JobQueue(tmp_path / "studio_queue.json").jobs[0].state == PENDING


def test_queue_remove_and_retry(tmp_path):
    queue = JobQueue(tmp_path / "q.json")
    a, b, c = queue.add({}), queue.add({}), queue.add({})
    a.state, b.state = FAILED, CANCELLED
    queue.save()
    assert queue.retry(a.id) and a.state == PENDING and not queue.retry(c.id)
    assert queue.remove([b.id, c.id]) == 2 and [j.id for j in queue.jobs] == [a.id]

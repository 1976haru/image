from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest
from PIL import Image

from covermorph.job_queue import CANCELLED, DONE, FAILED, PENDING, RUNNING, JobQueue, QueueRunner
from covermorph.prompt_compilers import NO_PEOPLE, PURPOSES, compile_prompt
from covermorph.quality_engines import (
    BACKENDS,
    EngineError,
    EngineJob,
    Flux2KleinCppBackend,
    Reference,
    ResourceSnapshot,
    ZImageCppBackend,
    check_resources,
    make_backend,
)


# ------------------------------------------------------------------ queue
def test_queue_persists_and_recovers_running_job(tmp_path):
    path = tmp_path / "큐 キュー" / "queue.json"
    queue = JobQueue(path)
    first = queue.add({"prompt": "a"})
    queue.add({"prompt": "b"})
    first.state = RUNNING
    queue.save()
    reopened = JobQueue(path)
    assert [job.state for job in reopened.jobs] == [PENDING, PENDING]
    assert reopened.jobs[0].payload == {"prompt": "a"}
    assert not list(path.parent.glob(".queue_*"))


def test_pause_after_current_finishes_running_job_then_stops(tmp_path):
    queue = JobQueue(tmp_path / "q.json")
    for name in "abc":
        queue.add({"name": name})
    seen: list[str] = []

    def executor(payload, cancel):
        seen.append(payload["name"])
        queue.pause_after_current()  # user presses pause while the first job runs
        return {"ok": True}

    runner = QueueRunner(queue, executor)
    assert runner.run() == "paused"
    assert seen == ["a"]
    assert [job.state for job in queue.jobs] == [DONE, PENDING, PENDING]
    assert JobQueue(tmp_path / "q.json").paused is True

    queue.resume()
    runner.executor = lambda payload, cancel: seen.append(payload["name"]) or {}
    assert runner.run() == "empty"
    assert seen == ["a", "b", "c"]


def test_cancel_pending_and_failed_job_does_not_stop_queue(tmp_path):
    queue = JobQueue(tmp_path / "q.json")
    bad, good, dropped = queue.add({"n": 1}), queue.add({"n": 2}), queue.add({"n": 3})
    assert queue.cancel(dropped.id)
    assert not queue.cancel(dropped.id)

    def executor(payload, cancel):
        if payload["n"] == 1:
            raise RuntimeError("boom")
        return {"n": payload["n"]}

    assert QueueRunner(queue, executor).run() == "empty"
    assert (bad.state, good.state, dropped.state) == (FAILED, DONE, CANCELLED)
    assert "boom" in bad.error
    assert queue.cancel_pending() == 0


def test_low_resources_keep_job_queued(tmp_path):
    queue = JobQueue(tmp_path / "q.json")
    job = queue.add({"engine": "zimage_turbo"})
    calls = []
    runner = QueueRunner(queue, lambda p, c: calls.append(p) or {},
                         resource_gate=lambda payload: (False, ["GPU free 2000 MiB < 6500 MiB"]))
    assert runner.run(wait_seconds=0, max_waits=2) == "waiting"
    assert calls == [] and job.state == PENDING
    assert "GPU free" in JobQueue(tmp_path / "q.json").jobs[0].waiting_reason


def test_cancel_current_marks_job_cancelled(tmp_path):
    queue = JobQueue(tmp_path / "q.json")
    job = queue.add({})
    runner = QueueRunner(queue, lambda p, cancel: runner.cancel_current() or {})
    runner.run_next()
    assert job.state == CANCELLED


# ------------------------------------------------------------------ resources
def test_check_resources_thresholds():
    ok_snap = ResourceSnapshot(12288, 800, 11488, 13000, 26000, 63000)
    assert check_resources("interactive_low_memory", ok_snap) == (True, [])
    low = ResourceSnapshot(12288, 9000, 3288, 3000, 58000, 63000)
    ok, reasons = check_resources("interactive_low_memory", low)
    assert not ok and len(reasons) == 3
    ok, reasons = check_resources("night_best", ok_snap, engine_vram_mib=12000)
    assert not ok and "GPU free" in reasons[0]


# ------------------------------------------------------------------ backends
def _fake_models(tmp_path: Path, backend) -> None:
    for path in backend.required_files().values():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")


def test_zimage_command_low_memory_flags(tmp_path, monkeypatch):
    monkeypatch.setenv("COVERMORPH_ASCII_WORKDIR", str(tmp_path / "ascii"))
    backend = ZImageCppBackend(tmp_path / "モデル 모델")
    _fake_models(tmp_path, backend)
    command = backend.build_command(EngineJob(prompt="p", negative_prompt="n", seed=7), tmp_path / "o.png")
    assert "--offload-to-cpu" in command and "--diffusion-fa" in command
    assert command[command.index("--steps") + 1] == "8" and command[command.index("--cfg-scale") + 1] == "1.0"
    assert "-n" not in command  # turbo/distilled: no negative prompt dependence
    assert command[command.index("--llm") + 1].endswith("Qwen3-4B-Q8_0.gguf")


def test_zimage_rejects_references(tmp_path):
    backend = ZImageCppBackend(tmp_path)
    ref = tmp_path / "r.png"
    Image.new("RGB", (8, 8)).save(ref)
    with pytest.raises(EngineError):
        backend.generate(EngineJob(prompt="p", references=[Reference(ref, "PERSON")]))


def test_flux_stages_references_and_edit_image_first(tmp_path, monkeypatch):
    backend = Flux2KleinCppBackend(tmp_path / "models")
    _fake_models(tmp_path, backend)
    source = tmp_path / "한글 日本語"
    source.mkdir()
    person, edit = source / "person.png", source / "edit.bmp"
    Image.new("RGB", (16, 16), "red").save(person)
    Image.new("RGB", (16, 16), "blue").save(edit)
    captured = {}

    def fake_run(job, cancel, progress):
        import tempfile
        folder = Path(tempfile.mkdtemp(dir=tmp_path))
        staged = backend._stage_references(job, folder)
        captured["command"] = backend.build_command(job, folder / "o.png", staged)
        captured["roles"] = [r.role for r in job.references]
        captured["colors"] = [Image.open(p).getpixel((0, 0)) for p in staged]
        return None

    monkeypatch.setattr(backend, "_run", fake_run)
    backend.edit(EngineJob(prompt="p", edit_image=edit, references=[Reference(person, "PERSON")]))
    refs = [captured["command"][i + 1] for i, a in enumerate(captured["command"]) if a == "-r"]
    assert len(refs) == 2 and all(r.isascii() and r.endswith(".png") for r in refs)
    assert captured["roles"] == ["EDIT", "PERSON"]
    assert captured["colors"] == [(0, 0, 255), (255, 0, 0)]


def test_unknown_reference_role_and_backend():
    with pytest.raises(EngineError):
        Reference(Path("x.png"), "FACE")
    with pytest.raises(EngineError):
        make_backend("flux1_dev", Path("."))


def test_backend_licenses_are_commercial():
    for cls in BACKENDS.values():
        assert cls.info.commercial_ok, cls.info.name


# ------------------------------------------------------------------ prompt compilers
def test_compilers_keep_original_prompt_and_engine_style():
    user = "도쿄 비 오는 밤 side profile of a young woman"
    z = compile_prompt("zimage_turbo", user, "Tokyo Chill", composition="SOLO_MEDIUM", person=True)
    f = compile_prompt("flux2_klein_4b", user, "Tokyo Chill", composition="SOLO_MEDIUM", person=True)
    r = compile_prompt("realvisxl_v5", user, "Tokyo Chill", composition="SOLO_MEDIUM", person=True)
    assert z.original_prompt == f.original_prompt == r.original_prompt == user
    assert "Tokyo" in z.positive and "rainy" in z.positive
    assert z.negative == "" and f.negative == "" and "deformed face" in r.negative
    assert r.prompt_parts and r.negative_parts
    assert "left third" in z.positive and "no text" in z.positive


def test_scenery_prompts_exclude_people_only_when_asked():
    for engine in ("zimage_turbo", "flux2_klein_4b"):
        compiled = compile_prompt(engine, "rainy night street, no people", "Tokyo Chill", person=False)
        assert NO_PEOPLE in compiled.positive and "main subject" not in compiled.positive
        assert NO_PEOPLE in compile_prompt(engine, "도쿄 비 오는 밤 거리, 사람 없음", person=False).positive
        # people detection can miss words; without an explicit request nothing removes people
        assert NO_PEOPLE not in compile_prompt(engine, "rainy night street", person=False).positive


def test_v2_compilers_never_pass_cjk_script():
    """Z-Image paints CJK prompt words as lettering, so compiled prompts must be CJK-free."""
    from covermorph.prompt_translate import needs_translation

    user = "비 오는 도쿄 거리의 젊은 여성 옆모습 가나다"
    for engine in ("zimage_turbo", "flux2_klein_4b", "realvisxl_v5"):
        compiled = compile_prompt(engine, user, "Tokyo Chill", person=True)
        assert not needs_translation(compiled.positive) and "Tokyo" in compiled.positive
        assert compiled.original_prompt == user


def test_translate_prompt_fallback_and_cache(tmp_path, monkeypatch):
    from covermorph import prompt_translate as pt

    english = pt.translate_prompt("rainy street", tmp_path)
    assert english == {"english": "rainy street", "method": "none", "seconds": 0.0, "error": ""}
    fallback = pt.translate_prompt("도쿄 비 오는 밤 알수없는말", tmp_path)  # translator not installed
    assert fallback["method"] == "vocabulary" and "not installed" in fallback["error"]
    assert "Tokyo" in fallback["english"] and not pt.needs_translation(fallback["english"])

    monkeypatch.setattr(pt, "translator_ready", lambda models_dir: True)
    monkeypatch.setattr(pt, "_translate_cached", lambda text, models_dir, threads, timeout: "A rainy Tokyo night")
    assert pt.translate_prompt("도쿄 비 오는 밤", tmp_path)["english"] == "A rainy Tokyo night"

    def broken(*args):
        raise RuntimeError("translation failed")
    monkeypatch.setattr(pt, "_translate_cached", broken)
    assert pt.translate_prompt("도쿄 비 오는 밤", tmp_path)["method"] == "vocabulary"


def test_flux_reference_roles_are_explicit(tmp_path):
    refs = [Reference(tmp_path / "a.png", "person"), Reference(tmp_path / "b.png", "STYLE"),
            Reference(tmp_path / "c.png", "PRODUCT")]
    compiled = compile_prompt("flux2_klein_4b", "walking in a park", "", "shopify_hero_banner", references=refs)
    assert [r["role"] for r in compiled.reference_roles] == ["PERSON", "STYLE", "PRODUCT"]
    assert "image 1" in compiled.positive and "image 2" in compiled.positive and "image 3" in compiled.positive
    assert "do not add new claims" in compiled.positive
    json.dumps(compiled.to_dict())


def test_shopify_purposes_have_engine_friendly_sizes():
    for name, spec in PURPOSES.items():
        width, height = spec["size"]
        assert width % 16 == 0 and height % 16 == 0, name
    with pytest.raises(ValueError):
        compile_prompt("zimage_turbo", "x", purpose="instagram")


# ------------------------------------------------------------------ quality modes + bridge route
from covermorph import quality_modes as qm  # noqa: E402
from covermorph.quality_engines import BackendInfo, EngineResult  # noqa: E402


def test_plan_engines_defaults_and_fallback():
    assert qm.plan_engines("preview", has_references=False) == [("flux2_klein_4b", 1)]
    assert qm.plan_engines("balanced", has_references=False) == [("zimage_turbo", 2)]
    assert qm.plan_engines("best", has_references=False) == [("zimage_turbo", 2), ("flux2_klein_4b", 2)]
    assert qm.plan_engines("best", has_references=True) == [("flux2_klein_4b", 3), ("realvisxl_v5", 1)]
    assert qm.plan_engines("best", has_references=True, edit=True) == [("flux2_klein_4b", 3)]
    only_realvis = lambda name: name == "realvisxl_v5"  # noqa: E731
    assert qm.plan_engines("balanced", has_references=False, available=only_realvis) == [("realvisxl_v5", 2)]
    assert qm.plan_engines("balanced", has_references=False, edit=True, available=only_realvis) == []
    with pytest.raises(ValueError):
        qm.plan_engines("ultra", has_references=False)


def test_low_memory_shrinks_wide_shopify_generation():
    final, work = qm._generation_size("shopify_hero_banner", "interactive_low_memory")
    assert final == (1792, 768) and work[0] * work[1] <= qm.LOW_MEMORY_MAX_PIXELS and work[0] % 16 == 0
    assert qm._generation_size("shopify_hero_banner", "night_best") == ((1792, 768), (1792, 768))
    assert qm._generation_size("youtube_thumbnail_background", "interactive_low_memory")[1] == (1280, 720)


class FakeBackend:
    loaded: list[str] = []
    unloaded: list[str] = []
    resident: set[str] = set()

    def __init__(self, name, models_dir, ready=True):
        self.name, self.ready = name, ready
        self.info = BackendInfo(name=name, model=name.upper(), repository="fake/" + name, quantization="Q",
                                license_id="apache-2.0", commercial_ok=True, supports_t2i=True,
                                supports_single_reference=name != "zimage_turbo", supports_multi_reference=False,
                                supports_edit=name == "flux2_klein_4b", supports_transparency=False,
                                estimated_vram_mib=1, default_steps=4, default_guidance=1.0,
                                uses_negative_prompt=False)

    def status(self):
        return {"ready": self.ready}

    def generate(self, job, cancel=None, progress=None):
        # only one engine may be resident: the previous one must have been unloaded already
        assert FakeBackend.resident <= {self.name}, FakeBackend.resident
        FakeBackend.resident.add(self.name)
        FakeBackend.loaded.append(self.name)
        return EngineResult(Image.new("RGB", (job.width, job.height), (90, 100, 110)), 1.0, 5000, 4000, 100, 120)

    edit = generate

    def unload(self):
        FakeBackend.resident.discard(self.name)
        FakeBackend.unloaded.append(self.name)


@pytest.fixture
def fake_backends(monkeypatch):
    FakeBackend.loaded, FakeBackend.unloaded, FakeBackend.resident = [], [], set()
    monkeypatch.setattr(qm, "check_resources", lambda policy, snap=None, engine_vram_mib=None: (True, []))
    monkeypatch.setattr(qm, "resource_snapshot", lambda: None)
    return FakeBackend


def test_best_runs_engines_sequentially_and_returns_all_candidates(tmp_path, fake_backends):
    result = qm.run_quality_job({"models_dir": str(tmp_path), "prompt": "rainy street", "mode": "best", "seed": 5},
                                backend_factory=fake_backends)
    assert [c["manifest"]["backend"] for c in result["candidates"]].count("zimage_turbo") == 2
    assert len(result["candidates"]) == 4
    assert fake_backends.unloaded == ["zimage_turbo", "flux2_klein_4b"]
    record = result["candidates"][0]["manifest"]
    for key in ("backend", "model", "quantization", "model_license", "commercial_use_flag", "original_prompt",
                "compiled_prompt", "reference_roles", "quality_mode", "memory_profile", "timing", "peak_vram_mib",
                "system_commit_before_mib", "system_commit_after_mib"):
        assert key in record
    assert record["original_prompt"] == "rainy street" and record["memory_profile"] == "night_best"


def test_low_resources_skip_engine_with_warning(tmp_path, fake_backends, monkeypatch):
    monkeypatch.setattr(qm, "check_resources", lambda policy, snap=None, engine_vram_mib=None: (False, ["GPU low"]))
    result = qm.run_quality_job({"models_dir": str(tmp_path), "prompt": "x", "mode": "preview"},
                                backend_factory=fake_backends)
    assert result["candidates"] == [] and "GPU low" in result["warnings"][0]


@pytest.mark.v2_engines
def test_bridge_generate_routes_to_v2_and_records_manifest(monkeypatch, tmp_path, fake_backends):
    import json as _json

    from covermorph import thumbnail_bridge_runtime as runtime
    from covermorph.thumbnail_bridge import ThumbnailBridgeRequest, standard_output_paths

    monkeypatch.setattr(runtime, "_v2_ready", lambda models_dir: {"zimage_turbo": True, "flux2_klein_4b": True})
    real_run = qm.run_quality_job
    monkeypatch.setattr(qm, "run_quality_job",
                        lambda payload, cancel=None, progress=None: real_run(payload, backend_factory=fake_backends))
    project = tmp_path / "프로젝트 プロジェクト"
    project.mkdir()
    request = ThumbnailBridgeRequest.from_dict({
        "protocol_version": 1, "request_id": "v2", "action": "generate", "project_dir": str(project),
        "prompt": "도쿄 비 오는 밤 거리", "channel": "Tokyo Chill",
        "options": {"seed": 9, "models_dir": str(tmp_path)}})
    response = runtime.handle_request(request)
    assert response.ok, response.message
    manifest = _json.loads(standard_output_paths(project)["project_manifest"].read_text(encoding="utf-8"))
    assert manifest["backend"] == "zimage_turbo" and manifest["quality_mode"] == "balanced"
    assert manifest["original_prompt"] == "도쿄 비 오는 밤 거리" and "Tokyo" in manifest["compiled_prompt"]
    assert manifest["commercial_use_flag"] is True
    assert fake_backends.loaded == ["zimage_turbo"]  # bridge default: 1 candidate

    legacy = ThumbnailBridgeRequest.from_dict({
        "protocol_version": 1, "request_id": "v2b", "action": "generate", "project_dir": str(project),
        "prompt": "x", "options": {"engine": "nope", "models_dir": str(tmp_path)}})
    assert not runtime.handle_request(legacy).ok


def test_sdcpp_command_uses_ascii_model_paths(tmp_path, monkeypatch):
    from covermorph import quality_engines as qe

    monkeypatch.setenv("COVERMORPH_ASCII_WORKDIR", str(tmp_path / "ascii"))
    monkeypatch.setattr(qe, "_short_path", lambda path: None)  # force the junction route
    backend = ZImageCppBackend(tmp_path / "모델 モデル")
    _fake_models(tmp_path, backend)
    command = backend.build_command(EngineJob(prompt="p"), tmp_path / "o.png")
    for flag in ("--diffusion-model", "--vae", "--llm"):
        value = command[command.index(flag) + 1]
        assert value.isascii() and Path(value).exists(), value


def test_run_quality_job_records_translation(tmp_path, fake_backends, monkeypatch):
    from covermorph import prompt_translate as pt

    monkeypatch.setattr(pt, "translator_ready", lambda models_dir: True)
    monkeypatch.setattr(pt, "_translate_cached", lambda text, models_dir, threads, timeout: "A couple in their fifties")
    result = qm.run_quality_job({"models_dir": str(tmp_path), "prompt": "50대 부부", "mode": "preview", "people": 2},
                                backend_factory=fake_backends)
    record = result["candidates"][0]["manifest"]
    assert record["original_prompt"] == "50대 부부" and record["translated_prompt"] == "A couple in their fifties"
    assert "in their fifties" in record["compiled_prompt"] and not pt.needs_translation(record["compiled_prompt"])
    realvis = compile_prompt("realvisxl_v5", "rainy night street", "Tokyo Chill", person=False)
    assert "people positioned" not in realvis.positive

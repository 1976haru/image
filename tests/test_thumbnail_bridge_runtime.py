"""Bridge runtime tests. Every generation here uses an injected fake engine (mock, never real AI)."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

import app
from covermorph import thumbnail_bridge_runtime as runtime
from covermorph.generation import GenerationError
from covermorph.thumbnail_bridge import ThumbnailBridgeRequest, standard_output_paths
from covermorph.thumbnail_bridge_ai import (
    MEMORY_PROFILES,
    BridgeActionError,
    build_prompt_plan,
    generate_with_memory_fallback,
    parse_edit_instruction,
    select_memory_profile,
)
from covermorph.thumbnail_bridge_assets import STAGING_PREFIX, commit_outputs, stage_outputs

REPO = Path(__file__).resolve().parents[1]
READY_ENV = {"status": "ready", "reference_status": "ready", "gpu": "Fake GPU", "vram_bytes": 12 * 1024**3,
             "cuda": True, "torch": "fake", "diffusers": "fake"}
MISSING_ENV = {"status": "model_missing", "reference_status": "model_missing", "gpu": "Fake GPU",
               "vram_bytes": 12 * 1024**3, "cuda": True, "model": {"failure_reason": "model directory is missing"}}


class FakeOOM(RuntimeError):
    pass


FakeOOM.__name__ = "OutOfMemoryError"


class FakeEngine:
    """Stand-in for SDXLTextToImageEngine: paints a scene with two face-free 'people' blocks."""

    calls: list[str] = []
    fail_profiles: set[str] = set()
    fail_with: type[BaseException] = FakeOOM

    def __init__(self, profile: str):
        self.profile = profile
        self.pipeline = None
        self.last_generation_metrics = {}
        self.last_reference_applied = False
        self.unloaded = False

    def load(self, progress=None):
        self.pipeline = type("Pipe", (), {"tokenizer": None})()

    def generate_one(self, prompt, negative_prompt, config, seed, cancel_event, progress=None, reference_image=None):
        FakeEngine.calls.append(self.profile)
        if self.profile in FakeEngine.fail_profiles:
            message = "CUDA out of memory" if FakeEngine.fail_with is FakeOOM else "scheduler exploded"
            raise GenerationError("generation failed") from FakeEngine.fail_with(message)
        width, height = config.size
        image = Image.new("RGB", (width, height), (40, 52, 70))
        draw = ImageDraw.Draw(image)
        draw.rectangle((int(width * 0.62), int(height * 0.2), int(width * 0.9), height), fill=(180, 140, 120))
        for x in range(0, int(width * 0.6), 24):
            draw.line((x, 0, x + 40, height), fill=(60, 74, 96), width=3)
        self.last_generation_metrics = {"generation_time_seconds": 0.01, "peak_memory_allocated": None,
                                        "peak_memory_reserved": None, "cuda_out_of_memory": False}
        self.last_reference_applied = reference_image is not None and config.reference_mode != "off"
        return image

    def unload(self):
        self.unloaded = True


@pytest.fixture(autouse=True)
def reset_fake_engine():
    FakeEngine.calls = []
    FakeEngine.fail_profiles = set()
    FakeEngine.fail_with = FakeOOM
    yield


@pytest.fixture
def fake_ai(monkeypatch):
    monkeypatch.setattr(runtime, "_environment", lambda request, model_id=None: (dict(READY_ENV), Path("models"), "fake-sdxl"))
    monkeypatch.setattr(runtime, "_engine_factory", lambda *args, **kwargs: FakeEngine)
    return FakeEngine


@pytest.fixture
def project(tmp_path):
    folder = tmp_path / "브리지 프로젝트 東京 001"
    folder.mkdir()
    return folder


def _request(project: Path, action: str = "generate", **extra) -> ThumbnailBridgeRequest:
    data = {"protocol_version": 1, "request_id": f"{action}-1", "action": action, "project_dir": str(project),
            "channel": "Tokyo Chill", "title": "目が合っただけなのに", "prompt": "rainy Tokyo station, 東京 夜景",
            "options": {"ratio": "16:9", "width": 1280, "height": 720, "prefer_text_space": "left", "seed": 7}}
    data.update(extra)
    return ThumbnailBridgeRequest.from_dict(data)


def _snapshot(project: Path) -> dict[str, bytes]:
    return {path.name: path.read_bytes() for path in standard_output_paths(project).values() if path.is_file()}


def _in_unit_interval(record: dict) -> bool:
    return all(0.0 <= float(record[key]) <= 1.0 for key in ("x", "y", "w", "h")) and \
        record["x"] + record["w"] <= 1.0001 and record["y"] + record["h"] <= 1.0001


# ------------------------------------------------------------------ dispatch / CLI
@pytest.mark.parametrize("argv", [["--thumbnail-bridge-json"], ["--image-bridge"], ["--action", "status", "--project-dir", "x"]])
def test_bridge_flags_dispatch_headless(monkeypatch, argv):
    seen = {}
    monkeypatch.setattr(sys, "argv", ["app.py", *argv])
    monkeypatch.setattr(runtime, "run_bridge_cli", lambda argv=None, stdin_text=None: seen.setdefault("argv", argv) and 0)
    monkeypatch.setattr(app, "_run_gui", lambda: pytest.fail("GUI must not start in bridge mode"))
    app.main()
    assert seen["argv"] == argv


def test_normal_launch_still_runs_gui(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["app.py"])
    monkeypatch.setattr(app, "_run_gui", lambda: 123)
    assert app.main() == 123


def test_stdout_is_single_json_even_when_handler_prints(monkeypatch, capfd, project):
    def noisy(request):
        print("python noise")
        os.write(1, b"native noise\n")
        return runtime.ThumbnailBridgeResponse(request_id=request.request_id, action="status",
                                               project_dir=request.project_dir, ok=True, message="정상")

    monkeypatch.setattr(runtime, "handle_request", noisy)
    code = runtime.run_bridge_cli(json.dumps({"protocol_version": 1, "action": "status", "project_dir": str(project)}),
                                  argv=["--thumbnail-bridge-json"])
    out, err = capfd.readouterr()
    assert code == 0
    assert json.loads(out)["message"] == "정상"
    assert "python noise" in err and "native noise" in err


def test_invalid_request_is_json_and_nonzero(capfd):
    code = runtime.run_bridge_cli("{bad", argv=["--thumbnail-bridge-json"])
    payload = json.loads(capfd.readouterr().out)
    assert code != 0 and payload["ok"] is False and payload["error_code"] == "INVALID_REQUEST"


def test_subprocess_stdout_is_json_only_for_unicode_path(project):
    request = {"protocol_version": 1, "request_id": "sub-1", "action": "edit", "project_dir": str(project),
               "edit_instruction": "왼쪽 문구 공간을 넓혀줘"}
    env = {key: value for key, value in os.environ.items() if key not in ("PYTHONIOENCODING", "PYTHONUTF8")}
    done = subprocess.run([sys.executable, str(REPO / "app.py"), "--thumbnail-bridge-json"], cwd=REPO, env=env,
                          input=json.dumps(request, ensure_ascii=False).encode("utf-8"), capture_output=True, timeout=300)
    payload = json.loads(done.stdout.decode("utf-8"))
    assert project.name in done.stderr.decode("utf-8")  # logs are UTF-8 even on a cp949 console
    assert done.returncode != 0
    assert payload["error_code"] == "NO_SOURCE_CANVAS" and payload["project_dir"] == str(project)


def test_youtubesum_argv_form_maps_metadata():
    request = runtime.parse_argv_request([
        "--action", "generate", "--project-dir", r"D:\일본 칠리랩\男_001", "--channel", "Tokyo Chill",
        "--title", "目が合っただけなのに", "--prompt", "rainy station",
        "--options-json", json.dumps({"episode": "EP.001", "preferred_typography": "Japanese Impact", "seed": 3})])
    assert request.action == "generate" and request.project_dir.endswith("男_001")
    assert request.episode == "EP.001" and request.preferred_typography == "Japanese Impact"
    assert request.options == {"seed": 3} and request.request_id.startswith("youtubesum-")


def test_youtubesum_json_stdin_form_maps_to_contract():
    payload = {"protocol": "youtubesum-image-bridge/1", "action": "edit", "project_dir": r"D:\프로젝트",
               "edit_instruction": "배경을 어둡게", "title": "제목", "options": {"episode": "EP.2"},
               "outputs": ["canvas_clean.png"]}
    request = runtime.parse_request(["--image-bridge"], json.dumps(payload, ensure_ascii=False))
    assert request.protocol_version == 1 and request.edit_instruction == "배경을 어둡게" and request.episode == "EP.2"


# ------------------------------------------------------------------ status
def test_status_structure(monkeypatch, project):
    monkeypatch.setattr(runtime, "_environment", lambda request, model_id=None: (dict(READY_ENV), Path("models"), "fake-sdxl"))
    response = runtime.handle_request(_request(project, "status"))
    payload = response.outputs["status"]
    assert response.ok and payload["project_dir_writable"]
    assert payload["capabilities"]["edit"] == ["recompose", "regenerate", "reference_regenerate"]
    assert payload["runtime_profile"]["memory_profile"] == "balanced"
    assert payload["runtime_profile"]["generation_working_size"] == [1344, 768]


def test_status_reports_only_recompose_when_model_missing(monkeypatch, project):
    monkeypatch.setattr(runtime, "_environment", lambda request, model_id=None: (dict(MISSING_ENV), Path("models"), "fake-sdxl"))
    payload = runtime.handle_request(_request(project, "status")).outputs["status"]
    assert payload["capabilities"]["edit"] == ["recompose"] and payload["capabilities"]["generate"] is False


# ------------------------------------------------------------------ generate (mock engine)
def test_generate_mock_writes_all_outputs_and_schemas(fake_ai, project):
    response = runtime.handle_request(_request(project))
    assert response.ok, response.message
    paths = standard_output_paths(project)
    with Image.open(paths["canvas_clean"]) as canvas:
        assert canvas.size == (1280, 720)
    with Image.open(paths["preview_reference"]) as preview:
        assert preview.size == (1344, 768)
    generation = json.loads(paths["project_manifest"].read_text(encoding="utf-8"))["generation"]
    assert generation["real_ai"] is False and generation["engine"] == "FakeEngine"
    assert generation["working_size"] == [1344, 768] and generation["memory_profile"] == "balanced"
    safe = json.loads(paths["safe_zones"].read_text(encoding="utf-8"))
    assert safe["preferred_text_regions"] and all(_in_unit_interval(item) for item in safe["preferred_text_regions"])
    assert all(_in_unit_interval(item) for item in safe["avoid_regions"])
    assert safe["preferred_text_regions"][0]["score"] >= safe["preferred_text_regions"][-1]["score"]
    subjects = json.loads(paths["subject_boxes"].read_text(encoding="utf-8"))["subjects"]
    assert subjects and all(_in_unit_interval(item) for item in subjects)
    assert {item["role"] for item in subjects} <= {"protagonist", "counterpart", "other"}
    palette = json.loads(paths["palette"].read_text(encoding="utf-8"))
    for key in ("dominant", "accent", "recommended_text_light", "recommended_text_dark", "recommended_stroke",
                "recommended_shadow", "fill_color", "stroke_color", "highlight_color"):
        assert key in palette
    assert all(color.startswith("#") and len(color) == 7 for color in palette["dominant"])
    composition = json.loads(paths["composition"].read_text(encoding="utf-8"))
    assert composition["source_generation_size"] == {"width": 1344, "height": 768}
    assert composition["final_size"] == {"width": 1280, "height": 720}
    assert set(composition["positions"]) >= {"main_title", "subtitle"}
    assert not list(project.glob(f"{STAGING_PREFIX}*"))


def test_generate_prompt_is_textless_and_excludes_title(fake_ai, project):
    plan = build_prompt_plan("rainy Tokyo station, 東京 夜景", "Tokyo Chill", "left", True)
    prompt = ", ".join(plan.prompt_parts)
    assert "textless" in prompt and "night cityscape" in prompt and "目が合" not in prompt
    assert "text" in plan.negative_prompt and "watermark" in plan.negative_prompt and "logo" in plan.negative_prompt


def test_generate_rejects_non_16x9(fake_ai, project):
    response = runtime.handle_request(_request(project, options={"ratio": "1:1"}))
    assert not response.ok and response.error_code == "UNSUPPORTED_OPTION"


def test_generate_model_not_ready_is_structured_and_writes_nothing(monkeypatch, project):
    monkeypatch.setattr(runtime, "_environment", lambda request, model_id=None: (dict(MISSING_ENV), Path("models"), "fake-sdxl"))
    response = runtime.handle_request(_request(project))
    assert not response.ok and response.error_code == "MODEL_NOT_READY"
    assert not standard_output_paths(project)["canvas_clean"].exists()


def test_failed_generate_preserves_previous_outputs(fake_ai, project):
    assert runtime.handle_request(_request(project)).ok
    before = _snapshot(project)
    FakeEngine.fail_profiles = set(MEMORY_PROFILES)
    FakeEngine.fail_with = ValueError  # not an OOM -> no retry
    response = runtime.handle_request(_request(project))
    assert not response.ok and response.error_code == "GENERATION_FAILED"
    assert _snapshot(project) == before
    assert not list(project.glob(f"{STAGING_PREFIX}*"))


# ------------------------------------------------------------------ OOM retry policy
def test_oom_retries_exactly_once_with_lower_profile():
    FakeEngine.fail_profiles = {"balanced"}
    warnings: list[str] = []
    plan = build_prompt_plan("scene", "", "left", True)
    image, metrics = generate_with_memory_fallback(FakeEngine, "balanced", plan, {"steps": 4}, 1, warnings)
    assert FakeEngine.calls == ["balanced", "conservative"]
    assert metrics["oom_retry"] is True and metrics["memory_profile"] == "conservative"
    assert image.size == tuple(MEMORY_PROFILES["conservative"]["working_size"]) and warnings


def test_oom_never_loops():
    FakeEngine.fail_profiles = set(MEMORY_PROFILES)
    with pytest.raises(BridgeActionError) as caught:
        generate_with_memory_fallback(FakeEngine, "standard", build_prompt_plan("s", "", "left", True), {"steps": 4}, 1, [])
    assert caught.value.code == "CUDA_OUT_OF_MEMORY"
    assert FakeEngine.calls == ["standard", "balanced"]


def test_bottom_edge_face_hits_are_ignored(monkeypatch):
    from covermorph import thumbnail_bridge_assets as assets

    class Cascade:
        def empty(self):
            return False

        def detectMultiScale(self, pixels, **kwargs):
            height, width = pixels.shape
            return [(int(width * 0.5), int(height * 0.88), int(width * 0.06), int(height * 0.1)),
                    (int(width * 0.2), int(height * 0.2), int(width * 0.1), int(height * 0.18))]

    monkeypatch.setattr(assets, "_FACE_MODEL", None)  # exercise the Haar path
    monkeypatch.setattr(assets, "_face_cascade", lambda name="": Cascade())
    faces = assets.detect_faces(Image.new("RGB", (1280, 720)))
    assert faces and all(face[1] + face[3] / 2 <= 0.85 for face in faces)


def test_memory_profile_thresholds():
    assert select_memory_profile(24 * 1024**3) == "standard"
    assert select_memory_profile(12 * 1024**3) == "balanced"
    assert select_memory_profile(8 * 1024**3) == "balanced"
    assert select_memory_profile(6 * 1024**3) == "conservative"
    assert select_memory_profile(None) == "conservative"
    assert select_memory_profile(6 * 1024**3, "standard") == "standard"


# ------------------------------------------------------------------ edit routing
@pytest.mark.parametrize("instruction,level", [
    ("인물을 오른쪽으로 조금 이동하고 왼쪽 문구 공간을 넓혀줘", "recompose"),
    ("배경을 조금 더 어둡게 하고 얼굴은 그대로 유지", "recompose"),
    ("왼쪽 40%를 문구 공간으로 단순하게 만들어줘", "recompose"),
    ("도쿄 야경으로 바꾸되 두 사람의 포즈는 유지", "reference_regenerate"),
    ("背景を東京の夜景に変えて", "regenerate"),
    ("regenerate the background", "regenerate"),
])
def test_edit_capability_routing(instruction, level):
    assert parse_edit_instruction(instruction).level == level


@pytest.mark.parametrize("instruction", ["여자 옆에 고양이를 추가해줘", "남자 표정을 웃는 얼굴로 바꿔줘", "제목 글자를 넣어줘",
                                         "좀 더 감성적으로", "remove the umbrella"])
def test_unsupported_edits_are_refused(fake_ai, project, instruction):
    assert runtime.handle_request(_request(project)).ok
    before = _snapshot(project)
    response = runtime.handle_request(_request(project, "edit", edit_instruction=instruction))
    assert not response.ok and response.error_code == "UNSUPPORTED_EDIT"
    assert response.details["available_edit_levels"]
    assert _snapshot(project) == before


def test_regenerate_edit_is_unsupported_without_model(monkeypatch, fake_ai, project):
    assert runtime.handle_request(_request(project)).ok
    monkeypatch.setattr(runtime, "_environment", lambda request, model_id=None: (dict(MISSING_ENV), Path("models"), "fake-sdxl"))
    response = runtime.handle_request(_request(project, "edit", edit_instruction="도쿄 야경으로 배경을 바꿔줘"))
    assert not response.ok and response.error_code == "UNSUPPORTED_EDIT"


def test_recompose_darken_keeps_people_and_darkens_background(fake_ai, project):
    assert runtime.handle_request(_request(project)).ok
    paths = standard_output_paths(project)
    before = np.asarray(Image.open(paths["canvas_clean"]).convert("L"), dtype=float)
    subject = json.loads(paths["subject_boxes"].read_text(encoding="utf-8"))["subjects"][0]
    response = runtime.handle_request(_request(project, "edit", edit_instruction="배경만 조금 어둡게 해줘"))
    assert response.ok, response.message
    after = np.asarray(Image.open(paths["canvas_clean"]).convert("L"), dtype=float)
    x0, y0 = int((subject["x"] + subject["w"] * 0.3) * 1280), int((subject["y"] + subject["h"] * 0.3) * 720)
    x1, y1 = int((subject["x"] + subject["w"] * 0.7) * 1280), int((subject["y"] + subject["h"] * 0.7) * 720)
    assert abs(after[y0:y1, x0:x1].mean() - before[y0:y1, x0:x1].mean()) < 2.0
    assert after[:, :200].mean() < before[:, :200].mean() - 3.0
    manifest = json.loads(paths["project_manifest"].read_text(encoding="utf-8"))
    assert manifest["edit_history"][-1]["level"] == "recompose" and manifest["title"] == "目が合っただけなのに"
    with Image.open(paths["preview_reference"]) as preview:
        assert np.array_equal(np.asarray(preview.convert("L"), dtype=float), before)


def test_recompose_moves_subject_and_widens_left_space(fake_ai, project):
    assert runtime.handle_request(_request(project)).ok
    paths = standard_output_paths(project)
    first = json.loads(paths["subject_boxes"].read_text(encoding="utf-8"))["subjects"][0]
    response = runtime.handle_request(_request(project, "edit", edit_instruction="인물을 오른쪽으로 조금 이동하고 왼쪽 문구 공간을 넓혀줘"))
    assert response.ok, response.message
    moved = json.loads(paths["subject_boxes"].read_text(encoding="utf-8"))["subjects"][0]
    assert moved["x"] > first["x"]
    regions = json.loads(paths["safe_zones"].read_text(encoding="utf-8"))["preferred_text_regions"]
    assert any(region["name"].startswith("left") for region in regions)


def test_reference_regenerate_refuses_when_people_not_located(fake_ai, project):
    assert runtime.handle_request(_request(project)).ok  # fake scene has no detectable face -> fallback box
    before = _snapshot(project)
    response = runtime.handle_request(_request(project, "edit", edit_instruction="두 사람은 그대로 두고 도쿄 야경으로 바꿔줘"))
    assert not response.ok and response.error_code == "UNSUPPORTED_EDIT"
    assert _snapshot(project) == before


def test_reference_regenerate_uses_ip_adapter_reference(fake_ai, project):
    assert runtime.handle_request(_request(project)).ok
    boxes = standard_output_paths(project)["subject_boxes"]
    data = json.loads(boxes.read_text(encoding="utf-8"))
    data["subjects"][0].update(source="user", face={"x": 0.7, "y": 0.25, "w": 0.1, "h": 0.18})
    boxes.write_text(json.dumps(data), encoding="utf-8")
    response = runtime.handle_request(_request(project, "edit", edit_instruction="두 사람은 그대로 두고 도쿄 야경으로 바꿔줘"))
    assert response.ok, response.message
    generation = response.details["generation"]
    assert generation["reference_mode"] == "person" and generation["reference_applied"] is True
    assert generation["real_ai"] is False
    assert any("IP-Adapter" in warning for warning in response.warnings)


# ------------------------------------------------------------------ atomic commit
def _write_project(project: Path, color) -> None:
    sidecars = {key: {"version": 1, "marker": str(color)} for key in
                ("subject_boxes", "safe_zones", "palette", "composition", "project_manifest")}
    commit_outputs(project, stage_outputs(project, Image.new("RGB", (1280, 720), color), sidecars))


def test_atomic_commit_replaces_all_outputs(project):
    _write_project(project, (10, 10, 10))
    _write_project(project, (200, 10, 10))
    with Image.open(project / "canvas_clean.png") as canvas:
        assert canvas.getpixel((5, 5)) == (200, 10, 10)
    assert json.loads((project / "palette.json").read_text(encoding="utf-8"))["marker"] == str((200, 10, 10))
    assert not list(project.glob(f"{STAGING_PREFIX}*"))


def test_commit_rolls_back_on_mid_replace_failure(project):
    _write_project(project, (10, 10, 10))
    before = _snapshot(project)
    staging = stage_outputs(project, Image.new("RGB", (1280, 720), (0, 200, 0)),
                            {key: {"version": 2} for key in ("subject_boxes", "safe_zones", "palette", "composition",
                                                             "project_manifest")})
    calls = {"count": 0}

    def flaky_replace(src, dst):
        calls["count"] += 1
        if calls["count"] == 5:
            raise OSError("disk full")
        os.replace(src, dst)

    with pytest.raises(OSError):
        commit_outputs(project, staging, replace=flaky_replace)
    assert _snapshot(project) == before
    assert not staging.exists()


def test_stage_rejects_wrong_canvas_size(project):
    from covermorph.thumbnail_bridge_assets import BridgeOutputError, validate_staged

    staging = stage_outputs(project, Image.new("RGB", (1280, 720)), {key: {"version": 1} for key in
                            ("subject_boxes", "safe_zones", "palette", "composition", "project_manifest")})
    with pytest.raises(BridgeOutputError):
        validate_staged(staging, (1920, 1080))


# ------------------------------------------------------------------ packaged bridge (optional)
@pytest.mark.skipif(not os.environ.get("COVERMORPH_BRIDGE_EXE"), reason="set COVERMORPH_BRIDGE_EXE to test a packaged build")
def test_packaged_bridge_status_unicode_path(project):
    request = {"protocol_version": 1, "request_id": "exe-status", "action": "status", "project_dir": str(project)}
    done = subprocess.run([os.environ["COVERMORPH_BRIDGE_EXE"], "--thumbnail-bridge-json"],
                          input=json.dumps(request, ensure_ascii=False).encode("utf-8"), capture_output=True, timeout=600)
    payload = json.loads(done.stdout.decode("utf-8"))
    assert done.returncode == 0 and payload["ok"] and payload["request_id"] == "exe-status"


@pytest.mark.optional_ai
@pytest.mark.skipif(not os.environ.get("COVERMORPH_REAL_AI"), reason="set COVERMORPH_REAL_AI=1 to run real SDXL on CUDA")
def test_optional_real_sdxl_bridge_generate(project):
    request = _request(project, options={"ratio": "16:9", "prefer_text_space": "left", "seed": 20261003, "steps": 20})
    response = runtime.handle_request(request)
    assert response.ok, response.message
    generation = response.details["generation"]
    assert generation["real_ai"] is True and generation["peak_memory_allocated"]

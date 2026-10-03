"""Person-quality unit tests. Engines here are fakes: these never count as real AI quality evidence."""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFilter

from covermorph import person_quality as pq
from covermorph import thumbnail_bridge_runtime as runtime
from covermorph.thumbnail_bridge import ThumbnailBridgeRequest, standard_output_paths
from covermorph.thumbnail_bridge_ai import build_prompt_plan, fit_prompt

READY_ENV = {"status": "ready", "reference_status": "ready", "gpu": "Fake GPU", "vram_bytes": 12 * 1024**3}
REAL_YUNET = Path(os.environ.get("COVERMORPH_MODELS_DIR") or "models") / "face_detection" / pq.YUNET_FILENAME


def _textured(size=(1344, 768), seed=0) -> Image.Image:
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 255, (size[1], size[0], 3), dtype=np.uint8))


def _face(box, confidence=0.9):
    return {"box": box, "confidence": confidence, "landmarks": [(box[0] + box[2] / 2, box[1] + box[3] / 2)] * 5}


# ------------------------------------------------------------------ selection / prompts
@pytest.mark.parametrize("prompt,expected", [
    ("a young couple at a cafe window", 2), ("solo woman on an evening street", 1), ("man on a train platform", 1),
    ("two people walking in the rain", 2), ("empty rainy Tokyo street at night", 0), ("mature woman in first snow", 1),
])
def test_people_count(prompt, expected):
    assert pq.people_count(prompt) == expected


def test_candidate_a_frames_people_larger_than_b_and_c():
    for people in (1, 2):
        heights = {c: pq.COMPOSITION_PROFILES[pq.choose_composition(people, c)]["min_face_height"] for c in "ABC"}
        assert heights["A"] >= heights["B"] >= heights["C"] and heights["A"] > heights["C"]
    assert pq.choose_composition(0) == ""
    assert pq.choose_composition(2, "A", "COUPLE_MEDIUM") == "COUPLE_MEDIUM"


def test_person_prompt_keeps_subject_framing_quality_and_negatives():
    plan = build_prompt_plan("young couple by a cafe window", "OLD POP LOUNGE", "left", True, person=True,
                             framing=pq.COMPOSITION_PROFILES["COUPLE_MEDIUM"]["framing"])
    joined = ", ".join(plan.prompt_parts)
    assert plan.prompt_parts[0].startswith("young couple") and "both faces clearly visible" in joined
    assert "natural skin texture" in joined and "textless photograph" in joined
    for bad in ("waxy skin", "extra fingers", "asymmetrical eyes", "watermark", "duplicate person"):
        assert bad in plan.negative_prompt


def test_negative_prompt_priority_survives_token_limit():
    class Tokenizer:  # crude 1 token per word + 2 specials
        model_max_length = 30

        def __call__(self, text, truncation=False, add_special_tokens=True):
            return {"input_ids": [0] * (len(text.replace(",", " ").split()) + 2)}

    fitted = fit_prompt(list(pq.PERSON_NEGATIVE_PARTS), Tokenizer(), [])
    assert fitted.startswith("text, watermark") and "deformed face" in fitted and "jpeg artifacts" not in fitted


def test_model_selection_prefers_photoreal_for_people_and_falls_back(tmp_path):
    warnings: list[str] = []
    name, path, profile = runtime._select_model({}, tmp_path, True, warnings)
    assert name == "sdxl_base" and warnings and "photoreal_sdxl" in warnings[0]
    name, _, _ = runtime._select_model({}, tmp_path, False, [])
    assert name == "sdxl_base"
    assert runtime._select_model({"model_id": "x"}, tmp_path, True, [])[0] == "custom"


def test_photoreal_ready_requires_fp16_unet(tmp_path):
    folder = tmp_path / "realvisxl_v5.0"
    for sub in ("unet", "vae", "text_encoder", "text_encoder_2", "tokenizer", "tokenizer_2", "scheduler"):
        (folder / sub).mkdir(parents=True)
    (folder / "model_index.json").write_text("{}", encoding="utf-8")
    for sub in ("vae", "text_encoder", "text_encoder_2"):
        (folder / sub / "model.fp16.safetensors").write_bytes(b"x")
    (folder / "unet" / "other.safetensors").write_bytes(b"x")
    assert not pq.inspect_photoreal_model(folder)["ready"]
    (folder / "unet" / "diffusion_pytorch_model.fp16.safetensors").write_bytes(b"x")
    assert pq.inspect_photoreal_model(folder)["ready"]


# ------------------------------------------------------------------ QA gate
def test_quality_check_flags_missing_small_and_soft_faces():
    sharp = _textured()
    soft = sharp.filter(ImageFilter.GaussianBlur(8))
    big = _face((0.4, 0.2, 0.12, 0.25))
    assert pq.quality_check(sharp, [big], 1, 0.18)["passed"]
    assert "expected 2 face(s), found 1" in pq.quality_check(sharp, [big], 2, 0.1)["problems"][0]
    assert any("smaller" in p for p in pq.quality_check(sharp, [_face((0.4, 0.4, 0.04, 0.06))], 1, 0.18)["problems"])
    assert any("soft" in p for p in pq.quality_check(soft, [big], 1, 0.18)["problems"])
    dark = Image.new("RGB", (1344, 768), (2, 2, 2))
    assert any("exposure" in p for p in pq.quality_check(dark, [], 0, 0)["problems"])
    assert any("possible duplicate" in w for w in pq.quality_check(sharp, [big] * 4, 1, 0.1)["warnings"])


def test_quality_check_never_judges_attractiveness_only_technical_fields():
    report = pq.quality_check(_textured(), [_face((0.4, 0.2, 0.12, 0.25))], 1, 0.1)
    assert set(report) == {"passed", "problems", "warnings", "faces", "exposure", "score"}


class FakeEngine:
    def __init__(self, images):
        self.images = list(images)
        self.calls = 0
        self.last_generation_metrics = {"generation_time_seconds": 0.1}
        self.ip_adapter_loaded = False

    def generate_one(self, prompt, negative, config, seed, cancel_event, reference_image=None):
        self.calls += 1
        return self.images.pop(0)


def test_quality_mode_regenerates_once_and_keeps_the_better_image(monkeypatch):
    bad, good = _textured(seed=1), _textured(seed=2)
    detections = {id(bad): [_face((0.4, 0.4, 0.03, 0.05))], id(good): [_face((0.4, 0.2, 0.12, 0.25))]}
    monkeypatch.setattr(pq, "detect_faces_yunet", lambda image, model, min_score=0.6: detections.get(id(image), []))
    engine = FakeEngine([good])
    image, info = pq.run_person_pass(engine, bad, config=None, mode="quality", prompt="p", negative_prompt="n", seed=5,
                                     expected_people=1, composition="SOLO_CLOSE", model_path=Path("yunet.onnx"),
                                     face_detail=False)
    assert engine.calls == 1 and info["regenerated"] and image is good and info["seed_used"] != 5


def test_regenerate_never_loops_even_if_both_fail(monkeypatch):
    first, second = _textured(seed=1), _textured(seed=2)
    monkeypatch.setattr(pq, "detect_faces_yunet", lambda image, model, min_score=0.6: [])
    engine = FakeEngine([second])
    image, info = pq.run_person_pass(engine, first, config=None, mode="quality", prompt="p", negative_prompt="n", seed=5,
                                     expected_people=2, composition="COUPLE_MEDIUM", model_path=Path("y"), face_detail=False)
    assert engine.calls == 1 and len(info["qa_attempts"]) == 2 and not info["qa"]["passed"]


def test_fast_mode_is_single_pass(monkeypatch):
    monkeypatch.setattr(pq, "detect_faces_yunet", lambda image, model, min_score=0.6: [])
    engine = FakeEngine([])
    image, info = pq.run_person_pass(engine, _textured(), config=None, mode="fast", prompt="p", negative_prompt="n",
                                     seed=5, expected_people=1, composition="SOLO_CLOSE", model_path=Path("y"))
    assert engine.calls == 0 and not info["regenerated"] and not info["face_detail_applied"]


# ------------------------------------------------------------------ face detail safety
def test_face_detail_rejects_softer_or_drifted_refinements(monkeypatch):
    crop = _textured((400, 400))
    ok, reason = pq._same_face(crop, crop.filter(ImageFilter.GaussianBlur(3)), None)
    assert not ok and "softer" in reason
    calls = iter([[_face((0.3, 0.3, 0.4, 0.4))], [_face((0.5, 0.5, 0.4, 0.4))]])
    monkeypatch.setattr(pq, "detect_faces_yunet", lambda image, model, min_score=0.5: next(calls))
    ok, reason = pq._same_face(crop, crop.copy(), Path("y"))
    assert not ok and "drifted" in reason


def test_face_detail_crop_keeps_hair_and_stays_inside_image():
    left, top, right, bottom = pq._crop_box((0.45, 0.02, 0.1, 0.2), (1344, 768), 2.0)
    assert 0 <= left < right <= 1344 and 0 <= top < bottom <= 768
    assert right - left > 0.1 * 1344 * 1.5
    assert pq._work_size((300, 200), 1024) == (1024, 680)


def test_finish_is_restrained():
    image = _textured((640, 360)).filter(ImageFilter.GaussianBlur(2))
    finished = np.asarray(pq.finish_image(image), dtype=float)
    original = np.asarray(image, dtype=float)
    assert abs(finished.mean() - original.mean()) < 1.0
    assert np.abs(finished.mean(axis=(0, 1)) - original.mean(axis=(0, 1))).max() < 1.0  # no colour cast


# ------------------------------------------------------------------ YuNet / manifest
@pytest.mark.skipif(not REAL_YUNET.is_file(), reason="YuNet model not prepared (scripts/prepare_person_quality_models.py)")
def test_yunet_loads_from_korean_japanese_space_path(tmp_path):
    folder = tmp_path / "얼굴 검출 顔検出 model"
    folder.mkdir()
    shutil.copy(REAL_YUNET, folder / REAL_YUNET.name)
    pq._yunet.cache_clear()
    image = Image.new("RGB", (640, 360), (90, 90, 90))
    ImageDraw.Draw(image).ellipse((280, 100, 360, 200), fill=(220, 190, 170))
    assert isinstance(pq.detect_faces_yunet(image, folder / REAL_YUNET.name), list)
    assert pq.inspect_yunet(tmp_path)["ready"] is False


def test_generate_manifest_exposes_quality_fields(monkeypatch, tmp_path):
    class BridgeFakeEngine:
        def __init__(self, profile):
            self.pipeline, self.last_generation_metrics, self.last_reference_applied = None, {}, False

        def load(self, progress=None):
            self.pipeline = type("Pipe", (), {"tokenizer": None})()

        def generate_one(self, prompt, negative, config, seed, cancel_event, progress=None, reference_image=None):
            return _textured(config.size)

        def unload(self):
            pass

    monkeypatch.setattr(runtime, "_environment", lambda request, model_id=None: (dict(READY_ENV), tmp_path, "fake"))
    monkeypatch.setattr(runtime, "_engine_factory", lambda *args, **kwargs: BridgeFakeEngine)
    project = tmp_path / "품질 プロジェクト"
    project.mkdir()
    request = ThumbnailBridgeRequest.from_dict({
        "protocol_version": 1, "request_id": "pq", "action": "generate", "project_dir": str(project),
        "prompt": "rainy street at night", "options": {"seed": 3, "quality_profile": "fast", "identity_mode": "instantid"}})
    response = runtime.handle_request(request)
    assert response.ok, response.message
    manifest = json.loads(standard_output_paths(project)["project_manifest"].read_text(encoding="utf-8"))
    for key in ("generation_model", "quality_profile", "face_detector", "face_detail_applied", "identity_backend",
                "quality_warnings"):
        assert key in manifest
    assert manifest["quality_profile"] == "fast" and manifest["generation_model"]["profile"] == "sdxl_base"
    assert any("non-commercial" in warning for warning in response.warnings)


def test_invalid_quality_profile_is_rejected(monkeypatch, tmp_path):
    monkeypatch.setattr(runtime, "_environment", lambda request, model_id=None: (dict(READY_ENV), tmp_path, "fake"))
    request = ThumbnailBridgeRequest.from_dict({"protocol_version": 1, "action": "generate", "project_dir": str(tmp_path),
                                                "prompt": "street", "options": {"quality_profile": "ultra"}})
    response = runtime.handle_request(request)
    assert not response.ok and response.error_code == "INVALID_REQUEST"


def test_status_reports_person_quality_readiness(monkeypatch, tmp_path):
    monkeypatch.setattr(runtime, "_environment", lambda request, model_id=None: (dict(READY_ENV), tmp_path, "fake"))
    request = ThumbnailBridgeRequest.from_dict({"protocol_version": 1, "action": "status", "project_dir": str(tmp_path),
                                                "options": {"models_dir": str(tmp_path)}})
    payload = runtime.handle_request(request).outputs["status"]["person_quality"]
    assert payload["model_profiles"]["photoreal_sdxl"]["license"] == "openrail++"
    assert payload["face_detector"]["ready"] is False and payload["identity_backend"]["ready"] is False
    assert set(payload["quality_modes"]) == {"fast", "quality"}

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from threading import Event

import pytest
from PIL import Image

from covermorph.generation import (
    DEFAULT_IP_ADAPTER,
    DEFAULT_IP_ADAPTER_REVISION,
    DEFAULT_IP_ADAPTER_WEIGHT,
    IP_ADAPTER_IMAGE_ENCODER,
    GenerationCancelled,
    GenerationConfig,
    GenerationError,
    SDXLTextToImageEngine,
    detect_generation_environment,
    generate_scene_candidates,
    inspect_ip_adapter,
    retry_failed_candidates,
)
from covermorph.project import (
    ProjectAssetError,
    SceneCard,
    add_person,
    add_person_reference,
    create_project,
    load_project,
    prepare_reference_image,
    save_project_atomic,
)


class FakeEngine:
    def __init__(self, failures: set[int] | None = None, cancel_after: int | None = None) -> None:
        self.seeds: list[int] = []
        self.failures = failures or set()
        self.cancel_after = cancel_after
        self.ip_adapter_loaded = False
        self.calls: list[tuple[str, float, bool]] = []

    def generate_one(self, prompt, negative_prompt, config, seed, cancel_event, progress=None, reference_image=None):
        self.seeds.append(seed)
        self.calls.append((config.reference_mode, config.reference_strength, reference_image is not None))
        self.ip_adapter_loaded = config.reference_mode != "off" and reference_image is not None
        if seed in self.failures:
            raise GenerationError("fake failure")
        if self.cancel_after is not None and len(self.seeds) > self.cancel_after:
            raise GenerationCancelled("fake cancellation")
        return Image.new("RGB", config.size, (seed % 255, 10, 20))


class RecordingPipeline:
    def __init__(self) -> None:
        self.load_calls: list[tuple[str, dict[str, object]]] = []
        self.scales: list[float] = []
        self.unload_calls = 0
        self.call_kwargs: dict[str, object] = {}

    def load_ip_adapter(self, model_id: str, **kwargs: object) -> None:
        self.load_calls.append((model_id, kwargs))

    def set_ip_adapter_scale(self, scale: float) -> None:
        self.scales.append(scale)

    def unload_ip_adapter(self) -> None:
        self.unload_calls += 1

    def __call__(self, **kwargs: object) -> types.SimpleNamespace:
        self.call_kwargs = kwargs
        return types.SimpleNamespace(images=[Image.new("RGB", (1024, 1024), (1, 2, 3))])


def confirmed_scene() -> SceneCard:
    return SceneCard("scene-1", prompt_auto="a scene", negative_prompt_auto="text", prompt_user="a scene", negative_prompt_user="text", prompt_confirmed=True, structured_request={"reference_image_ids": ["ref-1"]})


def test_unconfirmed_scene_is_rejected(tmp_path: Path) -> None:
    project = create_project(tmp_path / "project", "unconfirmed")
    scene = SceneCard("scene-1")
    with pytest.raises(GenerationError, match="not confirmed"):
        generate_scene_candidates(project, scene, FakeEngine(), GenerationConfig(), Event())


def test_candidates_run_sequentially_with_distinct_seeds_and_round_trip(tmp_path: Path) -> None:
    project = create_project(tmp_path / "project", "generation")
    scene = confirmed_scene()
    engine = FakeEngine()
    config = GenerationConfig(candidate_count=4, seed=900, output_ratio="16:9")
    result = generate_scene_candidates(project, scene, engine, config, Event())
    save_project_atomic(project)
    assert result.completed == 4
    assert engine.seeds == [900, 901, 902, 903]
    assert len(project.candidates) == 4
    assert all(candidate.scene_id == scene.scene_id for candidate in project.candidates)
    assert all(candidate.quality_status == "unverified" and not candidate.working_source_approved for candidate in project.candidates)
    assert all(candidate.generation_metadata["references_applied"] is False for candidate in project.candidates)
    loaded = load_project(project.project_file)
    assert len(loaded.generation_runs) == 1
    assert len(loaded.candidates) == 4
    assert loaded.candidates[0].generation_metadata["seed"] == 900


def test_failed_and_cancelled_items_are_not_registered(tmp_path: Path) -> None:
    project = create_project(tmp_path / "project", "failure")
    result = generate_scene_candidates(project, confirmed_scene(), FakeEngine({101}), GenerationConfig(candidate_count=3, seed=100), Event())
    assert result.completed == 2 and result.failed == 1
    assert len(project.candidates) == 2
    project2 = create_project(tmp_path / "project2", "cancel")
    result2 = generate_scene_candidates(project2, confirmed_scene(), FakeEngine(cancel_after=1), GenerationConfig(candidate_count=3), Event())
    assert result2.cancelled is True
    assert len(project2.candidates) == 1


def test_retry_only_uses_failed_indexes(tmp_path: Path) -> None:
    project = create_project(tmp_path / "project", "retry")
    scene = confirmed_scene()
    config = GenerationConfig(candidate_count=3, seed=100)
    first = generate_scene_candidates(project, scene, FakeEngine({101}), config, Event())
    retry_engine = FakeEngine()
    retry = retry_failed_candidates(project, scene, retry_engine, config, first.failed_indices, Event())
    assert retry_engine.seeds == [101]
    assert retry.completed == 1
    assert len(project.candidates) == 3


def test_cpu_environment_never_reports_generation_ready(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    environment = detect_generation_environment(tmp_path)
    assert environment["cuda"] is False
    assert environment["status"] in {"gpu_unavailable", "package_missing"}


def test_reference_off_and_on_pass_the_expected_condition(tmp_path: Path) -> None:
    project = create_project(tmp_path / "project", "refs")
    reference_source = tmp_path / "reference.png"
    Image.new("RGBA", (32, 24), (10, 20, 30, 128)).save(reference_source)
    person = add_person(project, "A")
    reference = add_person_reference(project, person, reference_source, "person", "face")
    scene = confirmed_scene()
    off_engine = FakeEngine()
    generate_scene_candidates(project, scene, off_engine, GenerationConfig(candidate_count=1, seed=1), Event())
    assert off_engine.calls == [("off", 0.5, False)]
    on_engine = FakeEngine()
    config = GenerationConfig(candidate_count=1, seed=2, reference_mode="person", reference_image_id=reference.image_id, reference_strength=0.8)
    result = generate_scene_candidates(project, scene, on_engine, config, Event())
    assert result.completed == 1
    assert on_engine.calls == [("person", 0.8, True)]
    assert project.candidates[-1].generation_metadata["reference"]["reference_image_id"] == reference.image_id
    assert project.candidates[-1].generation_metadata["references_applied"] is True


def test_reference_modes_do_not_leak_and_crop_cache_changes(tmp_path: Path) -> None:
    project = create_project(tmp_path / "project", "cache")
    person = add_person(project, "A")
    source = tmp_path / "ref.png"
    Image.new("RGB", (40, 30), (1, 2, 3)).save(source)
    reference = add_person_reference(project, person, source, "style")
    first, first_meta = prepare_reference_image(project, reference)
    second, second_meta = prepare_reference_image(project, reference, (0, 0, 20, 20))
    assert first != second
    assert first_meta["source_sha256"] == second_meta["source_sha256"]
    style_scene = confirmed_scene()
    style_engine = FakeEngine()
    generate_scene_candidates(project, style_scene, style_engine, GenerationConfig(reference_mode="style", reference_image_id=reference.image_id, candidate_count=1), Event())
    off_engine = FakeEngine()
    generate_scene_candidates(project, style_scene, off_engine, GenerationConfig(reference_mode="off", candidate_count=1), Event())
    assert style_engine.calls[0][0:3] == ("style", 0.5, True)
    assert off_engine.calls[0][0:3] == ("off", 0.5, False)


def test_missing_reference_blocks_without_registering_output(tmp_path: Path) -> None:
    project = create_project(tmp_path / "project", "missing-ref")
    with pytest.raises(ProjectAssetError, match="missing or ambiguous"):
        generate_scene_candidates(project, confirmed_scene(), FakeEngine(), GenerationConfig(reference_mode="person", reference_image_id="gone"), Event())
    assert project.candidates == []


def test_ip_adapter_load_uses_plus_vit_h_and_explicit_image_encoder() -> None:
    pipeline = RecordingPipeline()
    engine = SDXLTextToImageEngine(local_files_only=True)
    engine.pipeline = pipeline
    config = GenerationConfig(
        reference_mode="person",
        ip_adapter_id=DEFAULT_IP_ADAPTER,
        reference_strength=0.8,
    )
    engine.load_ip_adapter(config)
    assert pipeline.load_calls == [
        (
            DEFAULT_IP_ADAPTER,
            {
                "subfolder": "sdxl_models",
                "weight_name": DEFAULT_IP_ADAPTER_WEIGHT,
                "image_encoder_folder": IP_ADAPTER_IMAGE_ENCODER,
                "local_files_only": True,
                "revision": DEFAULT_IP_ADAPTER_REVISION,
            },
        )
    ]
    assert pipeline.scales == [0.8]
    engine.load_ip_adapter(GenerationConfig(reference_mode="person", reference_strength=0.5))
    assert pipeline.scales == [0.8, 0.5]
    engine.unload_ip_adapter()
    assert pipeline.unload_calls == 1


def test_generation_call_only_passes_ip_adapter_image_when_reference_is_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    import torch

    class FakeGenerator:
        def manual_seed(self, _seed: int) -> "FakeGenerator":
            return self

    monkeypatch.setattr(torch, "Generator", lambda device: FakeGenerator())
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    off_pipeline = RecordingPipeline()
    off_engine = SDXLTextToImageEngine(local_files_only=True)
    off_engine.pipeline = off_pipeline
    off_engine.generate_one("prompt", "negative", GenerationConfig(), 1, Event())
    assert "ip_adapter_image" not in off_pipeline.call_kwargs
    on_pipeline = RecordingPipeline()
    on_engine = SDXLTextToImageEngine(local_files_only=True)
    on_engine.pipeline = on_pipeline
    reference = Image.new("RGB", (32, 32), (4, 5, 6))
    on_engine.generate_one(
        "prompt",
        "negative",
        GenerationConfig(reference_mode="style", reference_strength=0.5),
        1,
        Event(),
        reference_image=reference,
    )
    assert on_pipeline.call_kwargs["ip_adapter_image"] is reference
    assert on_engine.last_reference_applied is True


def test_ip_adapter_download_writes_verified_manifest_without_auto_download(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeSafeFile:
        def __enter__(self) -> "FakeSafeFile":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def keys(self) -> list[str]:
            return ["image_proj"]

    def fake_safe_open(*_args: object, **_kwargs: object) -> FakeSafeFile:
        return FakeSafeFile()

    def fake_snapshot_download(*, repo_id: str, revision: str, local_dir: str, **_kwargs: object) -> str:
        assert repo_id == DEFAULT_IP_ADAPTER
        assert revision == DEFAULT_IP_ADAPTER_REVISION
        destination = Path(local_dir)
        (destination / "sdxl_models").mkdir(parents=True)
        (destination / IP_ADAPTER_IMAGE_ENCODER).mkdir(parents=True)
        (destination / "sdxl_models" / DEFAULT_IP_ADAPTER_WEIGHT).write_bytes(b"valid fixture")
        (destination / IP_ADAPTER_IMAGE_ENCODER / "config.json").write_text("{}", encoding="utf-8")
        (destination / IP_ADAPTER_IMAGE_ENCODER / "model.safetensors").write_bytes(b"encoder fixture")
        return str(destination)

    monkeypatch.setitem(sys.modules, "safetensors", types.SimpleNamespace(safe_open=fake_safe_open))
    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(snapshot_download=fake_snapshot_download))
    destination = tmp_path / "ip_adapter"
    result = SDXLTextToImageEngine.download_ip_adapter(destination)
    manifest = json.loads((destination / "adapter_ready.json").read_text(encoding="utf-8"))
    assert result == destination
    assert manifest["adapter_revision"] == DEFAULT_IP_ADAPTER_REVISION
    assert manifest["adapter_weight"] == DEFAULT_IP_ADAPTER_WEIGHT
    assert inspect_ip_adapter(destination)["ready"] is True


def test_reference_metadata_records_application_separately_from_visual_quality(tmp_path: Path) -> None:
    project = create_project(tmp_path / "project", "metadata")
    person = add_person(project, "A")
    source = tmp_path / "ref.png"
    Image.new("RGB", (40, 30), (1, 2, 3)).save(source)
    reference = add_person_reference(project, person, source, "person")
    result = generate_scene_candidates(
        project,
        confirmed_scene(),
        FakeEngine(),
        GenerationConfig(reference_mode="person", reference_image_id=reference.image_id, reference_strength=0.8),
        Event(),
    )
    assert result.completed == 1
    metadata = project.candidates[-1].generation_metadata
    assert metadata["requested_reference_mode"] == "person"
    assert metadata["actual_reference_applied"] is True
    assert metadata["reference"]["source_sha256"]
    assert metadata["reference"]["processed_sha256"]
    assert metadata["reference"]["crop_box"] is None
    assert metadata["reference"]["adapter_revision"] == DEFAULT_IP_ADAPTER_REVISION
    assert metadata["generation_status"] == "succeeded"
    assert metadata["visual_quality_status"] == "unverified"


def test_damaged_processed_reference_cache_blocks_generation(tmp_path: Path) -> None:
    project = create_project(tmp_path / "project", "damaged-cache")
    person = add_person(project, "A")
    source = tmp_path / "ref.png"
    Image.new("RGB", (40, 30), (1, 2, 3)).save(source)
    reference = add_person_reference(project, person, source, "person")
    processed_path, _ = prepare_reference_image(project, reference)
    processed_path.write_bytes(b"damaged cache")
    with pytest.raises(ProjectAssetError, match="Processed reference cache is damaged"):
        generate_scene_candidates(
            project,
            confirmed_scene(),
            FakeEngine(),
            GenerationConfig(reference_mode="person", reference_image_id=reference.image_id),
            Event(),
        )
    assert project.candidates == []

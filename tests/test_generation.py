from __future__ import annotations

from pathlib import Path
from threading import Event

import pytest
from PIL import Image

from covermorph.generation import (
    GenerationCancelled,
    GenerationConfig,
    GenerationError,
    detect_generation_environment,
    generate_scene_candidates,
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


def test_cpu_environment_never_reports_generation_ready(tmp_path: Path) -> None:
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

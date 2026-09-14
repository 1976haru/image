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
    SceneCard,
    create_project,
    load_project,
    save_project_atomic,
)


class FakeEngine:
    def __init__(self, failures: set[int] | None = None, cancel_after: int | None = None) -> None:
        self.seeds: list[int] = []
        self.failures = failures or set()
        self.cancel_after = cancel_after

    def generate_one(self, prompt, negative_prompt, config, seed, cancel_event, progress=None):
        self.seeds.append(seed)
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

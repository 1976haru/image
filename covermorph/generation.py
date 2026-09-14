from __future__ import annotations

import gc
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from threading import Event
from typing import Any, Callable

from PIL import Image

from .project import CoverMorphProject, SceneCard, add_generated_candidate, utc_now

DEFAULT_SDXL_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
SDXL_LICENSE_URL = "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0"
GENERATION_SIZES = {"1:1": (1024, 1024), "16:9": (1344, 768), "9:16": (768, 1344)}


class GenerationError(RuntimeError):
    pass


class PromptTooLongError(GenerationError):
    pass


class GenerationCancelled(GenerationError):
    pass


@dataclass(slots=True)
class GenerationConfig:
    model_id: str = DEFAULT_SDXL_MODEL
    revision: str | None = None
    scheduler: str = "default"
    output_ratio: str = "1:1"
    candidate_count: int = 1
    seed: int = 0
    steps: int = 28
    guidance_scale: float = 7.0
    local_files_only: bool = True

    @property
    def size(self) -> tuple[int, int]:
        try:
            return GENERATION_SIZES[self.output_ratio]
        except KeyError as exc:
            raise GenerationError(f"Unsupported generation ratio: {self.output_ratio}") from exc


@dataclass(slots=True)
class GenerationResult:
    scene_id: str
    completed: int = 0
    failed: int = 0
    cancelled: bool = False
    candidate_ids: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    failed_indices: list[int] = field(default_factory=list)
    run_id: str = ""


def detect_generation_environment(app_root: Path, model_id: str = DEFAULT_SDXL_MODEL) -> dict[str, Any]:
    result: dict[str, Any] = {"status": "package_missing", "torch": None, "cuda": False, "gpu": None, "vram_bytes": None, "model_paths": []}
    try:
        import torch
    except ImportError:
        return result
    result["torch"] = torch.__version__
    result["cuda"] = bool(torch.cuda.is_available())
    if result["cuda"]:
        device = torch.cuda.current_device()
        result["gpu"] = torch.cuda.get_device_name(device)
        result["vram_bytes"] = int(torch.cuda.get_device_properties(device).total_memory)
    try:
        import diffusers
        result["diffusers"] = diffusers.__version__
    except ImportError:
        result["diffusers"] = None
    result["model_paths"] = [str(path) for path in (app_root / "models").glob("*")] if (app_root / "models").is_dir() else []
    model_path = Path(model_id)
    result["model_ready"] = model_path.is_dir() and (model_path / "model_index.json").is_file()
    result["status"] = "ready" if result["cuda"] and result["diffusers"] and result["model_ready"] else ("gpu_unavailable" if not result["cuda"] else ("package_missing" if not result["diffusers"] else "model_missing"))
    return result


def validate_prompt_length(prompt: str, negative_prompt: str, tokenizer: Any | None = None) -> None:
    if tokenizer is None:
        return
    for label, value in (("prompt", prompt), ("negative prompt", negative_prompt)):
        encoded = tokenizer(value, truncation=False, add_special_tokens=True)
        if len(encoded["input_ids"]) > tokenizer.model_max_length:
            raise PromptTooLongError(f"{label} exceeds the model input limit; shorten it before generating.")


class SDXLTextToImageEngine:
    """Text-to-image SDXL loader. It never reuses the inpainting pipeline."""

    def __init__(self, model_id: str = DEFAULT_SDXL_MODEL, revision: str | None = None, local_files_only: bool = False):
        self.model_id = model_id
        self.revision = revision
        self.local_files_only = local_files_only
        self.pipeline: Any | None = None
        self.loaded_revision: str | None = None

    def load(self, progress: Callable[[dict[str, Any]], None] | None = None) -> None:
        try:
            import torch
            from diffusers import StableDiffusionXLPipeline
        except ImportError as exc:
            raise GenerationError("SDXL packages are not installed in this Python environment.") from exc
        if not torch.cuda.is_available():
            raise GenerationError("CUDA GPU is unavailable. CPU fallback is disabled for SDXL generation.")
        if progress:
            progress({"phase": "model_loading", "model": self.model_id})
        kwargs: dict[str, Any] = {"torch_dtype": torch.float16, "use_safetensors": True, "local_files_only": self.local_files_only}
        if self.revision:
            kwargs["revision"] = self.revision
        try:
            self.pipeline = StableDiffusionXLPipeline.from_pretrained(self.model_id, **kwargs)
            self.pipeline.enable_attention_slicing()
            self.pipeline.to("cuda")
            self.loaded_revision = self.revision or "model-default"
        except Exception as exc:
            self.pipeline = None
            raise GenerationError(f"SDXL model load failed: {exc}") from exc

    @staticmethod
    def download(model_id: str, destination: Path, progress: Callable[[dict[str, Any]], None] | None = None) -> Path:
        try:
            from diffusers import StableDiffusionXLPipeline
        except ImportError as exc:
            raise GenerationError("Diffusers is not installed; model download is unavailable.") from exc
        if progress:
            progress({"phase": "model_download", "model": model_id})
        destination.mkdir(parents=True, exist_ok=True)
        pipeline = StableDiffusionXLPipeline.from_pretrained(model_id, use_safetensors=True, local_files_only=False)
        pipeline.save_pretrained(destination)
        del pipeline
        return destination

    def generate_one(self, prompt: str, negative_prompt: str, config: GenerationConfig, seed: int, cancel_event: Event, progress: Callable[[dict[str, Any]], None] | None = None) -> Image.Image:
        if self.pipeline is None:
            self.load(progress)
        if cancel_event.is_set():
            raise GenerationCancelled("Cancellation requested before inference.")
        import torch

        validate_prompt_length(prompt, negative_prompt, getattr(self.pipeline, "tokenizer", None))
        generator = torch.Generator(device="cuda").manual_seed(seed)
        callback = None
        if progress:
            def callback(_pipe: Any, step: int, timestep: int, callback_kwargs: dict[str, Any]) -> dict[str, Any]:
                progress({"phase": "inference", "step": step + 1, "steps": config.steps, "timestep": int(timestep)})
                if cancel_event.is_set():
                    raise GenerationCancelled("Cancellation requested during inference.")
                return callback_kwargs
        try:
            result = self.pipeline(prompt=prompt, negative_prompt=negative_prompt, width=config.size[0], height=config.size[1], num_inference_steps=config.steps, guidance_scale=config.guidance_scale, generator=generator, callback_on_step_end=callback)
        except GenerationCancelled:
            raise
        except torch.cuda.OutOfMemoryError as exc:
            raise GenerationError("CUDA out of memory; lower resolution or settings and retry.") from exc
        image = result.images[0]
        if image.size != config.size:
            raise GenerationError(f"SDXL returned {image.size}, expected {config.size}; output rejected.")
        return image.convert("RGB")

    def unload(self) -> None:
        self.pipeline = None
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


def generate_scene_candidates(project: CoverMorphProject, scene: SceneCard, engine: SDXLTextToImageEngine, config: GenerationConfig, cancel_event: Event, progress: Callable[[dict[str, Any]], None] | None = None, candidate_indices: list[int] | None = None) -> GenerationResult:
    if not scene.prompt_confirmed:
        raise GenerationError("Scene prompt is not confirmed. Review and confirm it before generating.")
    if scene.structured_request.get("reference_image_ids"):
        # The reference metadata remains in the snapshot but is intentionally not sent to SDXL in 3-A.
        references_applied = False
    else:
        references_applied = False
    run_id = f"run_{uuid.uuid4().hex}"
    indices = candidate_indices if candidate_indices is not None else list(range(max(1, config.candidate_count)))
    snapshot = {"run_id": run_id, "created_at": utc_now(), "scene_id": scene.scene_id, "prompt": scene.prompt_user or scene.prompt_auto, "negative_prompt": scene.negative_prompt_user or scene.negative_prompt_auto, "structured_request": dict(scene.structured_request), "references_applied": references_applied, "model_id": config.model_id, "revision": config.revision, "scheduler": config.scheduler, "size": list(config.size), "steps": config.steps, "guidance_scale": config.guidance_scale, "base_seed": config.seed, "candidate_indices": list(indices)}
    project.generation_runs.append(snapshot)
    result = GenerationResult(scene.scene_id, run_id=run_id)
    prompt = snapshot["prompt"]
    negative = snapshot["negative_prompt"]
    for index in indices:
        seed = config.seed + index
        if progress:
            progress({"phase": "candidate_start", "scene_id": scene.scene_id, "candidate": index + 1, "total": config.candidate_count, "seed": seed})
        try:
            image = engine.generate_one(prompt, negative, config, seed, cancel_event, progress)
            temp = project.project_dir / "assets" / ".generation_tmp" / f"{run_id}_{index}.png"
            temp.parent.mkdir(parents=True, exist_ok=True)
            image.save(temp, "PNG")
            candidate = add_generated_candidate(project, scene, temp, {**snapshot, "seed": seed, "candidate_index": index + 1, "references_applied": False})
            temp.unlink(missing_ok=True)
            result.candidate_ids.append(candidate.candidate_id)
            result.completed += 1
            if progress:
                progress({"phase": "candidate_done", "scene_id": scene.scene_id, "candidate": index + 1, "total": config.candidate_count, "candidate_id": candidate.candidate_id})
        except GenerationCancelled as exc:
            result.cancelled = True
            result.errors.append(str(exc))
            result.failed_indices.extend(indices[indices.index(index):])
            break
        except Exception as exc:
            result.failed += 1
            result.failed_indices.append(index)
            result.errors.append(f"candidate {index + 1}: {exc}")
    snapshot["loaded_revision"] = getattr(engine, "loaded_revision", None) or config.revision or "model-default"
    return result


def retry_failed_candidates(project: CoverMorphProject, scene: SceneCard, engine: SDXLTextToImageEngine, config: GenerationConfig, failed_indices: list[int], cancel_event: Event, progress: Callable[[dict[str, Any]], None] | None = None) -> GenerationResult:
    """Retry only indexes that were not registered as successful candidates."""
    return generate_scene_candidates(project, scene, engine, config, cancel_event, progress, sorted(set(failed_indices)))

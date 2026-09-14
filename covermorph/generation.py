from __future__ import annotations

import gc
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from threading import Event
from typing import Any, Callable

from PIL import Image

from .project import (
    CoverMorphProject,
    ProjectAssetError,
    SceneCard,
    add_generated_candidate,
    prepare_reference_image,
    utc_now,
)

DEFAULT_SDXL_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
DEFAULT_IP_ADAPTER = "h94/IP-Adapter"
DEFAULT_IP_ADAPTER_WEIGHT = "ip-adapter_sdxl.bin"
DEFAULT_IP_ADAPTER_SUBFOLDER = "sdxl_models"
IP_ADAPTER_ENCODER = "CLIP vision encoder bundled by the IP-Adapter repository"
IP_ADAPTER_LICENSE = "Apache-2.0"
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
    reference_mode: str = "off"
    reference_image_id: str = ""
    reference_strength: float = 0.5
    reference_crop_box: tuple[int, int, int, int] | None = None
    ip_adapter_id: str = DEFAULT_IP_ADAPTER
    ip_adapter_revision: str | None = None
    ip_adapter_weight: str = DEFAULT_IP_ADAPTER_WEIGHT

    @property
    def size(self) -> tuple[int, int]:
        try:
            return GENERATION_SIZES[self.output_ratio]
        except KeyError as exc:
            raise GenerationError(f"Unsupported generation ratio: {self.output_ratio}") from exc

    def validate(self) -> None:
        if self.reference_mode not in {"off", "person", "style"}:
            raise GenerationError("Reference mode must be off, person, or style.")
        if not 0.0 <= self.reference_strength <= 1.0:
            raise GenerationError("Reference strength must be between 0.0 and 1.0.")
        if self.candidate_count < 1 or self.steps < 1:
            raise GenerationError("Candidate count and steps must be positive.")


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
        self.ip_adapter_loaded = False
        self.ip_adapter_revision: str | None = None

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
            self.loaded_revision = getattr(self.pipeline, "_commit_hash", None) or self.revision or "model-default"
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

    @staticmethod
    def download_ip_adapter(destination: Path, model_id: str = DEFAULT_IP_ADAPTER, revision: str | None = None, progress: Callable[[dict[str, Any]], None] | None = None) -> Path:
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise GenerationError("huggingface_hub is not installed; IP-Adapter preparation is unavailable.") from exc
        if progress:
            progress({"phase": "adapter_download", "model": model_id})
        snapshot_download(repo_id=model_id, revision=revision, local_dir=str(destination), allow_patterns=["sdxl_models/*", "image_encoder/*", "*.json", "*.md", "*.safetensors", "*.bin"])
        if not (destination / DEFAULT_IP_ADAPTER_SUBFOLDER / DEFAULT_IP_ADAPTER_WEIGHT).is_file():
            raise GenerationError("IP-Adapter download completed without the required SDXL checkpoint.")
        return destination

    def load_ip_adapter(self, config: GenerationConfig, progress: Callable[[dict[str, Any]], None] | None = None) -> None:
        if self.pipeline is None:
            self.load(progress)
        try:
            kwargs: dict[str, Any] = {"subfolder": DEFAULT_IP_ADAPTER_SUBFOLDER, "weight_name": config.ip_adapter_weight, "local_files_only": config.local_files_only}
            if config.ip_adapter_revision:
                kwargs["revision"] = config.ip_adapter_revision
            self.pipeline.load_ip_adapter(config.ip_adapter_id, **kwargs)
            self.pipeline.set_ip_adapter_scale(config.reference_strength)
            self.ip_adapter_loaded = True
            self.ip_adapter_revision = config.ip_adapter_revision or "model-default"
        except Exception as exc:
            self.ip_adapter_loaded = False
            raise GenerationError(f"IP-Adapter load failed; text-only fallback is disabled: {exc}") from exc

    def generate_one(self, prompt: str, negative_prompt: str, config: GenerationConfig, seed: int, cancel_event: Event, progress: Callable[[dict[str, Any]], None] | None = None, reference_image: Image.Image | None = None) -> Image.Image:
        if self.pipeline is None:
            self.load(progress)
        if cancel_event.is_set():
            raise GenerationCancelled("Cancellation requested before inference.")
        import torch

        config.validate()
        if config.reference_mode != "off" and reference_image is None:
            raise GenerationError("A reference image is required when reference mode is enabled.")
        if config.reference_mode != "off" and not self.ip_adapter_loaded:
            self.load_ip_adapter(config, progress)

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
            kwargs: dict[str, Any] = {"prompt": prompt, "negative_prompt": negative_prompt, "width": config.size[0], "height": config.size[1], "num_inference_steps": config.steps, "guidance_scale": config.guidance_scale, "generator": generator, "callback_on_step_end": callback}
            if config.reference_mode != "off":
                kwargs["ip_adapter_image"] = reference_image
            result = self.pipeline(**kwargs)
        except GenerationCancelled:
            raise
        except torch.cuda.OutOfMemoryError as exc:
            raise GenerationError("CUDA out of memory; lower resolution or settings and retry.") from exc
        image = result.images[0]
        if image.size != config.size:
            raise GenerationError(f"SDXL returned {image.size}, expected {config.size}; output rejected.")
        return image.convert("RGB")

    def unload(self) -> None:
        if self.pipeline is not None and self.ip_adapter_loaded:
            try:
                self.pipeline.unload_ip_adapter()
            except Exception:
                pass
        self.ip_adapter_loaded = False
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
    config.validate()
    reference_image: Image.Image | None = None
    reference_meta: dict[str, Any] = {"requested_mode": config.reference_mode, "references_applied": False}
    if config.reference_mode != "off":
        if not config.reference_image_id:
            raise GenerationError("Choose exactly one reference image before generating.")
        matches = [(person.person_id, reference) for person in project.people for reference in person.reference_images if reference.image_id == config.reference_image_id]
        if len(matches) != 1:
            raise ProjectAssetError("Selected reference image is missing or ambiguous.")
        person_id, reference = matches[0]
        if reference.role != config.reference_mode:
            raise GenerationError(f"Selected reference role is {reference.role}, not {config.reference_mode}.")
        processed_path, processed_meta = prepare_reference_image(project, reference, config.reference_crop_box)
        with Image.open(processed_path) as opened:
            reference_image = opened.convert("RGB").copy()
        reference_meta = {"requested_mode": config.reference_mode, "reference_applied": False, "person_id": person_id, "reference_image_id": reference.image_id, "reference_path": str(reference.path), "reference_sha256": processed_meta["source_sha256"], "processed_sha256": processed_meta["processed_sha256"], "crop_preprocess": processed_meta, "reference_strength": config.reference_strength, "ip_adapter_id": config.ip_adapter_id, "ip_adapter_weight": config.ip_adapter_weight, "ip_adapter_revision": config.ip_adapter_revision or "model-default", "image_encoder": IP_ADAPTER_ENCODER, "license": IP_ADAPTER_LICENSE}
    run_id = f"run_{uuid.uuid4().hex}"
    indices = candidate_indices if candidate_indices is not None else list(range(max(1, config.candidate_count)))
    snapshot = {"run_id": run_id, "created_at": utc_now(), "scene_id": scene.scene_id, "prompt": scene.prompt_user or scene.prompt_auto, "negative_prompt": scene.negative_prompt_user or scene.negative_prompt_auto, "references_applied": False, "reference": reference_meta, "model_id": config.model_id, "revision": config.revision, "scheduler": config.scheduler, "size": list(config.size), "steps": config.steps, "guidance_scale": config.guidance_scale, "base_seed": config.seed, "candidate_indices": list(indices), "failed_indices": [], "errors": [], "cancelled": False}
    project.generation_runs.append(snapshot)
    result = GenerationResult(scene.scene_id, run_id=run_id)
    prompt = snapshot["prompt"]
    negative = snapshot["negative_prompt"]
    for index in indices:
        seed = config.seed + index
        if progress:
            progress({"phase": "candidate_start", "scene_id": scene.scene_id, "candidate": index + 1, "total": config.candidate_count, "seed": seed})
        try:
            image = engine.generate_one(prompt, negative, config, seed, cancel_event, progress, reference_image)
            if config.reference_mode != "off":
                snapshot["reference"]["reference_applied"] = engine.ip_adapter_loaded
                snapshot["references_applied"] = engine.ip_adapter_loaded
            temp = project.project_dir / "assets" / ".generation_tmp" / f"{run_id}_{index}.png"
            temp.parent.mkdir(parents=True, exist_ok=True)
            image.save(temp, "PNG")
            candidate = add_generated_candidate(project, scene, temp, {**snapshot, "seed": seed, "candidate_index": index + 1})
            temp.unlink(missing_ok=True)
            result.candidate_ids.append(candidate.candidate_id)
            result.completed += 1
            if progress:
                progress({"phase": "candidate_done", "scene_id": scene.scene_id, "candidate": index + 1, "total": config.candidate_count, "candidate_id": candidate.candidate_id})
        except GenerationCancelled as exc:
            result.cancelled = True
            result.errors.append(str(exc))
            result.failed_indices.extend(indices[indices.index(index):])
            snapshot["cancelled"] = True
            break
        except Exception as exc:
            result.failed += 1
            result.failed_indices.append(index)
            result.errors.append(f"candidate {index + 1}: {exc}")
    snapshot["failed_indices"] = list(result.failed_indices)
    snapshot["errors"] = list(result.errors)
    snapshot["loaded_revision"] = getattr(engine, "loaded_revision", None) or config.revision or "model-default"
    return result


def retry_failed_candidates(project: CoverMorphProject, scene: SceneCard, engine: SDXLTextToImageEngine, config: GenerationConfig, failed_indices: list[int], cancel_event: Event, progress: Callable[[dict[str, Any]], None] | None = None) -> GenerationResult:
    """Retry only indexes that were not registered as successful candidates."""
    return generate_scene_candidates(project, scene, engine, config, cancel_event, progress, sorted(set(failed_indices)))

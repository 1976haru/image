from __future__ import annotations

import copy
import gc
import hashlib
import json
import uuid
from dataclasses import asdict, dataclass, field
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
    resolve_project_path,
    save_project_atomic,
    utc_now,
)

DEFAULT_SDXL_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
DEFAULT_IP_ADAPTER = "h94/IP-Adapter"
DEFAULT_IP_ADAPTER_WEIGHT = "ip-adapter-plus_sdxl_vit-h.safetensors"
DEFAULT_IP_ADAPTER_REVISION = "9bf28b38530e55ffa91c6d82e5161a982c22f284"
IP_ADAPTER_ENCODER_FOLDER = "models/image_encoder"
IP_ADAPTER_FILES = ("sdxl_models/" + DEFAULT_IP_ADAPTER_WEIGHT, "models/image_encoder/config.json", "models/image_encoder/model.safetensors")
DEFAULT_IP_ADAPTER_SUBFOLDER = "sdxl_models"
IP_ADAPTER_ENCODER = "h94/IP-Adapter/models/image_encoder (OpenCLIP ViT-H-14)"
IP_ADAPTER_LICENSE = "Apache-2.0"
IP_ADAPTER_HASHES = {
    "sdxl_models/" + DEFAULT_IP_ADAPTER_WEIGHT: "3f5062b8400c94b7159665b21ba5c62acdcd7682262743d7f2aefedef00e6581",
    "models/image_encoder/model.safetensors": "6ca9667da1ca9e0b0f75e46bb030f7e011f44f86cbfb8d5a36590fcd7507b030",
}
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
    ip_adapter_revision: str | None = DEFAULT_IP_ADAPTER_REVISION
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
    if not model_path.is_absolute():
        candidates = [model_path, app_root / model_path]
        model_path = next((candidate for candidate in candidates if candidate.is_dir()), candidates[-1])
    result["resolved_model_path"] = str(model_path) if model_path.is_dir() else None
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


def file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def validate_adapter_directory(directory: Path) -> dict[str, Any]:
    try:
        manifest = json.loads((directory / "adapter_ready.json").read_text(encoding="utf-8"))
        if not manifest.get("revision"):
            raise ValueError("Missing revision")
        for name in IP_ADAPTER_FILES:
            if name in IP_ADAPTER_HASHES and manifest["files"][name] != IP_ADAPTER_HASHES[name]:
                raise ValueError(f"Unsupported checkpoint: {name}")
            if file_sha256(directory / name) != manifest["files"][name]:
                raise ValueError(f"Changed model file: {name}")
        return manifest
    except (OSError, ValueError, KeyError) as exc:
        raise GenerationError(f"Reference model is incomplete; use the prepare button: {exc}") from exc


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
        self.adapter_signature = None
        self.memory_ready = False
        self.last_reference_applied = False

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
            self.pipeline.enable_vae_slicing()
            self.pipeline.enable_vae_tiling()
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
        from huggingface_hub import HfApi
        resolved = HfApi().model_info(model_id, revision=revision or DEFAULT_IP_ADAPTER_REVISION).sha
        destination.mkdir(parents=True, exist_ok=True)
        marker = destination / "adapter_ready.json"
        marker.unlink(missing_ok=True)
        snapshot_download(repo_id=model_id, revision=resolved, local_dir=str(destination), allow_patterns=list(IP_ADAPTER_FILES))
        hashes = {}
        for name in IP_ADAPTER_FILES:
            path = destination / name
            if not path.is_file() or path.stat().st_size == 0:
                raise GenerationError(f"Incomplete adapter download: {name}")
            if name.endswith(".json"):
                json.loads(path.read_text(encoding="utf-8"))
            else:
                from safetensors import safe_open
                with safe_open(str(path), framework="pt", device="cpu") as handle:
                    if not list(handle.keys()):
                        raise GenerationError(f"Empty checkpoint: {name}")
            hashes[name] = file_sha256(path)
            if name in IP_ADAPTER_HASHES and hashes[name] != IP_ADAPTER_HASHES[name]:
                raise GenerationError(f"Checkpoint checksum mismatch: {name}")
        temporary = destination / "adapter_ready.tmp"
        temporary.write_text(json.dumps({"repo_id": model_id, "revision": resolved, "files": hashes}, indent=2), encoding="utf-8")
        temporary.replace(marker)
        return destination

    def load_ip_adapter(self, config: GenerationConfig, progress: Callable[[dict[str, Any]], None] | None = None) -> None:
        if self.pipeline is None:
            self.load(progress)
        try:
            import torch
            from transformers import CLIPVisionModelWithProjection
            revision = config.ip_adapter_revision or DEFAULT_IP_ADAPTER_REVISION
            if Path(config.ip_adapter_id).is_dir():
                revision = validate_adapter_directory(Path(config.ip_adapter_id))["revision"]
            encoder = CLIPVisionModelWithProjection.from_pretrained(
                config.ip_adapter_id, subfolder=IP_ADAPTER_ENCODER_FOLDER,
                revision=revision, torch_dtype=torch.float16,
                use_safetensors=True, local_files_only=config.local_files_only)
            self.pipeline.register_modules(image_encoder=encoder)
            self.pipeline.load_ip_adapter(config.ip_adapter_id,
                subfolder=DEFAULT_IP_ADAPTER_SUBFOLDER, weight_name=config.ip_adapter_weight,
                image_encoder_folder=None, revision=revision, local_files_only=config.local_files_only)
            self.ip_adapter_loaded = True
            self.ip_adapter_revision = revision
        except Exception as exc:
            self.unload()
            raise GenerationError(f"IP-Adapter load failed; text-only fallback is disabled: {exc}") from exc

    def generate_one(self, prompt: str, negative_prompt: str, config: GenerationConfig, seed: int, cancel_event: Event, progress: Callable[[dict[str, Any]], None] | None = None, reference_image: Image.Image | None = None) -> Image.Image:
        self.last_reference_applied = False
        config.validate()
        if config.reference_mode != "off" and reference_image is None:
            raise GenerationError("A reference image is required when reference mode is enabled.")
        if cancel_event.is_set():
            raise GenerationCancelled("Cancellation requested before inference.")
        import torch

        signature = None if config.reference_mode == "off" else (config.reference_mode, config.ip_adapter_id, config.ip_adapter_revision, config.ip_adapter_weight)
        if self.pipeline is not None and self.adapter_signature != signature:
            self.unload()
        if self.pipeline is None:
            self.load(progress)
        if signature is not None:
            if not self.ip_adapter_loaded:
                self.load_ip_adapter(config, progress)
            self.pipeline.set_ip_adapter_scale(config.reference_strength)
        self.adapter_signature = signature
        if not self.memory_ready:
            self.pipeline.enable_model_cpu_offload()
            self.memory_ready = True

        validate_prompt_length(prompt, negative_prompt, getattr(self.pipeline, "tokenizer", None))
        generator = torch.Generator(device="cuda").manual_seed(seed)
        def callback(_pipe: Any, step: int, timestep: int, callback_kwargs: dict[str, Any]) -> dict[str, Any]:
            if progress:
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
        if cancel_event.is_set():
            raise GenerationCancelled("Cancellation requested after inference.")
        self.last_reference_applied = config.reference_mode != "off" and "ip_adapter_image" in kwargs
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
        self.adapter_signature = None
        self.memory_ready = False
        self.pipeline = None
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


def generate_scene_candidates(project: CoverMorphProject, scene: SceneCard, engine: SDXLTextToImageEngine, config: GenerationConfig, cancel_event: Event, progress: Callable[[dict[str, Any]], None] | None = None, candidate_indices: list[int] | None = None, reference_snapshot: dict[str, Any] | None = None) -> GenerationResult:
    config = copy.deepcopy(config)
    scene = copy.deepcopy(scene)
    if not scene.prompt_confirmed:
        raise GenerationError("Scene prompt is not confirmed. Review and confirm it before generating.")
    config.validate()
    reference_image: Image.Image | None = None
    reference_meta: dict[str, Any] = {"requested_mode": config.reference_mode, "reference_applied": False}
    try:
        if reference_snapshot is not None and config.reference_mode != "off":
            reference_meta = copy.deepcopy(reference_snapshot)
            reference_meta["reference_applied"] = False
            path = resolve_project_path(project, reference_meta["processed_path"])
            if file_sha256(path) != reference_meta["processed_sha256"]:
                raise ProjectAssetError("Saved reference snapshot changed; retry blocked.")
            with Image.open(path) as opened:
                reference_image = opened.convert("RGB").copy()
        elif config.reference_mode != "off":
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
            reference_meta = {"requested_mode": config.reference_mode, "reference_applied": False, "person_id": person_id, "reference_image_id": reference.image_id, "reference_path": str(reference.path), "reference_sha256": processed_meta["source_sha256"], "processed_sha256": processed_meta["processed_sha256"], "crop_preprocess": processed_meta, "reference_strength": config.reference_strength, "ip_adapter_id": config.ip_adapter_id, "ip_adapter_weight": config.ip_adapter_weight, "ip_adapter_revision": config.ip_adapter_revision or "model-default", "image_encoder": IP_ADAPTER_ENCODER, "image_encoder_revision": config.ip_adapter_revision, "processed_path": processed_path.relative_to(project.project_dir).as_posix(), "license": IP_ADAPTER_LICENSE}
    except (OSError, ValueError, ProjectAssetError, GenerationError) as exc:
        project.generation_runs.append({"run_id": f"run_{uuid.uuid4().hex}", "scene_id": scene.scene_id,
            "config": asdict(config), "reference": reference_meta, "references_applied": False,
            "generation_status": "blocked", "visual_quality_status": "unverified", "errors": [str(exc)]})
        save_project_atomic(project)
        raise
    run_id = f"run_{uuid.uuid4().hex}"
    indices = candidate_indices if candidate_indices is not None else list(range(max(1, config.candidate_count)))
    snapshot = {"run_id": run_id, "created_at": utc_now(), "scene_id": scene.scene_id, "prompt": scene.prompt_user or scene.prompt_auto, "negative_prompt": scene.negative_prompt_user or scene.negative_prompt_auto, "references_applied": False, "reference": reference_meta, "model_id": config.model_id, "revision": config.revision, "scheduler": config.scheduler, "size": list(config.size), "steps": config.steps, "guidance_scale": config.guidance_scale, "base_seed": config.seed, "candidate_indices": list(indices), "failed_indices": [], "errors": [], "cancelled": False}
    snapshot["config"] = asdict(config)
    snapshot["visual_quality_status"] = "unverified"
    snapshot["generation_status"] = "running"
    snapshot["outcomes"] = []
    project.generation_runs.append(snapshot)
    result = GenerationResult(scene.scene_id, run_id=run_id)
    prompt = snapshot["prompt"]
    negative = snapshot["negative_prompt"]
    for index in indices:
        seed = config.seed + index
        if progress:
            progress({"phase": "candidate_start", "scene_id": scene.scene_id, "candidate": index + 1, "total": config.candidate_count, "seed": seed})
        try:
            image = engine.generate_one(prompt, negative, config, seed, cancel_event, progress, reference_image.copy() if reference_image is not None else None)
            if config.reference_mode != "off":
                snapshot["reference"]["reference_applied"] = getattr(engine, "last_reference_applied", False)
                snapshot["references_applied"] = getattr(engine, "last_reference_applied", False)
                snapshot["reference"]["ip_adapter_revision"] = getattr(engine, "ip_adapter_revision", config.ip_adapter_revision)
                snapshot["reference"]["image_encoder_revision"] = snapshot["reference"]["ip_adapter_revision"]
                if not snapshot["references_applied"]:
                    raise GenerationError("Requested reference was not applied; output rejected.")
            if cancel_event.is_set():
                raise GenerationCancelled("Cancelled before output registration.")
            temp = project.project_dir / "assets" / ".generation_tmp" / f"{run_id}_{index}.png"
            temp.parent.mkdir(parents=True, exist_ok=True)
            image.save(temp, "PNG")
            candidate = add_generated_candidate(project, scene, temp, {**copy.deepcopy(snapshot), "seed": seed, "candidate_index": index + 1, "generation_status": "generated"})
            temp.unlink(missing_ok=True)
            result.candidate_ids.append(candidate.candidate_id)
            snapshot["outcomes"].append({"index": index, "status": "generated", "reference_applied": snapshot["references_applied"]})
            result.completed += 1
            save_project_atomic(project)
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
            snapshot["outcomes"].append({"index": index, "status": "failed", "error": str(exc), "reference_applied": False})
    snapshot["generation_status"] = "cancelled" if result.cancelled else ("partial" if result.failed and result.completed else "failed" if result.failed else "generated")
    snapshot["failed_indices"] = list(result.failed_indices)
    snapshot["errors"] = list(result.errors)
    snapshot["loaded_revision"] = getattr(engine, "loaded_revision", None) or config.revision or "model-default"
    save_project_atomic(project)
    return result


def retry_failed_candidates(project: CoverMorphProject, scene: SceneCard, engine: SDXLTextToImageEngine, config: GenerationConfig, failed_indices: list[int], cancel_event: Event, progress: Callable[[dict[str, Any]], None] | None = None) -> GenerationResult:
    """Retry only indexes that were not registered as successful candidates."""
    previous = next((run for run in reversed(project.generation_runs) if run["scene_id"] == scene.scene_id and run.get("failed_indices") == failed_indices and "config" in run), None)
    reference_snapshot = None
    if previous:
        config = GenerationConfig(**copy.deepcopy(previous["config"]))
        scene = copy.deepcopy(scene)
        scene.prompt_user = previous["prompt"]
        scene.negative_prompt_user = previous["negative_prompt"]
        scene.prompt_confirmed = True  # Retry the original confirmed request, not current edits.
        reference_snapshot = previous.get("reference")
        engine.model_id = config.model_id
        engine.revision = config.revision
    return generate_scene_candidates(project, scene, engine, config, cancel_event, progress, sorted(set(failed_indices)), reference_snapshot)

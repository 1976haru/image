from __future__ import annotations

import copy
import gc
import hashlib
import json
import time
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
DEFAULT_IP_ADAPTER_REVISION = "9bf28b38530e55ffa91c6d82e5161a982c22f284"
DEFAULT_IP_ADAPTER_WEIGHT = "ip-adapter-plus_sdxl_vit-h.safetensors"
DEFAULT_IP_ADAPTER_SUBFOLDER = "sdxl_models"
IP_ADAPTER_IMAGE_ENCODER = "models/image_encoder"
IP_ADAPTER_ENCODER = IP_ADAPTER_IMAGE_ENCODER
IP_ADAPTER_ENCODER_FOLDER = IP_ADAPTER_IMAGE_ENCODER
IP_ADAPTER_FILES = (
    f"{DEFAULT_IP_ADAPTER_SUBFOLDER}/{DEFAULT_IP_ADAPTER_WEIGHT}",
    f"{IP_ADAPTER_IMAGE_ENCODER}/config.json",
    f"{IP_ADAPTER_IMAGE_ENCODER}/model.safetensors",
)
IP_ADAPTER_LICENSE = "Apache-2.0"
ADAPTER_READY_FILENAME = "adapter_ready.json"
SDXL_LICENSE_URL = "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0"
GENERATION_SIZES = {"1:1": (1024, 1024), "16:9": (1344, 768), "9:16": (768, 1344)}


class GenerationError(RuntimeError):
    pass


class PromptTooLongError(GenerationError):
    pass


class GenerationCancelled(GenerationError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_sha256(path: Path) -> str:
    """Return a file digest for callers that use the historical public helper."""
    return _sha256(path)


def _has_model_weight(directory: Path) -> bool:
    return any(
        path.is_file()
        for pattern in ("*.safetensors", "*.bin", "*.safetensors.index.json", "*.bin.index.json")
        for path in directory.glob(pattern)
    )


def inspect_sdxl_model(model_path: Path) -> dict[str, Any]:
    """Inspect a local SDXL snapshot without downloading or mutating it."""
    required = ("unet", "vae", "text_encoder", "text_encoder_2", "tokenizer", "tokenizer_2", "scheduler")
    result: dict[str, Any] = {
        "path": str(model_path),
        "exists": model_path.is_dir(),
        "model_index": (model_path / "model_index.json").is_file(),
        "ready": False,
        "failure_reason": None,
    }
    if not result["exists"]:
        result["failure_reason"] = "model directory is missing"
        return result
    if not result["model_index"]:
        result["failure_reason"] = "model_index.json is missing"
        return result
    try:
        json.loads((model_path / "model_index.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        result["failure_reason"] = f"model_index.json is invalid: {exc}"
        return result
    missing = [name for name in required if not (model_path / name).is_dir()]
    unweighted = [name for name in required if (model_path / name).is_dir() and name not in {"tokenizer", "tokenizer_2", "scheduler"} and not _has_model_weight(model_path / name)]
    if missing or unweighted:
        result["failure_reason"] = f"incomplete SDXL snapshot; missing={missing}, unweighted={unweighted}"
        return result
    result["ready"] = True
    return result


def inspect_ip_adapter(
    destination: Path,
    model_id: str = DEFAULT_IP_ADAPTER,
    revision: str = DEFAULT_IP_ADAPTER_REVISION,
    weight_name: str = DEFAULT_IP_ADAPTER_WEIGHT,
) -> dict[str, Any]:
    """Validate the exact local IP-Adapter Plus SDXL ViT-H preparation."""
    weight_path = destination / DEFAULT_IP_ADAPTER_SUBFOLDER / weight_name
    encoder_path = destination / IP_ADAPTER_IMAGE_ENCODER
    manifest_path = destination / ADAPTER_READY_FILENAME
    result: dict[str, Any] = {
        "path": str(destination),
        "adapter_id": model_id,
        "adapter_revision": revision,
        "adapter_weight": weight_name,
        "weight_path": str(weight_path),
        "weight_exists": weight_path.is_file(),
        "weight_sha256": None,
        "weight_readable": False,
        "hash_verified": False,
        "image_encoder": IP_ADAPTER_IMAGE_ENCODER,
        "image_encoder_revision": revision,
        "image_encoder_path": str(encoder_path),
        "image_encoder_ready": False,
        "manifest_path": str(manifest_path),
        "manifest_valid": False,
        "ready": False,
        "failure_reason": None,
    }
    if not weight_path.is_file():
        result["failure_reason"] = "required IP-Adapter safetensors file is missing"
        return result
    try:
        from safetensors import safe_open

        with safe_open(str(weight_path), framework="pt", device="cpu") as opened:
            result["weight_tensor_count"] = len(opened.keys())
        result["weight_sha256"] = _sha256(weight_path)
        result["weight_readable"] = True
    except Exception as exc:
        result["failure_reason"] = f"IP-Adapter safetensors is unreadable: {exc}"
        return result
    encoder_weights = _has_model_weight(encoder_path)
    result["image_encoder_ready"] = (encoder_path / "config.json").is_file() and encoder_weights
    if not result["image_encoder_ready"]:
        result["failure_reason"] = "models/image_encoder is incomplete"
        return result
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        manifest = None
    expected_manifest = {
        "adapter_id": model_id,
        "adapter_revision": revision,
        "adapter_weight": weight_name,
        "weight_sha256": result["weight_sha256"],
        "image_encoder": IP_ADAPTER_IMAGE_ENCODER,
        "image_encoder_revision": revision,
    }
    result["manifest_valid"] = isinstance(manifest, dict) and all(manifest.get(key) == value for key, value in expected_manifest.items())
    result["hash_verified"] = bool(result["manifest_valid"])
    if not result["manifest_valid"]:
        result["failure_reason"] = "adapter_ready.json is missing or does not match the prepared files"
        return result
    result["ready"] = True
    return result


def validate_adapter_directory(directory: Path) -> dict[str, Any]:
    """Validate current adapter manifests and older hash manifests."""
    try:
        manifest = json.loads((directory / ADAPTER_READY_FILENAME).read_text(encoding="utf-8"))
        if "files" in manifest:
            if not manifest.get("revision"):
                raise ValueError("Missing revision")
            files = manifest["files"]
            for name in IP_ADAPTER_FILES:
                expected = files[name]
                if file_sha256(directory / name) != expected:
                    raise ValueError(f"Changed model file: {name}")
            return manifest
        inspection = inspect_ip_adapter(
            directory,
            manifest.get("adapter_id", DEFAULT_IP_ADAPTER),
            manifest.get("adapter_revision", DEFAULT_IP_ADAPTER_REVISION),
            manifest.get("adapter_weight", DEFAULT_IP_ADAPTER_WEIGHT),
        )
        if not inspection["ready"]:
            raise ValueError(inspection["failure_reason"] or "incomplete adapter")
        return manifest
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise GenerationError(f"Reference model is incomplete; use the prepare button: {exc}") from exc


def _write_adapter_manifest(destination: Path, inspection: dict[str, Any]) -> Path:
    manifest_path = destination / ADAPTER_READY_FILENAME
    payload = {
        "adapter_id": inspection["adapter_id"],
        "adapter_revision": inspection["adapter_revision"],
        "adapter_weight": inspection["adapter_weight"],
        "weight_sha256": inspection["weight_sha256"],
        "image_encoder": inspection["image_encoder"],
        "image_encoder_revision": inspection["image_encoder_revision"],
        "prepared_at": utc_now(),
    }
    manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest_path


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
    result: dict[str, Any] = {
        "status": "package_missing",
        "torch": None,
        "cuda": False,
        "gpu": None,
        "vram_bytes": None,
        "model_paths": [],
        "diffusers": None,
        "transformers": None,
        "accelerate": None,
        "safetensors": None,
    }
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
        import accelerate
        result["accelerate"] = accelerate.__version__
    except (ImportError, AttributeError):
        pass
    try:
        import diffusers
        result["diffusers"] = diffusers.__version__
    except (ImportError, AttributeError):
        pass
    try:
        import safetensors
        result["safetensors"] = safetensors.__version__
    except (ImportError, AttributeError):
        pass
    try:
        import transformers
        result["transformers"] = transformers.__version__
    except (ImportError, AttributeError):
        pass
    result["model_paths"] = [str(path) for path in (app_root / "models").glob("*")] if (app_root / "models").is_dir() else []
    model_path = Path(model_id)
    if not model_path.is_absolute():
        candidates = [model_path, app_root / model_path]
        model_path = next((candidate for candidate in candidates if candidate.is_dir()), candidates[-1])
    result["resolved_model_path"] = str(model_path) if model_path.is_dir() else None
    result["model"] = inspect_sdxl_model(model_path)
    result["model_prepared"] = result["model"]["ready"]
    # Keep the original marker-level field for clients that used it before strict inspection.
    result["model_ready"] = model_path.is_dir() and (model_path / "model_index.json").is_file()
    result["ip_adapter"] = inspect_ip_adapter(app_root / "models" / "ip_adapter")
    result["ip_adapter_ready"] = result["ip_adapter"]["ready"]
    packages_ready = all(result[name] for name in ("diffusers", "transformers", "accelerate", "safetensors"))
    result["status"] = (
        "gpu_unavailable"
        if not result["cuda"]
        else ("package_missing" if not packages_ready else ("model_missing" if not result["model_prepared"] else "ready"))
    )
    result["reference_status"] = "ready" if result["status"] == "ready" and result["ip_adapter_ready"] else (
        "adapter_missing" if result["status"] == "ready" else result["status"]
    )
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
        self.ip_adapter_config: tuple[str, str, str, str] | None = None
        self.adapter_signature: tuple[str, str, str, str] | None = None
        self.last_reference_applied = False
        self.last_generation_metrics: dict[str, Any] = {}

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
        inspection = inspect_sdxl_model(destination)
        if not inspection["ready"]:
            raise GenerationError(f"SDXL model preparation verification failed: {inspection['failure_reason']}")
        return destination

    @staticmethod
    def download_ip_adapter(destination: Path, model_id: str = DEFAULT_IP_ADAPTER, revision: str | None = DEFAULT_IP_ADAPTER_REVISION, progress: Callable[[dict[str, Any]], None] | None = None) -> Path:
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise GenerationError("huggingface_hub is not installed; IP-Adapter preparation is unavailable.") from exc
        if progress:
            progress({"phase": "adapter_download", "model": model_id})
        adapter_revision = revision or DEFAULT_IP_ADAPTER_REVISION
        snapshot_download(repo_id=model_id, revision=adapter_revision, local_dir=str(destination), allow_patterns=["sdxl_models/*", "models/image_encoder/*", "*.json", "*.md", "*.safetensors", "*.bin"])
        inspection = inspect_ip_adapter(destination, model_id, adapter_revision)
        if not inspection["weight_readable"] or not inspection["image_encoder_ready"]:
            raise GenerationError(f"IP-Adapter download did not produce a complete preparation: {inspection['failure_reason']}")
        _write_adapter_manifest(destination, inspection)
        verified = inspect_ip_adapter(destination, model_id, adapter_revision)
        if not verified["ready"]:
            raise GenerationError(f"IP-Adapter preparation verification failed: {verified['failure_reason']}")
        return destination

    def load_ip_adapter(self, config: GenerationConfig, progress: Callable[[dict[str, Any]], None] | None = None) -> None:
        if self.pipeline is None:
            self.load(progress)
        adapter_revision = config.ip_adapter_revision or DEFAULT_IP_ADAPTER_REVISION
        adapter_key = (config.reference_mode, config.ip_adapter_id, adapter_revision, config.ip_adapter_weight)
        if self.ip_adapter_loaded and self.ip_adapter_config != adapter_key:
            self.unload()
            self.load(progress)
        if Path(config.ip_adapter_id).is_dir():
            inspection = inspect_ip_adapter(Path(config.ip_adapter_id), DEFAULT_IP_ADAPTER, adapter_revision, config.ip_adapter_weight)
            if not inspection["ready"]:
                raise GenerationError(f"IP-Adapter preparation is incomplete: {inspection['failure_reason']}")
        if self.ip_adapter_loaded:
            try:
                if hasattr(self.pipeline, "set_ip_adapter_scale"):
                    self.pipeline.set_ip_adapter_scale(config.reference_strength)
                    return
            except Exception as exc:
                raise GenerationError(f"IP-Adapter scale update failed: {exc}") from exc
        try:
            kwargs: dict[str, Any] = {
                "subfolder": DEFAULT_IP_ADAPTER_SUBFOLDER,
                "weight_name": config.ip_adapter_weight,
                "local_files_only": config.local_files_only,
                "revision": adapter_revision,
            }
            if hasattr(self.pipeline, "register_modules"):
                import torch
                from transformers import CLIPVisionModelWithProjection

                encoder = CLIPVisionModelWithProjection.from_pretrained(
                    config.ip_adapter_id,
                    subfolder=IP_ADAPTER_ENCODER_FOLDER,
                    revision=adapter_revision,
                    torch_dtype=torch.float16,
                    use_safetensors=True,
                    local_files_only=config.local_files_only,
                )
                self.pipeline.register_modules(image_encoder=encoder)
                # The pipeline was moved to CUDA before this module existed, so
                # registering it does not migrate the encoder automatically.
                # Move the complete pipeline again to keep image embeddings and
                # IP-Adapter weights on the same device during inference.
                if hasattr(self.pipeline, "to"):
                    self.pipeline.to("cuda")
                kwargs["image_encoder_folder"] = None
            else:
                kwargs["image_encoder_folder"] = IP_ADAPTER_IMAGE_ENCODER
            # Diffusers cannot replace already-sliced attention processors with
            # IP-Adapter processors because SlicedAttnProcessor requires a
            # slice_size constructor argument. Restore the default processors
            # for the adapter load. Keep slicing disabled afterward because
            # enabling it would replace IPAdapterAttnProcessor with a generic
            # sliced processor that cannot consume image embeddings.
            attention_slicing = hasattr(self.pipeline, "disable_attention_slicing")
            if attention_slicing:
                self.pipeline.disable_attention_slicing()
            try:
                self.pipeline.load_ip_adapter(config.ip_adapter_id, **kwargs)
            except Exception:
                if attention_slicing:
                    self.pipeline.enable_attention_slicing()
                raise
            if hasattr(self.pipeline, "set_ip_adapter_scale"):
                self.pipeline.set_ip_adapter_scale(config.reference_strength)
            self.ip_adapter_loaded = True
            self.ip_adapter_revision = adapter_revision
            self.ip_adapter_config = adapter_key
            self.adapter_signature = adapter_key
        except Exception as exc:
            self.unload()
            raise GenerationError(f"IP-Adapter load failed; text-only fallback is disabled: {exc}") from exc

    def generate_one(self, prompt: str, negative_prompt: str, config: GenerationConfig, seed: int, cancel_event: Event, progress: Callable[[dict[str, Any]], None] | None = None, reference_image: Image.Image | None = None) -> Image.Image:
        config.validate()
        if self.pipeline is None:
            self.load(progress)
        if cancel_event.is_set():
            raise GenerationCancelled("Cancellation requested before inference.")
        import torch

        if config.reference_mode != "off" and reference_image is None:
            raise GenerationError("A reference image is required when reference mode is enabled.")
        if config.reference_mode == "off":
            if self.ip_adapter_loaded:
                self.unload()
                self.load(progress)
            else:
                self.unload_ip_adapter()
        else:
            adapter_key = (config.reference_mode, config.ip_adapter_id, config.ip_adapter_revision or DEFAULT_IP_ADAPTER_REVISION, config.ip_adapter_weight)
            if self.ip_adapter_loaded and self.ip_adapter_config is not None and self.ip_adapter_config != adapter_key:
                self.unload()
                self.load(progress)
            if not self.ip_adapter_loaded:
                self.load_ip_adapter(config, progress)
                self.ip_adapter_config = adapter_key
                self.adapter_signature = adapter_key
            else:
                try:
                    if hasattr(self.pipeline, "set_ip_adapter_scale"):
                        self.pipeline.set_ip_adapter_scale(config.reference_strength)
                except Exception as exc:
                    raise GenerationError(f"IP-Adapter scale update failed: {exc}") from exc

        validate_prompt_length(prompt, negative_prompt, getattr(self.pipeline, "tokenizer", None))
        generator = torch.Generator(device="cuda").manual_seed(seed)
        def callback(_pipe: Any, step: int, timestep: int, callback_kwargs: dict[str, Any]) -> dict[str, Any]:
            if progress:
                progress({"phase": "inference", "step": step + 1, "steps": config.steps, "timestep": int(timestep)})
            if cancel_event.is_set():
                raise GenerationCancelled("Cancellation requested during inference.")
            return callback_kwargs
        started = time.perf_counter()
        cuda_oom = False
        self.last_reference_applied = False
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        try:
            kwargs: dict[str, Any] = {"prompt": prompt, "negative_prompt": negative_prompt, "width": config.size[0], "height": config.size[1], "num_inference_steps": config.steps, "guidance_scale": config.guidance_scale, "generator": generator, "callback_on_step_end": callback}
            if config.reference_mode != "off":
                kwargs["ip_adapter_image"] = reference_image
            result = self.pipeline(**kwargs)
        except GenerationCancelled:
            raise
        except torch.cuda.OutOfMemoryError as exc:
            cuda_oom = True
            raise GenerationError("CUDA out of memory; lower resolution or settings and retry.") from exc
        finally:
            peak_allocated = None
            peak_reserved = None
            if torch.cuda.is_available():
                peak_allocated = int(torch.cuda.max_memory_allocated())
                peak_reserved = int(torch.cuda.max_memory_reserved())
            self.last_generation_metrics = {
                "generation_time_seconds": round(time.perf_counter() - started, 3),
                "peak_memory_allocated": peak_allocated,
                "peak_memory_reserved": peak_reserved,
                "cuda_out_of_memory": cuda_oom,
            }
        image = result.images[0]
        if image.size != config.size:
            raise GenerationError(f"SDXL returned {image.size}, expected {config.size}; output rejected.")
        self.last_reference_applied = config.reference_mode != "off" and self.ip_adapter_loaded and reference_image is not None
        return image.convert("RGB")

    def unload(self) -> None:
        self.unload_ip_adapter()
        self.pipeline = None
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def unload_ip_adapter(self) -> None:
        if self.pipeline is not None and self.ip_adapter_loaded:
            try:
                self.pipeline.unload_ip_adapter()
            except Exception:
                pass
        self.ip_adapter_loaded = False
        self.ip_adapter_revision = None
        self.ip_adapter_config = None
        self.adapter_signature = None
        self.last_reference_applied = False


def generate_scene_candidates(
    project: CoverMorphProject,
    scene: SceneCard,
    engine: SDXLTextToImageEngine,
    config: GenerationConfig,
    cancel_event: Event,
    progress: Callable[[dict[str, Any]], None] | None = None,
    candidate_indices: list[int] | None = None,
    reference_snapshot: dict[str, Any] | None = None,
) -> GenerationResult:
    config = copy.deepcopy(config)
    scene = copy.deepcopy(scene)
    if not scene.prompt_confirmed:
        raise GenerationError("Scene prompt is not confirmed. Review and confirm it before generating.")
    config.validate()
    reference_image: Image.Image | None = None
    adapter_revision = config.ip_adapter_revision or DEFAULT_IP_ADAPTER_REVISION
    adapter_id = DEFAULT_IP_ADAPTER if Path(config.ip_adapter_id).is_dir() else config.ip_adapter_id
    reference_meta: dict[str, Any] = {
        "requested_mode": config.reference_mode,
        "requested_reference_mode": config.reference_mode,
        "reference_applied": False,
        "actual_reference_applied": False,
        "person_id": None,
        "reference_image_id": None,
        "reference_sha256": None,
        "source_sha256": None,
        "processed_sha256": None,
        "crop_box": None,
        "crop_preprocess": None,
        "preprocessing": None,
        "adapter_id": adapter_id,
        "ip_adapter_id": adapter_id,
        "adapter_revision": adapter_revision,
        "ip_adapter_revision": adapter_revision,
        "adapter_weight": config.ip_adapter_weight,
        "image_encoder": IP_ADAPTER_IMAGE_ENCODER,
        "image_encoder_revision": adapter_revision,
        "reference_strength": config.reference_strength if config.reference_mode != "off" else None,
        "generation_status": "pending",
        "failure_reason": None,
        "visual_quality_status": "unverified",
        "reference_path": None,
        "processed_path": None,
        "references_applied": False,
    }
    try:
        if config.reference_mode != "off":
            if reference_snapshot is not None:
                reference_meta.update(copy.deepcopy(reference_snapshot))
                processed_value = reference_meta.get("processed_path")
                if not processed_value:
                    raise ProjectAssetError("Saved reference snapshot has no processed image path.")
                processed_path = resolve_project_path(project, processed_value)
                if file_sha256(processed_path) != reference_meta.get("processed_sha256"):
                    raise ProjectAssetError("Saved reference snapshot changed; retry blocked.")
                with Image.open(processed_path) as opened:
                    reference_image = opened.convert("RGB").copy()
            else:
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
                reference_meta.update(
                    {
                        "person_id": person_id,
                        "reference_image_id": reference.image_id,
                        "reference_path": str(reference.path),
                        "reference_sha256": processed_meta["source_sha256"],
                        "source_sha256": processed_meta["source_sha256"],
                        "processed_sha256": processed_meta["processed_sha256"],
                        "crop_box": processed_meta["crop_box"],
                        "crop_preprocess": processed_meta,
                        "preprocessing": processed_meta.get("preprocessing") or processed_meta.get("preprocess"),
                        "processed_path": processed_path.relative_to(project.project_dir).as_posix(),
                        "license": IP_ADAPTER_LICENSE,
                    }
                )
    except (OSError, ValueError, ProjectAssetError, GenerationError) as exc:
        project.generation_runs.append(
            {
                "run_id": f"run_{uuid.uuid4().hex}",
                "scene_id": scene.scene_id,
                "config": asdict(config),
                "reference": reference_meta,
                "references_applied": False,
                "generation_status": "blocked",
                "visual_quality_status": "unverified",
                "errors": [str(exc)],
            }
        )
        save_project_atomic(project)
        raise
    run_id = f"run_{uuid.uuid4().hex}"
    indices = candidate_indices if candidate_indices is not None else list(range(max(1, config.candidate_count)))
    snapshot = {
        "run_id": run_id,
        "created_at": utc_now(),
        "scene_id": scene.scene_id,
        "prompt": scene.prompt_user or scene.prompt_auto,
        "negative_prompt": scene.negative_prompt_user or scene.negative_prompt_auto,
        "requested_reference_mode": config.reference_mode,
        "person_id": reference_meta["person_id"],
        "reference_image_id": reference_meta["reference_image_id"],
        "source_sha256": reference_meta["source_sha256"],
        "processed_sha256": reference_meta["processed_sha256"],
        "crop_box": reference_meta["crop_box"],
        "preprocessing": reference_meta["preprocessing"],
        "adapter_id": adapter_id,
        "adapter_revision": adapter_revision,
        "adapter_weight": config.ip_adapter_weight,
        "image_encoder": IP_ADAPTER_IMAGE_ENCODER,
        "image_encoder_revision": adapter_revision,
        "reference_strength": reference_meta["reference_strength"],
        "references_applied": False,
        "actual_reference_applied": False,
        "reference": reference_meta,
        "model_id": config.model_id,
        "revision": config.revision,
        "scheduler": config.scheduler,
        "size": list(config.size),
        "steps": config.steps,
        "guidance_scale": config.guidance_scale,
        "base_seed": config.seed,
        "candidate_indices": list(indices),
        "candidate_results": [],
        "config": asdict(config),
        "outcomes": [],
        "failed_indices": [],
        "errors": [],
        "failure_reason": None,
        "generation_status": "running",
        "visual_quality_status": "unverified",
        "cancelled": False,
    }
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
            actual_reference_applied = bool(getattr(engine, "last_reference_applied", config.reference_mode != "off"))
            if config.reference_mode != "off":
                if not actual_reference_applied:
                    raise GenerationError("Requested reference was not applied; output rejected.")
                snapshot["reference"]["actual_reference_applied"] = actual_reference_applied
                snapshot["reference"]["reference_applied"] = actual_reference_applied
                snapshot["reference"]["references_applied"] = actual_reference_applied
                snapshot["actual_reference_applied"] = actual_reference_applied
                snapshot["references_applied"] = actual_reference_applied
            snapshot["reference"]["generation_status"] = "succeeded"
            snapshot["reference"]["failure_reason"] = None
            candidate_metadata = copy.deepcopy(snapshot)
            candidate_metadata.update(
                {
                    "seed": seed,
                    "candidate_index": index + 1,
                    "generation_status": "succeeded",
                    "failure_reason": None,
                    "visual_quality_status": "unverified",
                    "actual_reference_applied": actual_reference_applied,
                    "references_applied": actual_reference_applied,
                    "metrics": copy.deepcopy(getattr(engine, "last_generation_metrics", {})),
                }
            )
            candidate_metadata["reference"]["generation_status"] = "succeeded"
            candidate_metadata["reference"]["failure_reason"] = None
            temp = project.project_dir / "assets" / ".generation_tmp" / f"{run_id}_{index}.png"
            temp.parent.mkdir(parents=True, exist_ok=True)
            image.save(temp, "PNG")
            candidate = add_generated_candidate(project, scene, temp, candidate_metadata)
            temp.unlink(missing_ok=True)
            snapshot["candidate_results"].append(
                {
                    "candidate_index": index + 1,
                    "seed": seed,
                    "generation_status": "succeeded",
                    "failure_reason": None,
                    "actual_reference_applied": actual_reference_applied,
                    "visual_quality_status": "unverified",
                    "metrics": copy.deepcopy(getattr(engine, "last_generation_metrics", {})),
                }
            )
            snapshot["outcomes"].append({"index": index, "status": "generated", "reference_applied": actual_reference_applied})
            result.candidate_ids.append(candidate.candidate_id)
            result.completed += 1
            if progress:
                progress({"phase": "candidate_done", "scene_id": scene.scene_id, "candidate": index + 1, "total": config.candidate_count, "candidate_id": candidate.candidate_id})
        except GenerationCancelled as exc:
            result.cancelled = True
            result.errors.append(str(exc))
            result.failed_indices.extend(indices[indices.index(index):])
            snapshot["cancelled"] = True
            snapshot["failure_reason"] = str(exc)
            snapshot["reference"]["failure_reason"] = str(exc)
            snapshot["candidate_results"].append(
                {
                    "candidate_index": index + 1,
                    "seed": seed,
                    "generation_status": "cancelled",
                    "failure_reason": str(exc),
                    "actual_reference_applied": False,
                    "visual_quality_status": "unverified",
                    "metrics": copy.deepcopy(getattr(engine, "last_generation_metrics", {})),
                }
            )
            snapshot["outcomes"].append({"index": index, "status": "failed", "error": str(exc), "reference_applied": False})
            break
        except Exception as exc:
            result.failed += 1
            result.failed_indices.append(index)
            result.errors.append(f"candidate {index + 1}: {exc}")
            snapshot["failure_reason"] = str(exc)
            snapshot["reference"]["failure_reason"] = str(exc)
            snapshot["candidate_results"].append(
                {
                    "candidate_index": index + 1,
                    "seed": seed,
                    "generation_status": "failed",
                    "failure_reason": str(exc),
                    "actual_reference_applied": False,
                    "visual_quality_status": "unverified",
                    "metrics": copy.deepcopy(getattr(engine, "last_generation_metrics", {})),
                }
            )
    snapshot["failed_indices"] = list(result.failed_indices)
    snapshot["errors"] = list(result.errors)
    snapshot["generation_status"] = "cancelled" if result.cancelled else ("succeeded" if result.failed == 0 else ("partial" if result.completed else "failed"))
    snapshot["reference"]["generation_status"] = snapshot["generation_status"]
    snapshot["reference"]["failure_reason"] = snapshot["failure_reason"]
    snapshot["loaded_revision"] = getattr(engine, "loaded_revision", None) or config.revision or "model-default"
    return result


def retry_failed_candidates(project: CoverMorphProject, scene: SceneCard, engine: SDXLTextToImageEngine, config: GenerationConfig, failed_indices: list[int], cancel_event: Event, progress: Callable[[dict[str, Any]], None] | None = None) -> GenerationResult:
    """Retry only indexes that were not registered as successful candidates."""
    requested = sorted(set(failed_indices))
    previous = next(
        (
            run
            for run in reversed(project.generation_runs)
            if run.get("scene_id") == scene.scene_id
            and run.get("failed_indices") == requested
            and "config" in run
        ),
        None,
    )
    reference_snapshot = None
    if previous is not None:
        config = GenerationConfig(**copy.deepcopy(previous["config"]))
        scene = copy.deepcopy(scene)
        scene.prompt_user = previous["prompt"]
        scene.negative_prompt_user = previous["negative_prompt"]
        scene.prompt_confirmed = True
        reference_snapshot = previous.get("reference")
        engine.model_id = config.model_id
        engine.revision = config.revision
    return generate_scene_candidates(project, scene, engine, config, cancel_event, progress, requested, reference_snapshot)

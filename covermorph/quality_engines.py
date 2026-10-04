"""Quality Engine V2: one interface over Z-Image-Turbo, FLUX.2-klein-4B and RealVisXL.

The two modern engines run through a stable-diffusion.cpp subprocess (``sd-cli.exe``, CUDA build,
GGUF weights, ``--offload-to-cpu --diffusion-fa``), so the model leaves memory when the process exits.
RealVisXL keeps the proven Diffusers path in ``generation.py``. Only one backend is resident at a time:
callers run engines sequentially and call ``unload()`` before switching.
"""
from __future__ import annotations

import ctypes
import os
import re
import subprocess
import tempfile
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from threading import Event
from typing import Any, Callable

from PIL import Image

SDCPP_RELEASE = "master-929-3f8527a"
QWEN3_4B_FILE = "qwen3_4b/Qwen3-4B-Q8_0.gguf"


class EngineError(RuntimeError):
    pass


class EngineCancelled(EngineError):
    pass


# ------------------------------------------------------------------ resources
@dataclass(slots=True)
class ResourceSnapshot:
    gpu_total_mib: int | None = None
    gpu_used_mib: int | None = None
    gpu_free_mib: int | None = None
    ram_available_mib: int | None = None
    commit_total_mib: int | None = None
    commit_limit_mib: int | None = None

    @property
    def commit_free_mib(self) -> int | None:
        if self.commit_total_mib is None or self.commit_limit_mib is None:
            return None
        return self.commit_limit_mib - self.commit_total_mib

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "commit_free_mib": self.commit_free_mib}


def _nvidia_smi(fields: str) -> list[str] | None:
    try:
        out = subprocess.run(["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10,
                             creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except (OSError, subprocess.SubprocessError):
        return None
    line = out.stdout.strip().splitlines()[0] if out.returncode == 0 and out.stdout.strip() else ""
    return [part.strip() for part in line.split(",")] if line else None


def gpu_used_mib() -> int | None:
    values = _nvidia_smi("memory.used")
    return int(values[0]) if values else None


class _PerformanceInfo(ctypes.Structure):
    _fields_ = [("cb", ctypes.c_ulong), ("CommitTotal", ctypes.c_size_t), ("CommitLimit", ctypes.c_size_t),
                ("CommitPeak", ctypes.c_size_t), ("PhysicalTotal", ctypes.c_size_t),
                ("PhysicalAvailable", ctypes.c_size_t), ("SystemCache", ctypes.c_size_t),
                ("KernelTotal", ctypes.c_size_t), ("KernelPaged", ctypes.c_size_t),
                ("KernelNonpaged", ctypes.c_size_t), ("PageSize", ctypes.c_size_t),
                ("HandleCount", ctypes.c_ulong), ("ProcessCount", ctypes.c_ulong), ("ThreadCount", ctypes.c_ulong)]


def resource_snapshot() -> ResourceSnapshot:
    snap = ResourceSnapshot()
    values = _nvidia_smi("memory.total,memory.used,memory.free")
    if values:
        snap.gpu_total_mib, snap.gpu_used_mib, snap.gpu_free_mib = (int(v) for v in values)
    if os.name == "nt":
        info = _PerformanceInfo()
        info.cb = ctypes.sizeof(info)
        if ctypes.windll.psapi.GetPerformanceInfo(ctypes.byref(info), info.cb):
            page = info.PageSize
            snap.commit_total_mib = info.CommitTotal * page // 2 ** 20
            snap.commit_limit_mib = info.CommitLimit * page // 2 ** 20
            snap.ram_available_mib = info.PhysicalAvailable * page // 2 ** 20
    return snap


class VramSampler:
    """Polls total GPU memory in use. WDDM hides per-process numbers, so the job's share is peak - baseline."""

    def __init__(self, interval: float = 0.25):
        self.interval = interval
        self.baseline = gpu_used_mib()
        self.peak = self.baseline
        self._stop = Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            used = gpu_used_mib()
            if used is not None and (self.peak is None or used > self.peak):
                self.peak = used

    def __enter__(self) -> "VramSampler":
        self._thread.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self._stop.set()
        self._thread.join(timeout=2)

    @property
    def delta_mib(self) -> int | None:
        return None if self.peak is None or self.baseline is None else self.peak - self.baseline


# Thresholds a job must clear before it starts; below them it stays queued instead of crashing.
MEMORY_POLICIES: dict[str, dict[str, Any]] = {
    "interactive_low_memory": {"min_gpu_free_mib": 6500, "min_commit_free_mib": 12000, "min_ram_available_mib": 6000},
    "balanced_idle": {"min_gpu_free_mib": 8000, "min_commit_free_mib": 10000, "min_ram_available_mib": 5000},
    "night_best": {"min_gpu_free_mib": 9000, "min_commit_free_mib": 8000, "min_ram_available_mib": 4000},
}


def check_resources(policy: str, snapshot: ResourceSnapshot | None = None,
                    engine_vram_mib: int | None = None) -> tuple[bool, list[str]]:
    rules = MEMORY_POLICIES[policy]
    snap = snapshot or resource_snapshot()
    reasons: list[str] = []
    need_gpu = max(rules["min_gpu_free_mib"], engine_vram_mib or 0)
    if snap.gpu_free_mib is not None and snap.gpu_free_mib < need_gpu:
        reasons.append(f"GPU 여유 {snap.gpu_free_mib} MiB (필요 {need_gpu} MiB)")
    if snap.commit_free_mib is not None and snap.commit_free_mib < rules["min_commit_free_mib"]:
        reasons.append(f"가상 메모리 여유 {snap.commit_free_mib} MiB (필요 {rules['min_commit_free_mib']} MiB)")
    if snap.ram_available_mib is not None and snap.ram_available_mib < rules["min_ram_available_mib"]:
        reasons.append(f"RAM 여유 {snap.ram_available_mib} MiB (필요 {rules['min_ram_available_mib']} MiB)")
    return not reasons, reasons


# ------------------------------------------------------------------ job/result
# EDIT is internal: the image being edited (always reference 1 of a FLUX.2 edit).
REFERENCE_ROLES = ("PERSON", "PRODUCT", "STYLE", "COMPOSITION", "BACKGROUND", "EDIT")


@dataclass(slots=True)
class Reference:
    path: Path
    role: str

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        self.role = self.role.upper()
        if self.role not in REFERENCE_ROLES:
            raise EngineError(f"Unknown reference role {self.role!r}; use one of {', '.join(REFERENCE_ROLES)}.")


@dataclass(slots=True)
class EngineJob:
    prompt: str                  # compiled, engine-specific
    negative_prompt: str = ""
    width: int = 1280
    height: int = 720
    seed: int = 0
    steps: int | None = None
    guidance: float | None = None
    references: list[Reference] = field(default_factory=list)
    edit_image: Path | None = None
    strength: float | None = None
    # SDXL only: parts in priority order, fitted to the 77-token CLIP limit by the backend.
    prompt_parts: list[str] = field(default_factory=list)
    negative_parts: list[str] = field(default_factory=list)


@dataclass(slots=True)
class EngineResult:
    image: Image.Image
    seconds: float
    peak_vram_mib: int | None
    vram_delta_mib: int | None
    commit_before_mib: int | None
    commit_after_mib: int | None
    command: list[str] = field(default_factory=list)


# ------------------------------------------------------------------ backends
@dataclass(slots=True)
class BackendInfo:
    name: str
    model: str
    repository: str
    quantization: str
    license_id: str
    commercial_ok: bool
    supports_t2i: bool
    supports_single_reference: bool
    supports_multi_reference: bool
    supports_edit: bool
    supports_transparency: bool
    estimated_vram_mib: int
    default_steps: int
    default_guidance: float
    uses_negative_prompt: bool


class ImageBackend:
    info: BackendInfo

    def __init__(self, models_dir: Path):
        self.models_dir = Path(models_dir)

    def required_files(self) -> dict[str, Path]:
        return {}

    def prepare(self) -> None:
        missing = [f"{key}: {path}" for key, path in self.required_files().items() if not path.exists()]
        if missing:
            raise EngineError(f"{self.info.name} is missing files: " + "; ".join(missing))

    def status(self) -> dict[str, Any]:
        files = self.required_files()
        return {"backend": self.info.name, "ready": all(p.exists() for p in files.values()),
                "files": {k: str(p) for k, p in files.items()}, **asdict(self.info)}

    def generate(self, job: EngineJob, cancel: Event | None = None,
                 progress: Callable[[dict[str, Any]], None] | None = None) -> EngineResult:
        raise NotImplementedError

    def edit(self, job: EngineJob, cancel: Event | None = None,
             progress: Callable[[dict[str, Any]], None] | None = None) -> EngineResult:
        raise EngineError(f"{self.info.name} does not support edit")

    def unload(self) -> None:
        """Release models. Subprocess backends hold nothing between jobs."""


# ------------------------------------------------------------------ ASCII paths for sd-cli
# sd-cli opens model/image files with narrow-char APIs: a Korean/Japanese folder name makes it report
# "file not found" (measured 2026-10-04). Non-ASCII folders are handed to it via their 8.3 short name or,
# when short names are disabled, an ASCII directory junction (no admin rights needed, target untouched).
def _short_path(path: Path) -> str | None:
    if os.name != "nt":
        return None
    buffer = ctypes.create_unicode_buffer(32768)
    length = ctypes.windll.kernel32.GetShortPathNameW(str(path), buffer, len(buffer))
    return buffer.value if 0 < length < len(buffer) else None


def ascii_work_root() -> Path:
    candidates = [os.environ.get("COVERMORPH_ASCII_WORKDIR"), os.environ.get("LOCALAPPDATA"),
                  os.environ.get("PUBLIC"), os.environ.get("ProgramData"), tempfile.gettempdir()]
    for base in filter(None, candidates):
        root = Path(base) / "CoverMorph" / "sdcpp"
        if not str(root).isascii():
            continue
        try:
            root.mkdir(parents=True, exist_ok=True)
            return root
        except OSError:
            continue
    raise EngineError("No writable ASCII-only folder for stable-diffusion.cpp; set COVERMORPH_ASCII_WORKDIR.")


def ascii_dir(path: Path) -> Path:
    """An ASCII-only path to the same existing directory."""
    path = Path(path)
    if str(path).isascii():
        return path
    short = _short_path(path)
    if short and short.isascii():
        return Path(short)
    import hashlib
    link = ascii_work_root() / "links" / hashlib.sha1(str(path.resolve()).encode("utf-8")).hexdigest()[:16]
    if not link.exists():
        link.parent.mkdir(parents=True, exist_ok=True)
        if os.path.lexists(link):  # stale junction whose target moved: rmdir removes only the link
            subprocess.run(["cmd", "/c", "rmdir", str(link)], capture_output=True,
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        done = subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(path)], capture_output=True,
                              creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        if done.returncode != 0 or not link.exists():
            raise EngineError(f"Could not create an ASCII junction for {path}; move the models to an ASCII folder "
                              "or set COVERMORPH_ASCII_WORKDIR.")
    return link


def ascii_file(path: Path) -> Path:
    path = Path(path)
    return path if str(path).isascii() else ascii_dir(path.parent) / path.name


# ------------------------------------------------------------------ one diffusion engine per machine
class EngineLock:
    """Machine-wide lock (a locked file under the ASCII work folder) so the studio queue and a youtubesum
    bridge call never run two diffusion engines at the same time. Waits while another process holds it."""

    def __init__(self, cancel: Event | None = None, on_wait: Callable[[str], None] | None = None):
        self.cancel = cancel
        self.on_wait = on_wait
        self._handle = None

    def acquire(self) -> None:
        import msvcrt
        path = ascii_work_root() / "engine.lock"
        handle = open(path, "a+b")
        announced = False
        while True:
            try:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                self._handle = handle
                return
            except OSError:
                if not announced and self.on_wait:
                    self.on_wait("다른 작업이 GPU 엔진을 사용 중 — 대기")
                    announced = True
                if self.cancel is not None and self.cancel.wait(0.5):
                    handle.close()
                    raise EngineCancelled("Cancelled while waiting for the GPU engine.")
                if self.cancel is None:
                    time.sleep(0.5)

    def release(self) -> None:
        if self._handle is not None:
            import msvcrt
            try:
                self._handle.seek(0)
                msvcrt.locking(self._handle.fileno(), msvcrt.LK_UNLCK, 1)
            except OSError:
                pass
            self._handle.close()
            self._handle = None

    def __enter__(self) -> "EngineLock":
        if os.name == "nt":
            self.acquire()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.release()


def default_sdcpp_dir(models_dir: Path) -> Path:
    return Path(os.environ.get("COVERMORPH_SDCPP_DIR") or Path(models_dir) / "quality_v2" / "sdcpp")


_STEP = re.compile(r"\|\s*(\d+)/(\d+)\s*-")


class SdCppBackend(ImageBackend):
    """stable-diffusion.cpp subprocess: the model is loaded per job and freed when the process exits."""

    diffusion_file = ""
    vae_file = ""
    extra_args: tuple[str, ...] = ()

    def __init__(self, models_dir: Path, sdcpp_dir: Path | None = None, offload_to_cpu: bool = True):
        super().__init__(models_dir)
        self.sdcpp_dir = Path(sdcpp_dir) if sdcpp_dir else default_sdcpp_dir(self.models_dir)
        self.offload_to_cpu = offload_to_cpu
        self._process: subprocess.Popen | None = None
        self.last_log = ""

    def required_files(self) -> dict[str, Path]:
        root = self.models_dir / "quality_v2"
        return {"sd_cli": self.sdcpp_dir / "sd-cli.exe", "diffusion_model": root / self.diffusion_file,
                "vae": root / self.vae_file, "llm": root / QWEN3_4B_FILE}

    def build_command(self, job: EngineJob, output: Path, reference_paths: list[Path] | None = None) -> list[str]:
        files = self.required_files()
        command = [str(files["sd_cli"]), "--diffusion-model", str(ascii_file(files["diffusion_model"])),
                   "--vae", str(ascii_file(files["vae"])), "--llm", str(ascii_file(files["llm"])), "-p", job.prompt, "-W", str(job.width), "-H", str(job.height),
                   "--steps", str(job.steps or self.info.default_steps),
                   "--cfg-scale", str(job.guidance if job.guidance is not None else self.info.default_guidance),
                   "-s", str(job.seed), "-o", str(output), "--diffusion-fa", *self.extra_args]
        if job.negative_prompt and self.info.uses_negative_prompt:
            command += ["-n", job.negative_prompt]
        if self.offload_to_cpu:
            command.append("--offload-to-cpu")
        for path in reference_paths if reference_paths is not None else [r.path for r in job.references]:
            command += ["-r", str(path)]
        return command

    @staticmethod
    def _stage_references(job: EngineJob, folder: Path) -> list[Path]:
        """Copy references as PNG into the job's temp folder: absolute ASCII paths whatever the source path."""
        staged = []
        for index, reference in enumerate(job.references, 1):
            if not reference.path.exists():
                raise EngineError(f"Reference image not found: {reference.path}")
            target = folder / f"ref{index}.png"
            with Image.open(reference.path) as image:
                image.convert("RGB").save(target)
            staged.append(target)
        return staged

    def _run(self, job: EngineJob, cancel: Event | None, progress: Callable[[dict[str, Any]], None] | None) -> EngineResult:
        self.prepare()
        cancel = cancel or Event()
        temp_parent = None if tempfile.gettempdir().isascii() else ascii_work_root()
        with tempfile.TemporaryDirectory(prefix="covermorph_sdcpp_", dir=temp_parent) as tmp:
            output = Path(tmp) / "out.png"
            command = self.build_command(job, output, self._stage_references(job, Path(tmp)))
            before = resource_snapshot()
            started = time.perf_counter()
            lines: list[str] = []
            wait_note = (lambda text: progress({"phase": "waiting", "message": text})) if progress else None
            with EngineLock(cancel, wait_note), VramSampler() as sampler:
                self._process = subprocess.Popen(
                    command, cwd=str(self.sdcpp_dir), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, encoding="utf-8", errors="replace",
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                watcher = threading.Thread(target=self._watch_cancel, args=(self._process, cancel), daemon=True)
                watcher.start()
                assert self._process.stdout is not None
                for line in self._process.stdout:
                    lines.append(line.rstrip())
                    match = _STEP.search(line)
                    if progress and match:
                        progress({"phase": "inference", "step": int(match[1]), "steps": int(match[2])})
                code = self._process.wait()
                self._process = None
            after = resource_snapshot()
            self.last_log = "\n".join(lines[-80:])
            if cancel.is_set():
                raise EngineCancelled("Cancelled; sd-cli process terminated.")
            if code != 0 or not output.exists():
                raise EngineError(f"{self.info.name} sd-cli exited with {code}:\n" + "\n".join(lines[-12:]))
            image = Image.open(output).convert("RGB")
            image.load()
        return EngineResult(image, round(time.perf_counter() - started, 2), sampler.peak, sampler.delta_mib,
                            before.commit_total_mib, after.commit_total_mib, command)

    @staticmethod
    def _watch_cancel(process: subprocess.Popen, cancel: Event) -> None:
        while process.poll() is None:
            if cancel.wait(0.3):
                if process.poll() is None:
                    process.kill()
                return

    def generate(self, job, cancel=None, progress=None):
        if job.references and not self.info.supports_single_reference:
            raise EngineError(f"{self.info.name} does not take reference images")
        return self._run(job, cancel, progress)

    def unload(self) -> None:
        if self._process is not None and self._process.poll() is None:
            self._process.kill()
        self._process = None


class ZImageCppBackend(SdCppBackend):
    info = BackendInfo(
        name="zimage_turbo", model="Z-Image-Turbo",
        repository="Tongyi-MAI/Z-Image-Turbo (GGUF: leejet/Z-Image-Turbo-GGUF)",
        quantization="Q6_K", license_id="apache-2.0", commercial_ok=True, supports_t2i=True,
        supports_single_reference=False, supports_multi_reference=False, supports_edit=False,
        supports_transparency=False, estimated_vram_mib=7000, default_steps=8, default_guidance=1.0,
        uses_negative_prompt=False)
    diffusion_file = "zimage_turbo/z_image_turbo-Q6_K.gguf"
    vae_file = "vae/flux1_ae.safetensors"


class Flux2KleinCppBackend(SdCppBackend):
    info = BackendInfo(
        name="flux2_klein_4b", model="FLUX.2-klein-4B",
        repository="black-forest-labs/FLUX.2-klein-4B (GGUF: leejet/FLUX.2-klein-4B-GGUF)",
        quantization="Q8_0", license_id="apache-2.0", commercial_ok=True, supports_t2i=True,
        supports_single_reference=True, supports_multi_reference=True, supports_edit=True,
        supports_transparency=False, estimated_vram_mib=7000, default_steps=4, default_guidance=1.0,
        uses_negative_prompt=False)
    diffusion_file = "flux2_klein_4b/flux-2-klein-4b-Q8_0.gguf"
    vae_file = "vae/flux2_ae.safetensors"
    extra_args = ("--sampling-method", "euler")
    MAX_REFERENCES = 4

    def generate(self, job, cancel=None, progress=None):
        if len(job.references) > self.MAX_REFERENCES:
            raise EngineError(f"FLUX.2-klein takes at most {self.MAX_REFERENCES} references")
        return self._run(job, cancel, progress)

    def edit(self, job, cancel=None, progress=None):
        if job.edit_image is None:
            raise EngineError("edit needs edit_image")
        # FLUX.2 edits are reference-conditioned: the image being edited is reference 1.
        job.references = [Reference(job.edit_image, "EDIT"),
                          *(r for r in job.references if r.role != "EDIT")][: self.MAX_REFERENCES]
        return self._run(job, cancel, progress)


class RealVisDiffusersBackend(ImageBackend):
    """Proven SDXL path (RealVisXL V5.0 + optional IP-Adapter) kept as baseline and fallback."""

    info = BackendInfo(
        name="realvisxl_v5", model="RealVisXL V5.0", repository="SG161222/RealVisXL_V5.0", quantization="fp16",
        license_id="openrail++", commercial_ok=True, supports_t2i=True, supports_single_reference=True,
        supports_multi_reference=False, supports_edit=False, supports_transparency=False,
        estimated_vram_mib=9000, default_steps=30, default_guidance=5.0, uses_negative_prompt=True)

    def __init__(self, models_dir: Path, memory_profile: str = "balanced"):
        super().__init__(models_dir)
        self.memory_profile = memory_profile
        self.engine: Any = None
        self._lock: EngineLock | None = None  # held from load until unload: the pipeline stays resident

    def required_files(self) -> dict[str, Path]:
        return {"checkpoint": self.models_dir / "realvisxl_v5.0"}

    def _engine(self) -> Any:
        if self.engine is None:
            if self._lock is None and os.name == "nt":
                self._lock = EngineLock()
                self._lock.acquire()
            from .generation import SDXLTextToImageEngine
            from .person_quality import MODEL_PROFILES
            from .thumbnail_bridge_ai import MEMORY_PROFILES
            profile = MODEL_PROFILES["photoreal_sdxl"]
            engine = SDXLTextToImageEngine(model_id=str(self.models_dir / profile["folder"]), local_files_only=True)
            engine.memory_profile = {k: v for k, v in MEMORY_PROFILES[self.memory_profile].items() if k != "working_size"}
            engine.variant = profile["variant"]
            engine.scheduler_name = profile["scheduler"]
            engine.load()
            self.engine = engine
        return self.engine

    def generate(self, job, cancel=None, progress=None):
        from .generation import DEFAULT_IP_ADAPTER_REVISION, GENERATION_SIZES, GenerationConfig
        from .thumbnail_bridge_assets import fit_16x9
        self.prepare()
        before = resource_snapshot()
        started = time.perf_counter()
        with VramSampler() as sampler:
            engine = self._engine()
            reference = None
            prompt, negative = job.prompt, job.negative_prompt
            if job.prompt_parts:
                from .thumbnail_bridge_ai import fit_prompt
                tokenizer = getattr(engine.pipeline, "tokenizer", None)
                prompt = fit_prompt(job.prompt_parts, tokenizer, [])
                negative = fit_prompt(job.negative_parts or [job.negative_prompt], tokenizer, [])
            config = GenerationConfig(model_id=engine.model_id, steps=job.steps or self.info.default_steps,
                                      guidance_scale=job.guidance or self.info.default_guidance,
                                      working_size=GENERATION_SIZES["16:9"])
            if job.references:
                reference = Image.open(job.references[0].path).convert("RGB")
                config.reference_mode = "person" if job.references[0].role in ("PERSON", "PRODUCT") else "style"
                config.reference_strength = job.strength if job.strength is not None else 0.6
                config.ip_adapter_id = str(self.models_dir / "ip_adapter")  # same local adapter as the bridge
                config.ip_adapter_revision = DEFAULT_IP_ADAPTER_REVISION
            image = engine.generate_one(prompt, negative, config, job.seed, cancel or Event(),
                                        progress, reference_image=reference)
        after = resource_snapshot()
        if (job.width, job.height) == (1280, 720):
            image = fit_16x9(image)
        elif image.size != (job.width, job.height):
            image = image.resize((job.width, job.height), Image.Resampling.LANCZOS)
        return EngineResult(image, round(time.perf_counter() - started, 2), sampler.peak, sampler.delta_mib,
                            before.commit_total_mib, after.commit_total_mib)

    def unload(self) -> None:
        if self.engine is not None:
            self.engine.unload()
            self.engine = None
        if self._lock is not None:
            self._lock.release()
            self._lock = None


BACKENDS: dict[str, type[ImageBackend]] = {
    "zimage_turbo": ZImageCppBackend,
    "flux2_klein_4b": Flux2KleinCppBackend,
    "realvisxl_v5": RealVisDiffusersBackend,
}


def make_backend(name: str, models_dir: Path, **kwargs: Any) -> ImageBackend:
    try:
        return BACKENDS[name](models_dir, **kwargs)
    except KeyError as exc:
        raise EngineError(f"Unknown backend {name!r}") from exc

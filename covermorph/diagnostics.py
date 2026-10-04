"""'진단 정보 복사': a plain-text report for support. No images; prompts only when the user opts in.

User-specific path parts are replaced with %USERPROFILE% / %LOCALAPPDATA%.
"""
from __future__ import annotations

import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

from . import __version__


def build_id() -> str:
    try:
        from ._build_info import BUILD_COMMIT, BUILD_TIME  # written by the release build
        return f"{BUILD_COMMIT} ({BUILD_TIME})"
    except ImportError:
        pass
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5,
                             cwd=str(Path(__file__).resolve().parents[1]),
                             creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        return f"{out.stdout.strip()} (source)" if out.returncode == 0 else "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _gpu() -> str:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.used,driver_version",
                              "--format=csv,noheader"], capture_output=True, text=True, timeout=10,
                             creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        return out.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "nvidia-smi unavailable"


def build_report(app_root: Path, settings: dict[str, Any], queue: Any = None, include_prompts: bool = False) -> str:
    from . import app_paths
    from .app_paths import sanitize_path
    from .creator_settings import engine_readiness, resolve_models_dir, resolve_output_dir
    from .errors import parse_stored
    from .quality_engines import resource_snapshot

    models = resolve_models_dir(app_root, settings)
    snap = resource_snapshot()
    ready = engine_readiness(models)
    state = app_paths.section("state")
    lines = [
        "CoverMorph Studio 진단 정보",
        f"버전: {__version__} · 빌드: {build_id()} · {'EXE' if getattr(sys, 'frozen', False) else 'source'}",
        f"Windows: {platform.platform()} · Python {platform.python_version()}",
        f"GPU: {_gpu()}",
        f"메모리: GPU 사용 {snap.gpu_used_mib}/{snap.gpu_total_mib} MiB · RAM 여유 {snap.ram_available_mib} MiB · "
        f"커밋 여유 {snap.commit_free_mib} MiB",
        f"품질: {settings.get('quality_label')} · PC 사용: {settings.get('memory_label')} · "
        f"상품 처리: {settings.get('product_mode')}",
        f"모델 폴더: {sanitize_path(str(models))}",
        f"출력 폴더: {sanitize_path(str(resolve_output_dir(app_root, settings)))}",
        f"설정 파일: {sanitize_path(str(app_paths.settings_file()))}",
        "엔진: " + ", ".join(f"{name}={'준비됨' if info['ready'] else '없음'}" for name, info in ready["engines"].items())
        + f", 번역={'준비됨' if ready['translator'] else '없음'}",
    ]
    last = state.get("last_error") or {}
    lines.append(f"마지막 오류: {last.get('code', '-')} {last.get('message', '')} {last.get('time', '')}".rstrip())
    if queue is not None:
        counts: dict[str, int] = {}
        for job in queue.jobs:
            counts[job.state] = counts.get(job.state, 0) + 1
        lines.append(f"대기열: {counts or '비어 있음'} · 일시정지={queue.paused}")
        finished = [job for job in queue.jobs if job.finished]
        if finished:
            job = max(finished, key=lambda j: j.finished)
            result = job.result or {}
            engines = sorted({c.get("engine", "") for c in result.get("candidates") or []})
            code, message = parse_stored(job.error)
            lines.append(f"마지막 작업: {job.state} · 목적 {job.payload.get('purpose')} · 품질 {job.payload.get('quality')} · "
                         f"엔진 {engines or '-'} · {result.get('seconds', '-')}초 · 후보 {len(result.get('candidates') or [])}개"
                         + (f" · 오류 {code}" if code else ""))
            peaks = [c.get("peak_vram_mib") for c in result.get("candidates") or [] if c.get("peak_vram_mib")]
            if peaks:
                lines.append(f"마지막 작업 GPU 피크: {max(peaks)} MiB")
            if include_prompts:
                lines.append(f"마지막 작업 프롬프트: {job.payload.get('prompt', '')}")
    if not include_prompts:
        lines.append("(프롬프트와 이미지는 포함하지 않았습니다)")
    return "\n".join(lines)

"""User-facing errors: a code, a plain Korean message, what to do, and technical details kept out of the way.

The UI shows ``title`` / ``message`` / ``action``; ``details`` (tracebacks, backend output) is written to the log
folder and only shown behind "자세히 보기". Job failures are stored as ``"CODE|message"`` so the queue can show
the message and the diagnostics can report the code.
"""
from __future__ import annotations

import shutil
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

MIN_FREE_OUTPUT_GIB = 2.0

CODES: dict[str, tuple[str, str]] = {
    # code: (title, what to do)
    "MODELS_MISSING": ("모델 파일이 없습니다", "설정 탭에서 모델 폴더를 다시 지정하거나 모델 준비 스크립트를 실행하세요."),
    "MODELS_DIR_INVALID": ("모델 폴더를 찾을 수 없습니다", "설정 탭에서 올바른 모델 폴더(quality_v2 폴더가 들어 있는 곳)를 선택하세요."),
    "DISK_LOW": ("저장 공간이 부족합니다", "출력 폴더가 있는 드라이브의 공간을 확보하거나 설정에서 다른 출력 폴더를 고르세요."),
    "MEMORY_LOW": ("메모리가 부족합니다", "다른 프로그램을 닫을 필요는 없습니다. 작업은 대기열에서 자원이 생길 때까지 기다립니다."),
    "VRAM_LOW": ("그래픽 메모리가 부족합니다", "영상 편집 등 GPU를 쓰는 프로그램이 끝나면 자동으로 다시 시작합니다."),
    "BACKEND_CRASH": ("이미지 엔진이 중간에 멈췄습니다", "'실패 재시도'를 누르세요. 반복되면 '빠른 미리보기'나 '작업 중 PC 우선'으로 시도하세요."),
    "TRANSLATION_FAILED": ("프롬프트 번역을 건너뛰었습니다", "영어로 프롬프트를 쓰면 가장 정확합니다. 작업은 계속 진행됩니다."),
    "PROJECT_CORRUPT": ("작업 파일이 손상되었습니다", "손상된 파일은 따로 보관했습니다. 같은 설정으로 다시 생성하세요."),
    "QUEUE_CORRUPT": ("대기열 파일이 손상되었습니다", "손상된 대기열은 따로 보관하고 빈 대기열로 시작했습니다."),
    "OUTPUT_WRITE": ("결과를 저장하지 못했습니다", "출력 폴더에 쓰기 권한이 있는지, 다른 프로그램이 파일을 열고 있지 않은지 확인하세요."),
    "REFERENCE_INVALID": ("레퍼런스 이미지를 열 수 없습니다", "다른 이미지(JPG/PNG/WEBP)를 선택하거나 파일이 옮겨지지 않았는지 확인하세요."),
    "CANCELLED": ("작업을 취소했습니다", ""),
    "UNKNOWN": ("예상하지 못한 오류가 발생했습니다", "'진단 정보 복사'로 정보를 복사해 문의하세요. 대기열과 결과는 그대로 남아 있습니다."),
}


@dataclass
class UserError(Exception):
    code: str
    message: str
    details: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        super().__init__(self.message)

    @property
    def title(self) -> str:
        return CODES.get(self.code, CODES["UNKNOWN"])[0]

    @property
    def action(self) -> str:
        return CODES.get(self.code, CODES["UNKNOWN"])[1]

    def stored(self) -> str:
        return f"{self.code}|{self.message}"


def parse_stored(text: str) -> tuple[str, str]:
    """'CODE|message' (new) or a plain old error string."""
    if text and "|" in text and text.split("|", 1)[0] in CODES:
        code, message = text.split("|", 1)
        return code, message
    return ("UNKNOWN" if text else ""), text


def classify(exc: BaseException) -> UserError:
    """Map any failure to a UserError. Technical text goes to ``details`` only."""
    if isinstance(exc, UserError):
        return exc
    details = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    name = type(exc).__name__
    text = str(exc)
    lower = text.casefold()
    if name in ("EngineCancelled", "GenerationCancelled"):
        return UserError("CANCELLED", "취소되었습니다.", details)
    if "missing files" in lower or "is not installed" in lower or "no image engine is installed" in lower:
        return UserError("MODELS_MISSING", "필요한 모델 파일을 찾지 못했습니다.", details)
    if "sd-cli exited" in lower or "terminated" in lower or name == "EngineError" and "exited" in lower:
        return UserError("BACKEND_CRASH", "이미지 엔진 프로세스가 정상적으로 끝나지 않았습니다.", details)
    if "out of memory" in lower or name == "OutOfMemoryError":
        return UserError("VRAM_LOW", "그래픽 메모리가 부족해 생성하지 못했습니다.", details)
    if isinstance(exc, (PermissionError, IsADirectoryError)) or "permission denied" in lower:
        return UserError("OUTPUT_WRITE", "파일을 저장할 권한이 없습니다.", details)
    if isinstance(exc, OSError) and getattr(exc, "errno", None) == 28:
        return UserError("DISK_LOW", "디스크가 가득 찼습니다.", details)
    if name in ("UnidentifiedImageError",) or "cannot identify image" in lower or "reference image not found" in lower:
        return UserError("REFERENCE_INVALID", "레퍼런스 이미지를 열지 못했습니다.", details)
    if name == "JSONDecodeError":
        return UserError("PROJECT_CORRUPT", "작업 정보 파일을 읽지 못했습니다.", details)
    return UserError("UNKNOWN", f"{name}: {text.splitlines()[0][:160]}" if text else name, details)


def write_details(error: UserError, label: str) -> Path | None:
    """Keep the technical details in the log folder; return the file for "자세히 보기"."""
    try:
        from .app_paths import logs_dir
        path = logs_dir() / f"error_{time.strftime('%Y%m%d_%H%M%S')}_{label}.txt"
        path.write_text(f"{error.code}\n{error.message}\n\n{error.details}", encoding="utf-8")
        return path
    except OSError:
        return None


def remember_last_error(error: UserError, details_path: Path | None) -> None:
    try:
        from . import app_paths
        app_paths.update_section("state", {"last_error": {"code": error.code, "message": error.message,
                                                          "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                                                          "details": str(details_path or "")}})
    except OSError:
        pass


# ------------------------------------------------------------------ preflight
def preflight(payload: dict[str, Any], models_dir: Path, output_root: Path) -> None:
    """Checks that should fail fast with a clear message (resource shortages wait in the queue instead)."""
    from PIL import Image

    from .creator_settings import engine_readiness
    if not Path(models_dir).is_dir():
        raise UserError("MODELS_DIR_INVALID", f"모델 폴더가 없습니다: {models_dir}")
    ready = engine_readiness(Path(models_dir))["engines"]
    if not any(info["ready"] for info in ready.values()):
        raise UserError("MODELS_MISSING", "이 모델 폴더에서 사용할 수 있는 이미지 엔진을 찾지 못했습니다.",
                        "\n".join(m for info in ready.values() for m in info["missing"]))
    if payload.get("kind") == "edit" and not ready["flux2_klein_4b"]["ready"]:
        raise UserError("MODELS_MISSING", "편집에는 FLUX.2-klein 모델이 필요합니다.")
    output_root = Path(output_root)
    probe = output_root if output_root.exists() else next((p for p in output_root.parents if p.exists()), output_root)
    try:
        free = shutil.disk_usage(probe).free / 1024 ** 3
    except OSError:
        free = None
    if free is not None and free < MIN_FREE_OUTPUT_GIB:
        raise UserError("DISK_LOW", f"출력 드라이브 여유 공간이 {free:.1f} GB입니다 (최소 {MIN_FREE_OUTPUT_GIB:.0f} GB 필요).")
    try:
        output_root.mkdir(parents=True, exist_ok=True)
        test = output_root / ".write_test"
        test.write_text("ok", encoding="utf-8")
        test.unlink()
    except OSError as exc:
        raise UserError("OUTPUT_WRITE", f"출력 폴더에 쓸 수 없습니다: {output_root}", str(exc)) from exc
    for ref in payload.get("references") or []:
        validate_reference(Path(ref["path"]))
    if payload.get("edit_image"):
        validate_reference(Path(payload["edit_image"]))


def validate_reference(path: Path) -> None:
    from PIL import Image, UnidentifiedImageError
    if not path.exists():
        raise UserError("REFERENCE_INVALID", f"레퍼런스 파일이 없습니다: {path.name}")
    try:
        with Image.open(path) as opened:
            opened.verify()
        with Image.open(path) as opened:
            if min(opened.size) < 64:
                raise UserError("REFERENCE_INVALID", f"레퍼런스가 너무 작습니다({opened.size[0]}×{opened.size[1]}): {path.name}")
    except (UnidentifiedImageError, OSError, SyntaxError) as exc:
        raise UserError("REFERENCE_INVALID", f"이미지로 열 수 없는 파일입니다: {path.name}", str(exc)) from exc

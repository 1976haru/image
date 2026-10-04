import os
import traceback
from datetime import datetime
from pathlib import Path


def log_dir(root: Path) -> Path:
    """Per-user logs (%LOCALAPPDATA%\\CoverMorphStudio\\logs) so a rebuild never loses them."""
    try:
        from .app_paths import logs_dir
        return logs_dir()
    except OSError:
        primary = root / "logs"
        primary.mkdir(parents=True, exist_ok=True)
        return primary


def write_log(root: Path, message: str) -> None:
    log_dir_path = log_dir(root)
    path = log_dir_path / f"{datetime.now():%Y-%m-%d}.log"
    with path.open("a", encoding="utf-8") as f:
        f.write(f"[{datetime.now():%H:%M:%S}] {message}\n")


def write_exception(root: Path, prefix: str, exc: BaseException) -> None:
    write_log(root, f"{prefix}: {exc}\n{traceback.format_exc()}")

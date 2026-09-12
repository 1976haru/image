from pathlib import Path
from datetime import datetime
import traceback

def write_log(root: Path, message: str):
    log_dir = root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    p = log_dir / f"{datetime.now():%Y-%m-%d}.log"
    with p.open("a", encoding="utf-8") as f:
        f.write(f"[{datetime.now():%H:%M:%S}] {message}\n")

def write_exception(root: Path, prefix: str, exc: Exception):
    write_log(root, f"{prefix}: {exc}\n{traceback.format_exc()}")

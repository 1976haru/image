"""Headless runtime for the CoverMorph <-> thumbnail editor bridge."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from .generation import DEFAULT_SDXL_MODEL, detect_generation_environment
from .thumbnail_bridge import (
    ThumbnailBridgeError,
    ThumbnailBridgeRequest,
    ThumbnailBridgeResponse,
    read_request_json,
    write_response_json,
)


def _app_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[1]


def _status_payload(request: ThumbnailBridgeRequest) -> dict[str, Any]:
    model_id = str(request.options.get("model_id") or DEFAULT_SDXL_MODEL)
    env = detect_generation_environment(_app_root(), model_id=model_id)
    project_dir = request.project_path
    writable = False
    try:
        project_dir.mkdir(parents=True, exist_ok=True)
        probe = project_dir / ".thumbnail_bridge_write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
        writable = True
    except OSError:
        writable = False

    edit_capabilities = ["recompose"]
    if env.get("status") == "ready":
        edit_capabilities.append("regenerate")
    if env.get("reference_status") == "ready":
        edit_capabilities.append("reference_regenerate")

    return {
        "bridge_protocol_version": 1,
        "project_dir_writable": writable,
        "environment": env,
        "capabilities": {
            "status": True,
            "generate": env.get("status") == "ready" and writable,
            "edit": edit_capabilities,
            "headless": True,
            "atomic_output": True,
        },
    }


def handle_request(request: ThumbnailBridgeRequest) -> ThumbnailBridgeResponse:
    if request.action == "status":
        payload = _status_payload(request)
        ready = bool(payload["project_dir_writable"])
        return ThumbnailBridgeResponse(
            request_id=request.request_id,
            action=request.action,
            project_dir=request.project_dir,
            ok=ready,
            message="CoverMorph thumbnail bridge status.",
            outputs={"status": payload},
            warnings=[] if ready else ["project_dir is not writable"],
            error_code="" if ready else "PROJECT_NOT_WRITABLE",
        )

    # generate/edit are deliberately not faked here. Claude Code will wire them
    # to the real existing SDXL/composition stack on this development branch.
    return ThumbnailBridgeResponse(
        request_id=request.request_id,
        action=request.action,
        project_dir=request.project_dir,
        ok=False,
        message=f"Real bridge action '{request.action}' is not wired yet.",
        error_code="ACTION_NOT_IMPLEMENTED",
    )


def run_bridge_cli(stdin_text: str | None = None) -> int:
    text = sys.stdin.read() if stdin_text is None else stdin_text
    try:
        request = read_request_json(text)
        response = handle_request(request)
    except ThumbnailBridgeError as exc:
        response = ThumbnailBridgeResponse(
            request_id="",
            action="status",
            project_dir="",
            ok=False,
            message=str(exc),
            error_code="INVALID_REQUEST",
        )
    except Exception as exc:  # structured boundary: never leak traceback to stdout
        print(f"thumbnail bridge error: {exc}", file=sys.stderr)
        response = ThumbnailBridgeResponse(
            request_id="",
            action="status",
            project_dir="",
            ok=False,
            message=str(exc),
            error_code="INTERNAL_ERROR",
        )
    sys.stdout.write(write_response_json(response))
    sys.stdout.flush()
    return 0 if response.ok else 2

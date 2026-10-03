"""Thumbnail bridge protocol primitives shared by CLI and GUI integrations.

This module deliberately contains no model invocation. It validates/normalizes the
wire contract so generation/editing can be added without coupling the protocol to
one AI backend.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PROTOCOL_VERSION = 1
VALID_ACTIONS = {"generate", "edit", "status"}


class ThumbnailBridgeError(ValueError):
    pass


@dataclass(slots=True)
class ThumbnailBridgeRequest:
    protocol_version: int = PROTOCOL_VERSION
    request_id: str = ""
    action: str = "status"
    project_dir: str = ""
    channel: str = ""
    story_type: str = ""
    episode: str = ""
    title: str = ""
    subtitle: str = ""
    prompt: str = ""
    edit_instruction: str = ""
    preferred_typography: str = ""
    options: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ThumbnailBridgeRequest":
        if not isinstance(data, dict):
            raise ThumbnailBridgeError("Bridge request must be a JSON object.")
        req = cls(
            protocol_version=int(data.get("protocol_version") or PROTOCOL_VERSION),
            request_id=str(data.get("request_id") or ""),
            action=str(data.get("action") or "status"),
            project_dir=str(data.get("project_dir") or ""),
            channel=str(data.get("channel") or ""),
            story_type=str(data.get("story_type") or ""),
            episode=str(data.get("episode") or ""),
            title=str(data.get("title") or ""),
            subtitle=str(data.get("subtitle") or ""),
            prompt=str(data.get("prompt") or ""),
            edit_instruction=str(data.get("edit_instruction") or ""),
            preferred_typography=str(data.get("preferred_typography") or ""),
            options=dict(data.get("options") or {}),
        )
        req.validate()
        return req

    def validate(self) -> None:
        if self.protocol_version != PROTOCOL_VERSION:
            raise ThumbnailBridgeError(
                f"Unsupported protocol_version={self.protocol_version}; expected {PROTOCOL_VERSION}."
            )
        if self.action not in VALID_ACTIONS:
            raise ThumbnailBridgeError(f"Unsupported action: {self.action}")
        if not self.project_dir:
            raise ThumbnailBridgeError("project_dir is required.")
        if self.action == "edit" and not self.edit_instruction.strip():
            raise ThumbnailBridgeError("edit_instruction is required for edit action.")

    @property
    def project_path(self) -> Path:
        return Path(self.project_dir).expanduser()


@dataclass(slots=True)
class ThumbnailBridgeResponse:
    request_id: str
    action: str
    project_dir: str
    ok: bool = True
    outputs: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    message: str = ""
    error_code: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "protocol_version": PROTOCOL_VERSION,
            "request_id": self.request_id,
            "ok": self.ok,
            "action": self.action,
            "project_dir": self.project_dir,
            "outputs": dict(self.outputs),
            "warnings": list(self.warnings),
            "message": self.message,
        }
        if self.error_code:
            payload["error_code"] = self.error_code
        if self.details:
            payload["details"] = self.details
        return payload


def read_request_json(text: str) -> ThumbnailBridgeRequest:
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ThumbnailBridgeError(f"Invalid request JSON: {exc}") from exc
    return ThumbnailBridgeRequest.from_dict(data)


def write_response_json(response: ThumbnailBridgeResponse) -> str:
    return json.dumps(response.to_dict(), ensure_ascii=False, separators=(",", ":"))


def standard_output_paths(project_dir: Path) -> dict[str, Path]:
    return {
        "canvas_clean": project_dir / "canvas_clean.png",
        "preview_reference": project_dir / "preview_reference.png",
        "subject_boxes": project_dir / "subject_boxes.json",
        "safe_zones": project_dir / "safe_zones.json",
        "palette": project_dir / "palette.json",
        "composition": project_dir / "composition.json",
        "project_manifest": project_dir / "project_manifest.json",
    }

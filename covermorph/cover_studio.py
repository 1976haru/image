from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image, ImageColor, ImageDraw, ImageFont

from .project import CandidateRecord, CoverMorphProject, resolve_project_path, save_project_atomic, utc_now
from .settings import next_numbered_path

CANVAS_SIZE = 1400
TEMPLATES = ("상단 제목", "중앙 제목", "하단 제목")
FONT_EXTENSIONS = {".ttf", ".ttc", ".otf"}


class CoverRenderError(ValueError):
    pass


@dataclass(slots=True)
class TextLayer:
    text: str = ""
    visible: bool = True
    font_path: str = ""
    size: int = 96
    color: str = "#FFFFFF"
    align: str = "center"
    x: float = 0.5
    y: float = 0.15
    stroke_width: int = 3
    stroke_color: str = "#000000"
    shadow: bool = True

    @classmethod
    def from_dict(cls, value: Any, default: "TextLayer") -> "TextLayer":
        data = dict(value or {})
        return cls(**{key: data.get(key, getattr(default, key)) for key in cls.__dataclass_fields__})


@dataclass(slots=True)
class CoverEdit:
    template: str = "상단 제목"
    title: TextLayer = field(default_factory=lambda: TextLayer(size=118, y=0.13))
    subtitle: TextLayer = field(default_factory=lambda: TextLayer(size=58, y=0.29))
    label: TextLayer = field(default_factory=lambda: TextLayer(size=42, y=0.91))

    @classmethod
    def from_dict(cls, value: Any) -> "CoverEdit":
        data = dict(value or {})
        base = cls()
        return cls(
            template=str(data.get("template") or base.template),
            title=TextLayer.from_dict(data.get("title"), base.title),
            subtitle=TextLayer.from_dict(data.get("subtitle"), base.subtitle),
            label=TextLayer.from_dict(data.get("label"), base.label),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def system_font_paths() -> list[Path]:
    roots = []
    if os.name == "nt":
        roots.append(Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts")
    else:
        roots.extend((Path("/usr/share/fonts"), Path("/usr/local/share/fonts")))
    return sorted(
        {
            path
            for root in roots
            if root.is_dir()
            for path in root.rglob("*")
            if path.suffix.lower() in FONT_EXTENSIONS
        }
    )


def font_supports_text(font_path: str | Path, text: str) -> tuple[bool, list[str]]:
    required = {ord(char) for char in text if not char.isspace()}
    if not required:
        return True, []
    try:
        from fontTools.ttLib import TTFont

        font = TTFont(str(font_path), fontNumber=0, lazy=True)
        cmap = set()
        for table in font["cmap"].tables:
            cmap.update(table.cmap)
        font.close()
        missing = sorted(chr(code) for code in required - cmap)
        return not missing, missing
    except (ImportError, KeyError, OSError):
        try:
            font = ImageFont.truetype(str(font_path), 32)
            replacement = font.getmask(chr(0x10FFFF))
            replacement_signature = (replacement.size, replacement.getbbox(), bytes(replacement))
            missing = []
            for code in required:
                mask = font.getmask(chr(code))
                signature = (mask.size, mask.getbbox(), bytes(mask))
                if mask.getbbox() is None or signature == replacement_signature:
                    missing.append(chr(code))
            return not missing, missing
        except OSError:
            return False, sorted(chr(code) for code in required)


def choose_font(text: str, preferred: str = "") -> Path:
    paths = system_font_paths()
    if preferred:
        selected = Path(preferred)
        if selected.is_file() and font_supports_text(selected, text)[0]:
            return selected
    for path in paths:
        if font_supports_text(path, text)[0]:
            return path
    raise CoverRenderError(
        "입력 문자를 모두 지원하는 폰트가 없습니다. 한글·일본어 지원 폰트를 설치하거나 선택해 주세요."
    )


def apply_template(edit: CoverEdit, template: str) -> None:
    if template not in TEMPLATES:
        raise CoverRenderError(f"지원하지 않는 배치 템플릿입니다: {template}")
    edit.template = template
    positions = {
        "상단 제목": (0.13, 0.29, 0.91),
        "중앙 제목": (0.43, 0.57, 0.91),
        "하단 제목": (0.70, 0.83, 0.94),
    }
    edit.title.y, edit.subtitle.y, edit.label.y = positions[template]


def _wrapped_lines(
    draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont, width: int
) -> list[str]:
    lines: list[str] = []
    for paragraph in text.splitlines() or [""]:
        current = ""
        for char in paragraph:
            trial = current + char
            if current and draw.textbbox((0, 0), trial, font=font, stroke_width=0)[2] > width:
                lines.append(current)
                current = char
            else:
                current = trial
        lines.append(current)
    return lines


def _layout_layer(
    draw: ImageDraw.ImageDraw, layer: TextLayer, canvas: int
) -> tuple[ImageFont.FreeTypeFont, str, tuple[int, int, int, int]]:
    font_path = choose_font(layer.text, layer.font_path)
    max_width = int(canvas * 0.88)
    for size in range(min(layer.size, 260), 23, -2):
        font = ImageFont.truetype(str(font_path), size)
        rendered = "\n".join(_wrapped_lines(draw, layer.text, font, max_width))
        spacing = max(4, size // 5)
        box = draw.multiline_textbbox(
            (0, 0), rendered, font=font, spacing=spacing, align=layer.align, stroke_width=layer.stroke_width
        )
        if box[2] - box[0] <= max_width and box[3] - box[1] <= int(canvas * 0.25):
            return font, rendered, box
    raise CoverRenderError(f"문구가 최소 크기에도 들어가지 않습니다. 문구를 수정해 주세요: {layer.text}")


def render_cover(
    background: Image.Image, edit: CoverEdit, size: int = CANVAS_SIZE
) -> tuple[Image.Image, list[str]]:
    image = background.convert("RGB").resize((size, size), Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(image)
    warnings: list[str] = []
    placed: list[tuple[int, int, int, int]] = []
    for name, layer in (("제목", edit.title), ("부제", edit.subtitle), ("채널명", edit.label)):
        if not layer.visible or not layer.text.strip():
            continue
        font, rendered, box = _layout_layer(draw, layer, size)
        width, height = box[2] - box[0], box[3] - box[1]
        x = int(layer.x * size - width / 2)
        y = int(layer.y * size - height / 2)
        x = min(max(35, x), size - width - 35)
        y = min(max(35, y), size - height - 35)
        actual = (x, y, x + width, y + height)
        if any(
            not (
                actual[2] + 12 < old[0]
                or actual[0] > old[2] + 12
                or actual[3] + 12 < old[1]
                or actual[1] > old[3] + 12
            )
            for old in placed
        ):
            raise CoverRenderError(f"{name} 문구가 다른 문구와 겹칩니다. 위치나 크기를 조정해 주세요.")
        placed.append(actual)
        anchor_x = x if layer.align == "left" else x + width // 2 if layer.align == "center" else x + width
        anchor = "ma" if layer.align == "center" else "la" if layer.align == "left" else "ra"
        if layer.shadow:
            draw.multiline_text(
                (anchor_x + 5, y + 7),
                rendered,
                font=font,
                fill="#000000A0",
                spacing=max(4, font.size // 5),
                align=layer.align,
                anchor=anchor,
            )
        draw.multiline_text(
            (anchor_x, y),
            rendered,
            font=font,
            fill=ImageColor.getrgb(layer.color),
            spacing=max(4, font.size // 5),
            align=layer.align,
            stroke_width=layer.stroke_width,
            stroke_fill=ImageColor.getrgb(layer.stroke_color),
            anchor=anchor,
        )
    return image, warnings


def candidate_edit(candidate: CandidateRecord) -> CoverEdit:
    return CoverEdit.from_dict(candidate.generation_metadata.get("cover_edit"))


def save_candidate_edit(project: CoverMorphProject, candidate: CandidateRecord, edit: CoverEdit) -> None:
    candidate.generation_metadata["cover_edit"] = edit.to_dict()
    candidate.updated_at = utc_now()
    project.selected_candidate_ids = [candidate.candidate_id]
    save_project_atomic(project)


def export_cover(
    project: CoverMorphProject,
    candidate: CandidateRecord,
    edit: CoverEdit,
    output_dir: Path,
    stem: str = "cover",
) -> dict[str, Path]:
    source = resolve_project_path(project, candidate.original_path)
    if not source.is_file():
        raise CoverRenderError(f"글자 없는 원본을 찾을 수 없습니다: {source}")
    output_dir.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as opened:
        original = opened.convert("RGB")
        rendered, warnings = render_cover(original, edit, CANVAS_SIZE)
        original_size = list(original.size)
        png_path = next_numbered_path(output_dir / f"{stem}_textless.png")
        jpg_path = next_numbered_path(output_dir / f"{stem}_1400.jpg")
        meta_path = next_numbered_path(output_dir / f"{stem}_metadata.json")
        original.save(png_path, "PNG")
        rendered.save(jpg_path, "JPEG", quality=95, subsampling=0, optimize=True)
    metadata = {
        "candidate_id": candidate.candidate_id,
        "source_path": str(source),
        "textless_png": str(png_path),
        "cover_jpg": str(jpg_path),
        "source_size": original_size,
        "output_size": [CANVAS_SIZE, CANVAS_SIZE],
        "resize_method": "Pillow LANCZOS resize; AI upscaling/detail restoration not used",
        "cover_edit": edit.to_dict(),
        "generation": candidate.generation_metadata,
        "warnings": warnings,
        "saved_at": utc_now(),
    }
    meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    candidate.generation_metadata["last_export"] = {
        key: str(value)
        for key, value in {"textless_png": png_path, "cover_jpg": jpg_path, "metadata": meta_path}.items()
    }
    save_candidate_edit(project, candidate, edit)
    return {"textless_png": png_path, "cover_jpg": jpg_path, "metadata": meta_path}


def copy_textless(
    project: CoverMorphProject, candidate: CandidateRecord, output_dir: Path, stem: str = "cover"
) -> Path:
    source = resolve_project_path(project, candidate.original_path)
    if not source.is_file():
        raise CoverRenderError(f"글자 없는 원본을 찾을 수 없습니다: {source}")
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = next_numbered_path(output_dir / f"{stem}_textless.png")
    shutil.copy2(source, destination)
    return destination


def quick_project_dir(root: Path) -> Path:
    return root / "projects" / "quick_cover_current"


def atomic_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", suffix=".tmp", dir=path.parent, delete=False
    ) as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2)
        temp = Path(stream.name)
    temp.replace(path)

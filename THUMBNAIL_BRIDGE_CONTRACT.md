# Thumbnail Bridge Contract v1

This document defines the local file/subprocess contract between CoverMorph Studio (`1976haru/image`) and YouTube Dynamic Thumbnail Studio (`1976haru/youtubesum`).

## Goals

- CoverMorph creates or edits a **textless 16:9 canvas**.
- CoverMorph exports subject/safe-zone/palette/composition sidecars.
- YouTube Dynamic Thumbnail Studio renders typography and A/B/C thumbnail variants.
- Both applications remain independently usable.
- Korean/Japanese/space-containing Windows paths are first-class.
- Existing GUI behavior must remain unchanged when bridge arguments are not supplied.

## Invocation

Preferred executable form:

```text
CoverMorphStudio.exe --thumbnail-bridge-json
```

Development form:

```text
python app.py --thumbnail-bridge-json
```

The process reads exactly one UTF-8 JSON request from stdin and writes exactly one UTF-8 JSON response to stdout.

Human logs/errors go to stderr or normal CoverMorph log files. stdout in bridge mode is reserved for the JSON response.

## Request

```json
{
  "protocol_version": 1,
  "request_id": "uuid-or-caller-id",
  "action": "generate",
  "project_dir": "D:\\projects\\tokyo_001",
  "channel": "Tokyo Chill",
  "story_type": "남자 이야기",
  "episode": "EP.001",
  "title": "目が合っただけなのに",
  "subtitle": "A quiet story in the city",
  "prompt": "textless Tokyo train interior, cinematic...",
  "edit_instruction": "",
  "preferred_typography": "Japanese Impact",
  "options": {
    "ratio": "16:9",
    "width": 1280,
    "height": 720,
    "textless": true,
    "preserve_people": true,
    "prefer_text_space": "left"
  }
}
```

### action

- `generate`: create a new clean thumbnail background and sidecars.
- `edit`: edit/recompose the existing clean canvas according to `edit_instruction`.
- `status`: validate bridge readiness and project assets without generation.

## Response

Success:

```json
{
  "protocol_version": 1,
  "request_id": "same-as-request",
  "ok": true,
  "action": "generate",
  "project_dir": "D:\\projects\\tokyo_001",
  "outputs": {
    "canvas_clean": "canvas_clean.png",
    "preview_reference": "preview_reference.png",
    "subject_boxes": "subject_boxes.json",
    "safe_zones": "safe_zones.json",
    "palette": "palette.json",
    "composition": "composition.json",
    "project_manifest": "project_manifest.json"
  },
  "warnings": [],
  "message": "Generated thumbnail bridge assets."
}
```

Failure:

```json
{
  "protocol_version": 1,
  "request_id": "same-as-request",
  "ok": false,
  "action": "generate",
  "project_dir": "D:\\projects\\tokyo_001",
  "error_code": "MODEL_NOT_READY",
  "message": "SDXL model is not prepared.",
  "warnings": []
}
```

Bridge mode should return a non-zero process exit code on `ok=false`.

## Project outputs

### canvas_clean.png

- Required for successful `generate` / `edit`.
- Textless.
- Preferred output size: 1280x720.
- PNG.
- No channel title, logo, episode text, watermark, subtitles, or generated lettering.
- Keep important faces/person pixels usable for later thumbnail composition.

### preview_reference.png

Optional visual reference. May be a generated candidate or prior thumbnail reference. It is never treated as the clean typography canvas.

### subject_boxes.json

Coordinates are normalized 0..1.

```json
{
  "version": 1,
  "canvas": {"width": 1280, "height": 720},
  "subjects": [
    {
      "id": "main",
      "role": "protagonist",
      "kind": "person",
      "x": 0.18,
      "y": 0.16,
      "w": 0.25,
      "h": 0.64,
      "confidence": 0.93
    },
    {
      "id": "secondary",
      "role": "counterpart",
      "kind": "person",
      "x": 0.67,
      "y": 0.20,
      "w": 0.21,
      "h": 0.57,
      "confidence": 0.90
    }
  ]
}
```

Do not infer gender from computer vision. Roles come from project/user intent when known.

### safe_zones.json

```json
{
  "version": 1,
  "preferred_text_regions": [
    {
      "name": "left_lower",
      "x": 0.04,
      "y": 0.56,
      "w": 0.38,
      "h": 0.31,
      "score": 0.91
    }
  ],
  "avoid_regions": [
    {
      "name": "protagonist_face",
      "x": 0.21,
      "y": 0.17,
      "w": 0.15,
      "h": 0.19
    }
  ]
}
```

### palette.json

```json
{
  "version": 1,
  "dominant": ["#E9E4D8", "#7EB8D4", "#253747"],
  "accent": ["#82D9F7", "#FFD76C", "#F3A3C5"],
  "recommended_text_light": "#FFFDF8",
  "recommended_text_dark": "#152433",
  "recommended_stroke": "#183246",
  "recommended_shadow": "#0C1420"
}
```

### composition.json

```json
{
  "version": 1,
  "canvas": {"width": 1280, "height": 720},
  "protagonist_side": "left",
  "counterpart_side": "right",
  "recommended_text_side": "left",
  "scene_type": "relationship",
  "notes": ["Preserve skyline context", "Do not cover either face"]
}
```

### project_manifest.json

```json
{
  "version": 1,
  "source_app": "CoverMorph Studio",
  "bridge_protocol_version": 1,
  "channel": "Tokyo Chill",
  "story_type": "남자 이야기",
  "episode": "EP.001",
  "title": "目が合っただけなのに",
  "subtitle": "A quiet story in the city",
  "preferred_typography": "Japanese Impact",
  "generated_at": "ISO-8601 timestamp"
}
```

## Editing behavior

For `edit`, the existing `canvas_clean.png` should be used as the current project canvas unless the request references another explicit source.

Typical edit instructions:

- "인물을 오른쪽으로 조금 이동하고 왼쪽 문구 공간을 넓혀줘"
- "배경을 조금 더 어둡게 하고 얼굴은 그대로 유지"
- "도쿄 야경으로 바꾸되 두 사람의 포즈는 유지"
- "왼쪽 40%를 텍스트용으로 단순하게 만들어줘"

After editing, rewrite the canvas and all sidecars that changed.

## Atomicity

Write outputs to temporary names and replace final files only after successful completion. A failed generation/edit must not destroy the previous valid canvas/sidecars.

## Licensing / model policy

Do not bundle third-party model weights or proprietary font assets into the repository unless their licenses explicitly permit redistribution. Keep optional AI models external as CoverMorph already does.

## Backward compatibility

Normal GUI launch with no `--thumbnail-bridge-json` argument must behave exactly as before.

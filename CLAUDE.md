# CLAUDE.md — CoverMorph / Thumbnail AI Bridge

## Working branch
Work only on `thumbnail-bridge-dev`.
Do not modify, merge, or push to `main`.

## Read first
1. `THUMBNAIL_BRIDGE_CONTRACT.md`
2. `CLAUDE_THUMBNAIL_AI_BRIDGE_INSTRUCTIONS.md`
3. Existing generation/edit/project modules before changing architecture.

## Existing architecture to reuse
- `covermorph/generation.py`
  - `detect_generation_environment()`
  - `SDXLTextToImageEngine`
  - existing SDXL/IP-Adapter validation and generation metrics
- `covermorph/project.py`
  - project persistence and scene/person/reference structures
- `covermorph/processor.py`
  - OCR/text cleanup
  - face detection
  - smart crop / text-safe landscape composition
  - full-frame outpaint canvas helpers
- `covermorph/pipeline.py`
  - existing non-destructive processing pipeline
- existing GUI behavior and project compatibility

Do not build a second generation stack if the existing stack can do the work.

## Hardware rule
The user recently upgraded to an NVIDIA RTX 3060.
Do not assume a specific VRAM amount. Detect CUDA GPU name and VRAM at runtime using the existing environment detector and report the exact detected values.

## Bridge rule
Normal double-click launch must behave as the existing CoverMorph GUI.
Only `--thumbnail-bridge-json` enters headless JSON bridge mode.

In bridge mode:
- stdin: exactly one UTF-8 JSON request
- stdout: exactly one UTF-8 JSON response
- human logs/progress: stderr or normal log file
- no messageboxes
- nonzero exit code when `ok=false`

## Safety / data integrity
- never destroy the previous valid bridge output on failed generate/edit
- use staging + atomic replace
- no model weights/build/dist/logs/generated images in git
- no proprietary fonts in git
- preserve Korean/Japanese/space-containing Windows paths

## Validation
Separate these clearly:
- unit/mock PASS
- packaged EXE bridge PASS
- real CUDA/SDXL generate PASS
- real edit PASS
- real youtubesum handshake PASS

Never label a mock call as real AI generation.

## Final handoff
Report:
- commit SHA
- CoverMorph EXE path + SHA-256
- detected GPU and VRAM
- model readiness
- full PASS/FAIL table
- actual generation time and peak CUDA allocated/reserved memory for any real AI run
- actual youtubesum v0.6 handshake result
- remaining limitations

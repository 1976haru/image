# CLAUDE.md — CoverMorph Person Quality Upgrade

## Working branch
Work only on `person-quality-dev`.
Do not modify or merge `main`.
Base branch: `thumbnail-bridge-dev`.

## Problem statement
The AI bridge is technically working on the user's RTX 3060, but the generated human subject quality is unacceptable in real thumbnails.

This milestone is not about bridge plumbing. It is about visible image quality, especially:
- face realism
- skin texture
- eyes
- hair
- hands
- anatomy
- subject sharpness
- cinematic photographic quality
- preserving attractive human faces at thumbnail scale

The current SDXL base 1.0 output is not sufficient as the default final human-image generator.

## Read first
- `CLAUDE_PERSON_QUALITY_INSTRUCTIONS.md`
- existing `CLAUDE.md` history if useful
- `covermorph/generation.py`
- current bridge runtime
- current SDXL/IP-Adapter implementation

## Rules
- Reuse the current bridge and project contracts.
- Do not break youtubesum integration.
- Do not claim quality PASS from unit tests alone.
- Human quality must be validated visually on actual generated outputs.
- Keep model weights outside git.
- Detect GPU/VRAM at runtime.
- Preserve the current balanced RTX 3060 profile unless measurements justify a change.

## Handoff
Report:
- exact model/checkpoint used
- license/model-card source
- GPU/VRAM
- generation settings
- face-detail settings
- before/after crops
- generation time
- peak CUDA allocated/reserved
- visual score table
- known artifacts
- final commit / EXE / SHA-256

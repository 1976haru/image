# Codex brief — CoverMorph Thumbnail Bridge

Repository: 1976haru/image
Branch: thumbnail-bridge-dev
Do not modify main.

Two starter files already exist:
- THUMBNAIL_BRIDGE_CONTRACT.md
- covermorph/thumbnail_bridge.py

Read them first. Do not change the contract casually; if a change is necessary, keep the youtubesum v0.5.1 caller backward-compatible.

## Goal

Make CoverMorph Studio a real subprocess bridge target for YouTube Dynamic Thumbnail Studio.

Normal GUI behavior must remain intact.

## P0 — CLI dispatcher

Add bridge mode to app.py:

- no bridge args -> existing GUI exactly as today
- `--thumbnail-bridge-json` -> read one UTF-8 JSON request from stdin, perform bridge action, write one JSON response to stdout, then exit
- stdout is JSON-only in bridge mode
- logs and tracebacks go to stderr / existing log files

Support:
- generate
- edit
- status

Return non-zero exit code when response ok=false.

## P1 — project mapping

Map bridge request fields into existing CoverMorph project/generation concepts.

For generate:
- project_dir is the bridge output folder
- create or reuse a CoverMorph project safely
- purpose should align with thumbnail_background
- force textless result
- default output ratio 16:9
- use request prompt/story/channel metadata
- do not generate typography inside the image
- existing CoverMorph generation preset/channel logic should be reused, not duplicated

For edit:
- start from current canvas_clean.png or project-selected source
- preserve people/core subject when requested
- use existing edit/outpaint/composition capabilities where possible
- never overwrite the previous valid bridge output until the new result is complete

For status:
- report whether generation/edit dependencies are ready
- report useful warnings instead of opening dialogs

## P2 — bridge sidecars

After successful generate/edit create:
- canvas_clean.png
- preview_reference.png when useful
- subject_boxes.json
- safe_zones.json
- palette.json
- composition.json
- project_manifest.json

Use normalized 0..1 coordinates.

Do not infer gender from pixels. If project intent identifies protagonist/counterpart, preserve those roles.

Use existing CoverMorph knowledge where possible:
- person masks / segmentation
- text safe zone
- original person protection
- palette/image analysis
- project/channel composition hints

Implement missing lightweight analysis locally if needed.

## P3 — clean canvas requirements

- preferred final bridge canvas: 1280x720 PNG
- no generated title/logo/episode/watermark
- preserve protagonist face/person
- create usable negative space for text based on request options/channel
- Tokyo Chill relationship scenes should preserve story context
- OLD POP LOUNGE should prioritize calm readability and text space

## P4 — atomic output

Write into a temporary staging directory or temporary file names.
Only replace current bridge assets after the full action succeeds.
On failure:
- return ok=false
- preserve previous valid canvas/sidecars
- do not leave partially replaced outputs

## P5 — headless behavior

Bridge mode cannot show Tk messageboxes or require a GUI.
Any missing model / invalid project / generation failure must become a structured response.

## P6 — tests

Add automated tests for:
- request parser
- malformed JSON
- unsupported protocol/action
- Korean/Japanese/space path
- status action
- mocked generate action
- mocked edit action
- atomic rollback
- sidecar schemas
- normalized coordinates
- stdout contains JSON only
- GUI entry remains unchanged without bridge args

Do not require actual SDXL downloads for the normal unit suite.

Optional real AI integration test may be marked optional_ai.

## P7 — executable

Update BUILD_EXE.bat/spec so the packaged CoverMorph executable supports the same bridge CLI.

Verify:
1. GUI double-click still launches.
2. EXE --thumbnail-bridge-json status works.
3. EXE bridge call through a mock/lightweight path works on a Korean/Japanese project path.

## P8 — real youtubesum handshake

Once image-side EXE is built:
- configure youtubesum v0.5.1 IMAGE_PROGRAM_EXE to the new CoverMorph EXE
- run status
- run at least one bridge call without model download if a deterministic/mock fixture mode is available
- if the actual SDXL environment is installed and ready, perform one real generate
- verify youtubesum auto-refresh reads the new sidecars

Do not fake a real generation PASS if the model is not installed.

## Git discipline

- work only on thumbnail-bridge-dev
- do not push to main
- no model weights/generated media/build/dist/logs in git
- logical commits
- update README
- final report:
  - commit SHA
  - EXE path/SHA-256
  - PASS/FAIL table
  - real youtubesum handshake result
  - actual AI generate/edit result separately from mocked tests
  - remaining limitations

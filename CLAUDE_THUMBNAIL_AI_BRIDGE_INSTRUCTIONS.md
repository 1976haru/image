# Claude Code implementation brief — CoverMorph real AI Thumbnail Bridge

Repository: `1976haru/image`
Branch: `thumbnail-bridge-dev`
Do not touch `main`.

The paired thumbnail editor is `1976haru/youtubesum`, current accepted branch `v0.6-pro-editor-dev`.

The purpose of this milestone is to turn the existing starter bridge into a **real CoverMorph backend** for the Pro Editor.

---

# 1. Product workflow to complete

The final user flow should be:

```text
YouTube Dynamic Thumbnail Studio v0.6
        ↓ [배경 생성]
CoverMorph headless bridge
        ↓ RTX/CUDA image generation
canvas_clean.png + sidecars
        ↓
Pro Editor auto-refresh
        ↓
Typography/layers remain editable

Then:

Pro Editor [배경 편집]
        ↓ edit instruction
CoverMorph headless bridge
        ↓ regenerated/edited background
        ↓
Pro Editor updates background only
        ↓
existing text/badge/shape layers preserved
```

The user should not need to manually copy image files between programs.

---

# 2. Existing starter contract

Read and preserve:
- `THUMBNAIL_BRIDGE_CONTRACT.md`
- `covermorph/thumbnail_bridge.py`

Required actions:
- `status`
- `generate`
- `edit`

Protocol version remains 1 unless absolutely necessary.

Do not break the existing youtubesum v0.5.1/v0.6 caller contract.

---

# 3. Headless CLI dispatcher

Modify `app.py`:

Normal:
```text
CoverMorphStudio.exe
```
=> existing GUI behavior unchanged.

Bridge:
```text
CoverMorphStudio.exe --thumbnail-bridge-json
```
=> no Tk GUI, no messagebox.

Bridge mode:
1. read one UTF-8 JSON request from stdin
2. validate
3. perform action
4. emit exactly one JSON response on stdout
5. write logs/progress to stderr or existing logs
6. exit 0 on ok=true, nonzero on ok=false

Add tests proving stdout is JSON-only.

---

# 4. Real status action

Use existing `detect_generation_environment()`.

Return structured status information including:
- torch version
- CUDA available
- detected GPU name
- total VRAM bytes
- diffusers/transformers/accelerate/safetensors readiness
- SDXL model path and strict readiness
- IP-Adapter readiness
- edit/outpaint backend readiness
- project_dir writability

Do not trigger downloads during status.

The user's GPU is expected to be RTX 3060, but detect and report rather than assume.

---

# 5. Real generate action

Reuse existing `SDXLTextToImageEngine` and project generation structures.

Input intent:
- channel
- story_type
- episode/title/subtitle metadata
- prompt
- typography-safe options
- preferred text space
- protagonist/counterpart intent when available

Generation requirements:
- textless result only
- 16:9
- preserve story context
- no generated title, logo, watermark, episode marker, subtitle, readable signage
- use negative prompt to strongly suppress text/typography/watermark
- create useful text negative-space when requested
- protagonist/counterpart roles come from project/request intent, not gender inference

Preferred bridge result:
- render/generate at the most appropriate existing SDXL 16:9 size
- produce final `canvas_clean.png` at 1280x720 with high-quality Lanczos reduction/reframing
- do not stretch

Channel behavior:
### Tokyo Chill
- cinematic urban relationship framing
- preserve protagonist/counterpart relationship
- text space must not destroy faces or gaze line

### OLD POP LOUNGE
- calmer composition
- high readability
- generous low-detail text area
- mature photographic mood

Do not create typography; youtubesum owns typography.

---

# 6. RTX 3060 / VRAM-adaptive execution

The current engine already records GPU and VRAM. Extend execution carefully.

Do not assume desktop 12GB or laptop 6GB.

Create runtime profile based on detected VRAM.

Suggested policy, to be validated in code:
- >= 11 GB: normal CUDA fp16 pipeline; current 16:9 SDXL path first
- 8–11 GB: enable memory-safe features supported by current diffusers pipeline
- < 8 GB: conservative mode; reduce generation working size and use high-quality final resize, or return a structured limitation if quality cannot be guaranteed

Memory-safe options may include only when supported and tested:
- attention slicing
- VAE slicing
- VAE tiling
- model CPU offload / sequential CPU offload

Do not simultaneously enable incompatible modes blindly.

On `torch.cuda.OutOfMemoryError`:
1. clear failed pipeline safely
2. record OOM
3. perform at most one deterministic lower-memory retry
4. report that retry occurred
5. never loop indefinitely

Record:
- generation time
- peak allocated bytes
- peak reserved bytes
- retry profile
- working generation resolution

---

# 7. Real edit action

Goal: background edits requested from Pro Editor while preserving typography layers outside CoverMorph.

Typical instructions:
- "인물을 오른쪽으로 조금 이동하고 왼쪽 문구 공간을 넓혀줘"
- "배경만 조금 어둡게 해줘"
- "두 사람은 그대로 두고 도쿄 야경으로 바꿔줘"
- "왼쪽 40%를 문구 공간으로 단순하게 만들어줘"

Use existing CoverMorph functionality where applicable:
- outpaint canvas helpers
- original/person protection
- IP-Adapter person/style reference support
- OCR/text removal for an existing source if needed
- composition geometry

Do not claim arbitrary instruction-based editing is supported unless there is a real backend for it.

Implement edit capability levels explicitly:
- `recompose`: geometry/text-space/background extension supported locally
- `regenerate`: SDXL new background based on prompt/instruction
- `reference_regenerate`: IP-Adapter-assisted preservation when reference backend is ready

Return capabilities in `status`.

If a requested edit exceeds available capability, return structured `UNSUPPORTED_EDIT` rather than silently doing something unrelated.

---

# 8. Sidecar generation

After successful generate/edit write:

- `canvas_clean.png`
- `preview_reference.png` when useful
- `subject_boxes.json`
- `safe_zones.json`
- `palette.json`
- `composition.json`
- `project_manifest.json`

## Subject boxes

Prefer reliable sources in this order:
1. known project/person/reference composition data
2. segmentation/person mask geometry
3. detected face/person boxes
4. conservative fallback

Coordinates normalized 0..1.

Do not infer male/female from pixels.
Use roles:
- protagonist
- counterpart
- other

## Safe zones

Calculate from:
- subject/face avoid areas
- local edge density / busyness
- luminance variance
- requested preferred side

Output several ranked preferred text regions when possible, not just one.

## Palette

Extract:
- dominant colors
- useful accents
- recommended light text
- recommended dark text
- recommended stroke
- recommended shadow

Do not need to duplicate youtubesum's Background Fit algorithm exactly; emit useful source palette data.

## Composition

Include:
- protagonist side
- counterpart side
- recommended text side
- scene type
- notes
- source generation dimensions
- final dimensions

---

# 9. Atomic output

This is mandatory.

For generate/edit:
1. create staging directory inside project_dir
2. write all new assets there
3. validate all required outputs
4. fsync/close files where practical
5. replace final files atomically
6. remove staging

If failure occurs:
- preserve previous valid bridge outputs
- return ok=false
- clean staging
- no partial canvas/sidecars

Add rollback tests.

---

# 10. Project interoperability

Bridge project folder is not required to replace CoverMorph's own project schema.

But when useful:
- create/reuse a CoverMorph project internally
- store a link/reference in `project_manifest.json`
- keep CoverMorph project state separate from the portable youtubesum sidecars

Do not make youtubesum understand internal CoverMorph schema.

---

# 11. Real youtubesum v0.6 handshake

After building CoverMorph:

1. Locate packaged CoverMorph EXE.
2. Set/configure youtubesum:
   - `IMAGE_PROGRAM_EXE=<CoverMorph EXE>`
   - appropriate bridge mode
3. Run `status` from youtubesum/bridge layer.
4. Run generate if real model ready.
5. Confirm:
   - subprocess returns success
   - youtubesum auto-refresh sees new `canvas_clean.png`
   - sidecars load
   - Pro Editor background updates
   - existing text/badge layers remain unchanged
6. Run one edit if real edit capability is available.

Use Korean/Japanese/space-containing project path for at least one handshake.

---

# 12. Real AI acceptance test

If CUDA + SDXL model are ready, perform at least one actual generation.

Use a small, controlled Tokyo Chill case:
- 16:9
- textless
- cinematic Japanese urban transit/cafe/river scene
- one or two people
- requested text space on one side

Record:
- exact GPU name
- total VRAM
- generation working size
- final size
- steps
- guidance
- seed
- time
- peak CUDA allocated/reserved
- whether memory fallback/retry happened

Then visually inspect:
- no readable accidental text
- face/person not broken
- text area actually usable
- correct 16:9 composition
- no obvious stretch

If model is not installed/ready, report REAL AI = NOT RUN.
Do not download multi-GB weights without explicit user action if current program policy requires a prepare/download step.

---

# 13. Image edit acceptance test

If a genuine backend exists, run one real edit:
- preserve people
- create more left-side text space

Compare before/after:
- people preserved reasonably
- usable text area increased
- no typography baked in

If no genuine semantic edit backend exists, report the supported recompose/regenerate level accurately.

---

# 14. Packaging

Update build so packaged EXE supports:
- normal GUI double click
- `--thumbnail-bridge-json`

Optional AI model weights remain external unless redistribution permits and size is appropriate.

Verify:
- GUI starts
- bridge status from packaged EXE
- Unicode path request
- real generate if model ready
- no stdout contamination in JSON bridge mode

---

# 15. Tests

Keep existing tests passing.

Add tests for:
- CLI mode dispatch
- stdout JSON-only
- status structure
- Unicode path
- generate mock
- edit capability routing
- safe-zone schema
- palette schema
- normalized coordinates
- atomic commit
- rollback on failure
- prior valid output preservation
- OOM single retry
- no infinite retry
- packaged bridge invocation
- normal GUI entry unchanged

Optional tests:
- `optional_ai` real CUDA generation
- `optional_ai` real IP-Adapter generation

Do not make CI/unit suite download models.

---

# 16. Git discipline

- work only on `thumbnail-bridge-dev`
- do not touch main
- do not commit build/dist/models/generated media/logs
- small logical commits
- update README
- push branch only after validation

Final report must separate:

## A. Unit/mock
PASS/FAIL

## B. Packaged bridge
PASS/FAIL

## C. Real hardware
- GPU detected
- VRAM detected
- CUDA PASS/FAIL

## D. Real AI
- SDXL model ready?
- generate actually run?
- edit actually run?
- youtubesum real handshake?

Report final commit, EXE path, SHA-256, timings, peak memory and remaining limitations.

# CLAUDE — v1.0 Production Hardening

Repository: 1976haru/image
Branch: v1.0-production-hardening
Base: shopify-workflow-dev
Do not modify or merge main until this milestone passes.

## Goal

Turn the current working system into a v1.0 release candidate for daily use.

Do NOT add unrelated features.
Priority:
1. durable settings
2. original-product preservation path
3. first-run usability
4. error recovery
5. real-use validation
6. release packaging
7. merge decision

Preserve:
- Z-Image-Turbo Q6_K default T2I
- FLUX.2-klein-4B Q8_0 default reference/edit
- RealVisXL fallback
- YouTube + Shopify workflows
- queue / pause-after-current / resume
- low-memory policies
- bridge compatibility
- Unicode paths
- existing quality

## Mandatory progress report

At every meaningful checkpoint:

전체 진행률: NN%
현재 단계: ...
완료: ...
진행 중: ...
남은 것: ...
현재 리스크/막힘: ...

현재 단계 성공 가능성: NN%
전체 프로젝트 성공 가능성: NN%

---

# Phase 1 — persistent settings

Current problem:
EXE rebuild resets settings.

Move user settings out of the EXE/dist folder.

Preferred Windows storage:
- %LOCALAPPDATA%\CoverMorphStudio\settings.json
or equivalent stable per-user location.

Persist at minimum:
- models folder
- backend paths
- quality mode
- memory mode
- last export folder
- last project folder
- queue preferences
- Shopify preset/custom sizes
- reference defaults where appropriate

Requirements:
- atomic writes
- corrupt settings recovery
- schema version
- migration from old dist-local settings if found
- never delete user settings on rebuild/update

Add tests:
- rebuild simulation
- corrupt settings file
- Unicode path
- migration

---

# Phase 2 — first-run setup

On first launch, if required paths are missing:
show a simple setup wizard.

Step 1:
- models folder
- validate required engines/models

Step 2:
- choose default usage:
  - YouTube
  - Shopify
  - Both

Step 3:
- default quality:
  - 빠른 미리보기
  - 일반
  - 최고 품질

Step 4:
- default PC usage:
  - 작업 중 PC 우선
  - 균형
  - 자리 비움

Do not expose raw backend jargon unless user opens advanced settings.

Show status:
- Z-Image ready
- FLUX.2 ready
- RealVis fallback ready/not ready
- GPU / VRAM
- disk availability

---

# Phase 3 — original product preservation workflow

Current limitation:
FLUX.2 can redraw product geometry.

Implement a stronger "원본 상품 보존" path.

Preferred strategy:
1. keep original product pixels
2. segment/mask product
3. generate background separately
4. composite original product over generated background
5. generate contact shadow / grounding if practical
6. color-match background lightly without recoloring product
7. never redraw logo/text/product geometry in this mode

Use existing segmentation tooling where reliable.
If segmentation quality is uncertain:
- show mask preview
- let user adjust/refine or fall back to full reference-edit mode

Modes:
- 원본 그대로 합성 (safest)
- 자연광 보정 합성
- AI 재구성 (least strict, explicit warning)

For the strict mode:
- preserve product pixels exactly except optional edge feathering/shadow
- no generative modification of product interior

Validation:
- mug
- bottle
- boxed product
- irregular product shape

---

# Phase 4 — product mask workflow

Add:
- automatic product mask
- mask preview
- invert
- feather slider
- expand/contract
- simple brush add/remove if practical

Do not block v1.0 on a full Photoshop-class mask editor.
A simple correction tool is enough.

Persist corrected mask in project.

---

# Phase 5 — error handling

Create actionable errors for:
- missing models
- invalid model folder
- insufficient disk space
- low RAM/commit
- low VRAM
- backend crash
- translation failure
- corrupt project
- corrupt queue
- output write failure
- invalid reference image

Rules:
- no raw Python traceback in normal UI
- include "자세히 보기" for technical details
- recover where possible
- preserve queue/project state after failure

---

# Phase 6 — release diagnostics

Add "진단 정보 복사" button.

Include:
- app version
- commit/build id
- Windows version
- GPU
- VRAM
- memory mode
- model readiness
- model paths (sanitize user-sensitive path details if exporting logs)
- last error code
- last backend result
- queue state summary

Do not include image contents or prompts unless user explicitly opts in.

---

# Phase 7 — real-use validation

Use real workflows, not synthetic-only tests.

Minimum 20 jobs:

YouTube:
- Tokyo Chill 6
  - woman
  - man
  - couple
  - rainy night
  - cafe
  - train platform
- OLD POP 4
  - mature woman
  - mature man
  - mature couple
  - seasonal

Shopify:
- Hero 2
- Collection 2
- Product Lifestyle 4
  - strict original-preserve
  - natural-light composite
  - reference-edit comparison
- Promo 1
- Mobile 1

For each:
- output path
- engine/mode
- runtime
- peak GPU
- commit delta
- warnings
- visual verdict
- whether user would plausibly ship it

Do not declare pass if outputs are merely technically valid.

---

# Phase 8 — queue endurance

Run:
- 10 mixed jobs
- pause-after-current
- app close
- app reopen
- resume
- one forced backend failure
- one low-memory wait
- one Unicode project path

Confirm:
- no lost jobs
- completed remain completed
- running becomes safe resumable/pending state
- no zombie backend process
- memory returns close to baseline

---

# Phase 9 — packaging

Build final RC EXE/ZIP.

Requirements:
- double-click launch
- no shell env vars
- settings survive EXE rebuild
- models remain external
- no accidental user data in package
- no screenshots/logs containing unrelated desktop content
- one-folder package allowed if needed
- include README/quick-start

Produce:
- EXE path
- ZIP path
- SHA-256 for both
- version string
- build commit

Suggested version:
v1.0.0-rc1

---

# Phase 10 — merge decision

Do NOT merge automatically.

Final report must recommend one of:
- READY FOR MAIN
- READY WITH MINOR LIMITATIONS
- NOT READY

Criteria for READY:
- settings persistence fixed
- no major quality regression
- strict product-preserve works
- queue endurance passes
- packaged EXE passes
- 20-job real-use validation mostly ship-ready
- no critical data-loss bug

If not ready, list exact blockers.

---

# Final acceptance

v1.0 RC passes only if a normal user can:
1. install/unzip
2. launch by double click
3. point to model folder once
4. create YouTube or Shopify image
5. attach references
6. queue multiple jobs
7. pause/resume later
8. export final images
without PowerShell or manual environment variables.

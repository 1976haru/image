# CLAUDE — Shopify Creator + Workflow UX milestone

Repository: 1976haru/image
Branch: shopify-workflow-dev
Base: person-quality-dev
Do not modify or merge main.

## Goal

Turn the now-proven quality engine into a practical daily-use image workstation for:
1. YouTube thumbnails
2. Shopify store imagery
3. queued low-memory generation while the user is away

The priority order is:
1. output quality
2. reference fidelity
3. easy daily workflow
4. low memory pressure
5. Shopify presets
6. packaging/polish

Do not regress:
- Z-Image-Turbo Q6_K default T2I
- FLUX.2-klein-4B Q8_0 default reference/edit
- RealVisXL fallback
- PREVIEW/BALANCED/BEST
- queue persistence
- pause-after-current/resume
- one-model-at-a-time policy
- bridge compatibility
- Unicode Windows paths

## Progress reporting — mandatory

At every meaningful checkpoint report exactly:

전체 진행률: NN%
현재 단계: ...
완료: ...
진행 중: ...
남은 것: ...
현재 리스크/막힘: ...

현재 단계 성공 가능성: NN%
전체 프로젝트 성공 가능성: NN%

Percentages are estimates, not test results.

---

# Phase 1 — Real-use workflow shell

Create a clear top-level workflow selector:

- YouTube Thumbnail
- Shopify Hero Banner
- Shopify Collection Banner
- Shopify Product Lifestyle
- Shopify Promo Tile
- Shopify Mobile Banner
- Custom Canvas

User should not have to know model names.

Expose quality choices as:
- 빠른 미리보기
- 일반
- 최고 품질

Map internally to PREVIEW / BALANCED / BEST.

Expose memory behavior:
- 작업 중 PC 우선
- 균형
- 자리 비움/최고 품질

Default should favor low memory while user is actively using the computer.

---

# Phase 2 — Shopify Creator presets

Add configurable preset definitions, not hardcoded one-offs.

Minimum built-in presets:

## Hero Banner
Default starting canvas: 1800x700
Support custom dimensions.

## Collection Banner
Default starting canvas: 1600x900
Support square alternate.

## Product Lifestyle
Default starting canvas: 1600x1200

## Promo Tile
Default starting canvas: 1080x1080

## Mobile Banner
Default starting canvas: 1080x1350

Do not assume every Shopify theme uses one fixed size.
Always allow custom width/height.

Each preset needs:
- canvas size
- safe text region
- subject region
- optional CTA region
- export JPG/PNG
- textless background export
- optional composed export if text layers exist

---

# Phase 3 — Reference roles

UI must allow reference files with explicit roles:

- 인물
- 상품
- 스타일
- 구도
- 배경

Store roles in the project.

Never silently reinterpret the role.

For FLUX.2 reference/edit prompts, compile clear role-specific instructions.

Examples:
- product: preserve exact shape, material, color, packaging, visible branding
- person: preserve face, hairstyle, clothing, pose unless explicitly changed
- composition: follow placement/negative space, not subject identity
- style: transfer lighting/color/mood only
- background: use environmental inspiration without copying foreground subject

Allow multiple reference images up to backend limits.

---

# Phase 4 — Product preservation mode

Shopify product work needs stronger constraints than artistic generation.

Add a dedicated mode:
- "상품 형태 보존"

When enabled:
- FLUX.2 reference/edit is preferred
- avoid regenerating the product from scratch if an edit/composite path can preserve it
- keep product silhouette, logo placement, color, package geometry
- background/environment may change

If the backend visibly distorts the product:
- flag warning
- do not silently rank it as best
- retain original and other candidates

Add technical checks where practical:
- perceptual similarity on the product crop
- edge/silhouette difference
- color drift
- OCR/logo text mismatch warning when text is legible

Do not claim exact preservation when the model cannot guarantee it.

---

# Phase 5 — Prompt workspace

Create a practical prompt panel with:

- 사용자 프롬프트
- 채널/스토어 프리셋
- 레퍼런스 역할
- 생성 목적
- 품질 모드
- seed
- 후보 수

The app may compile/translate prompts internally, but show:
- original prompt
- compiled engine prompt
- engine selected
- translation applied yes/no

Add reusable prompt presets for:
- Tokyo Chill solo woman
- Tokyo Chill solo man
- Tokyo Chill couple
- OLD POP mature couple
- OLD POP mature solo
- Shopify clean studio
- Shopify editorial lifestyle
- Shopify soft daylight lifestyle
- Shopify luxury product
- Shopify seasonal promotion

Do not insert demographic/nationality assumptions unless preset explicitly defines them.

---

# Phase 6 — Channel/store locale hints

Add optional preset metadata:
- locale
- language
- audience
- appearance_hint

Use it only when explicitly configured.

For the user's Japanese channel preset, allow an explicit Japanese-person appearance hint.
Do not infer ethnicity from image pixels.

For OLD POP, do not automatically force Asian/Western appearance unless preset says so.

---

# Phase 7 — Queue UI

Build a user-facing queue panel:

Columns:
- status
- purpose
- prompt/title
- engine
- quality mode
- progress
- elapsed
- output folder

Controls:
- 시작
- 현재 작업 후 일시정지
- 재개
- 선택 취소
- 대기 항목 삭제
- 완료 항목 열기
- 실패 재시도

Persist queue across restart.

The app must never start two diffusion engines simultaneously on this machine.

---

# Phase 8 — Resource-aware scheduling

Before starting each job, inspect:
- GPU free/used VRAM
- Windows available RAM
- commit charge

If resources are below threshold:
- leave job pending
- show "PC 사용 중 — 자원 대기"
- recheck periodically

Do not kill user apps.

Add optional:
- "PC가 여유 있을 때 자동 시작"

Add idle scheduling hooks only if robust on Windows.
If true OS-idle detection is unreliable, implement a manual "자리 비움 모드" rather than pretending.

---

# Phase 9 — unload policy

For "작업 중 PC 우선":
- backend process exits after every image/job
- memory must return close to baseline
- no persistent diffusion pipeline

For "균형":
- may keep lightweight components only if they do not materially affect normal PC use

For "자리 비움/최고 품질":
- sequential ensemble allowed
- still never keep multiple heavy engines resident

Record memory before/after each job.

---

# Phase 10 — candidate review

Add a practical candidate review screen:
- 2–4 candidate thumbnails
- large preview
- 340 px thumbnail preview
- for YouTube: 180 px preview
- face crop if people present
- product crop if product reference present
- technical warnings
- generation engine
- seed
- generation time

Actions:
- 채택
- 다시 생성
- seed만 변경
- 프롬프트 수정
- 편집으로 보내기
- 기존 후보와 비교

Do not hide lower-ranked candidates.

---

# Phase 11 — export

YouTube:
- 1280x720 JPG/PNG

Shopify:
- preset native resolution
- configurable JPG quality
- PNG where transparency is needed
- textless version
- composed version

Sanitize filenames but preserve Unicode where Windows supports it.

---

# Phase 12 — validation

Run real GPU validation on RTX 3060:

YouTube:
1. Tokyo Chill woman
2. Tokyo Chill man
3. Tokyo Chill couple
4. OLD POP mature couple

Shopify:
5. hero banner no product
6. collection lifestyle
7. product-reference lifestyle
8. promo tile
9. mobile banner

Queue:
10. mixed 5-job queue
11. pause-after-current
12. restart app
13. resume
14. verify memory returns close to baseline after each heavy job

Report:
- visual quality notes
- reference fidelity
- timing
- GPU peak
- system commit
- warnings
- output paths

---

# Phase 13 — packaged EXE acceptance

Required:
- all existing tests pass
- new tests pass
- packaged EXE launches by double click
- no manual shell env vars for normal use
- model/backend paths can be configured in UI and persisted
- current bridge use remains compatible
- queue survives restart
- Unicode paths pass

Build EXE and report:
- path
- SHA-256
- final commit
- known limitations

---

# Stop conditions

Ask user before:
- deleting old model files
- changing Windows pagefile
- closing user programs
- downloading additional multi-GB models not already approved by this milestone
- accepting any paid/non-commercial/research-only model license

Do not ask before normal code/test/build work.

---

# Final acceptance standard

This milestone is PASS only if:
- the user can open the EXE and create either a YouTube or Shopify image without PowerShell
- the user can attach references with explicit roles
- product-reference mode visibly preserves products better than unconstrained generation
- queue/pause/resume works after restart
- low-memory mode returns resources after jobs
- no regression in current Quality Engine V2 output quality

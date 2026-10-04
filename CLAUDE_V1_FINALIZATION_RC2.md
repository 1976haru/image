# CLAUDE — v1.0 Finalization / RC2

Repository: 1976haru/image
Branch: v1.0-finalization
Base: v1.0-production-hardening
Do not modify or merge main until final report and user approval.

## Why this milestone exists

The visual review sheet has now been inspected by the user/assistant.

Observed quality:
- Tokyo Chill woman: strong, production-usable
- Tokyo Chill man: strong, production-usable
- Tokyo Chill couple: usable, but faces/emotional presence can be stronger
- OLD POP mature couple: usable, but slightly more stock-like / flatter lighting than desired
- Shopify Hero / Collection: production-usable
- Shopify Strict Composite: strong product preservation
- Shopify Natural-Light Composite: best balance of preservation + scene integration
- Shopify FLUX.2 Rebuild: visually plausible scene, but product label/identity can be lost, so it must NOT be the default product-preservation mode

This milestone is not for adding new models. It is for locking correct defaults, doing targeted final polish, and verifying the complete YouTube -> image engine -> editor -> export workflow.

---

# Mandatory progress report

At every meaningful checkpoint report exactly:

전체 진행률: NN%
현재 단계: ...
완료: ...
진행 중: ...
남은 것: ...
현재 리스크/막힘: ...

현재 단계 성공 가능성: NN%
전체 프로젝트 성공 가능성: NN%

---

# Phase 1 — Lock production defaults

Keep:
- Default T2I: Z-Image-Turbo Q6_K
- Default reference/edit: FLUX.2-klein-4B Q8_0
- Fallback: RealVisXL V5
- PREVIEW / BALANCED / BEST model routing unchanged unless a regression is found

Shopify product defaults:

1. Default product workflow:
   Natural-Light Composite

2. "절대 상품 보존" option:
   Strict Composite

3. FLUX.2 Rebuild:
   rename/label clearly as:
   "AI 재구성 · 상품 형태/라벨 변형 가능"
   This must NOT be selected by default for product-preservation jobs.

4. Product warning:
   when rebuild/reference-edit changes OCR/logo/label or product geometry is uncertain,
   show a visible Korean warning.
   Never hide the original-preserve candidates.

Do not weaken strict-preserve semantics.

---

# Phase 2 — Targeted YouTube quality polish

Do NOT change the global engine defaults.

## Tokyo Chill couple

Improve prompt/compiler preset only.

Goal:
- both faces slightly larger
- stronger relationship/emotional read at 340 px
- still retain meaningful city/environment context
- preserve clean text-safe area
- avoid overly staged wedding/editorial feeling

Preferred framing:
- medium / medium-close two-shot
- both faces readable
- natural interaction
- slight asymmetry rather than centered portrait

Run 4 controlled comparisons:
- current prompt preset x2
- revised prompt preset x2

Keep current if revised is not clearly better.

## OLD POP mature people

Improve preset only.

Goal:
- less generic stock-photo feeling
- warmer lived-in realism
- more natural skin texture
- stronger environmental storytelling
- avoid over-young faces
- avoid exaggerated wrinkles
- avoid sterile beige studio look

Test:
- mature woman
- mature man
- mature couple
- one seasonal scene

Again: keep current if revised is not visibly better.

Do not add nationality/ethnicity unless the channel preset explicitly requests it.

---

# Phase 3 — Composite grounding polish

Strict Composite:
- improve contact shadow / grounding only
- do not modify product interior pixels
- do not alter label/logo/color/material

Natural-Light Composite:
- preserve product pixels
- adjust only:
  - contact shadow
  - local surrounding light
  - restrained edge integration
  - background tone
- do NOT globally tint the product
- expose a simple intensity setting:
  - 약하게
  - 기본
  - 강하게

Default:
- 기본

Validation:
- bottle
- mug
- box
- irregular product

Pixel-preservation check must confirm product interior remains unchanged in strict mode and unchanged except explicitly permitted edge/shadow region in natural-light mode.

---

# Phase 4 — Final UI wording

Normal user should not need model names.

Primary labels:

Quality:
- 빠른 미리보기
- 일반
- 최고 품질

PC usage:
- 작업 중 PC 우선
- 균형
- 자리 비움

Product modes:
- 자연광 보정 합성 (추천)
- 원본 그대로 합성
- AI 재구성 · 변형 가능

Reference roles:
- 인물
- 상품
- 스타일
- 구도
- 배경

Advanced settings may show:
- Z-Image
- FLUX.2
- RealVisXL
- quantization
- backend paths

Keep advanced names out of the normal first screen.

---

# Phase 5 — End-to-end youtubesum integration

This is mandatory before calling the whole project v1.0 ready.

Use the existing main thumbnail app at:
D:\03_youtubesum

Do not modify youtubesum unless the integration genuinely requires a fix.

Test from the normal youtubesum EXE/workflow, not by calling CoverMorph directly:

1. Launch youtubesum by double click
2. Open/create Tokyo Chill project
3. Trigger AI background generation
4. Confirm CoverMorph v1.0 engine is discovered/configured without shell env vars
5. Generate Z-Image background
6. Confirm sidecars load
7. Confirm existing title/badge layers remain
8. Reposition text if needed
9. Export 1280x720
10. Reopen project
11. Confirm all layers/settings persist

Repeat:
- Tokyo Chill person
- Tokyo Chill couple
- OLD POP
- one Shopify image if youtubesum exposes Shopify workflow, otherwise verify via CoverMorph UI only

If youtubesum cannot persist engine/model settings:
- do not hack environment variables
- record exact issue
- fix through a dedicated persisted settings path only if necessary

---

# Phase 6 — Final user-review set

Create:
D:\03_youtubesum\USER_REVIEW_V1_FINAL

Use only real outputs from this milestone.

Include:
01 Tokyo Chill woman
02 Tokyo Chill man
03 Tokyo Chill couple
04 OLD POP mature woman/man/couple representative
05 Shopify Hero
06 Shopify Collection
07 Product Original
08 Strict Composite
09 Natural-Light Composite
10 AI Rebuild

Create:
FINAL_REVIEW_SHEET_RC2.jpg

Requirements:
- >= 4000 px wide or otherwise high enough to inspect faces/products
- full image + crop for people
- full image + crop for products
- engine/mode/seed/time/warning labels
- no desktop screenshots
- no unrelated user content

Do not regenerate purely to make the sheet prettier.

---

# Phase 7 — Regression / endurance

Required:
- all existing tests pass
- queue mixed jobs pass
- pause-after-current pass
- app restart/resume pass
- low-memory wait pass
- no zombie process
- memory release back near baseline
- persistent settings survive rebuild
- Unicode paths
- packaged EXE double-click
- youtubesum bridge end-to-end

Run one forced failure:
- confirm user sees actionable error
- retry succeeds
- no job silently marked done without outputs

---

# Phase 8 — RC2 packaging

Build:
CoverMorphStudio v1.0.0-rc2

Produce:
- EXE path
- ZIP path
- SHA-256 both
- final commit
- quick start
- release notes

Do not bundle model weights in git.
Do not include validation folders or user content in the ZIP unless intentionally documented.

---

# Phase 9 — Final decision

Do NOT merge main automatically.

Return one of:
- READY FOR MAIN
- READY WITH MINOR LIMITATIONS
- NOT READY

READY FOR MAIN requires:
- end-to-end youtubesum flow passes
- no critical quality regression
- product default = Natural-Light Composite
- strict preserve still pixel-safe
- AI Rebuild clearly marked risky
- settings persist
- queue/endurance pass
- RC2 double-click package pass

If READY FOR MAIN, stop and wait for explicit user approval before merge.

---

# Final report format

전체 진행률: 100%
현재 단계: v1.0 RC2 최종 판정
완료: ...
진행 중: 없음
남은 것: ...
현재 리스크/막힘: ...

현재 단계 성공 가능성: 완료
전체 프로젝트 성공 가능성: NN%

Then include:
- visual quality summary
- end-to-end youtubesum result
- product preservation result
- queue/memory result
- tests
- EXE + SHA
- ZIP + SHA
- final commit
- merge recommendation
- minor limitations

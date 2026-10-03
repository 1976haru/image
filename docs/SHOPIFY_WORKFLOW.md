# AI 이미지 스튜디오 — YouTube · Shopify workflow

Branch `shopify-workflow-dev` (base `person-quality-dev`). No new models: the studio drives the Quality Engine V2
defaults (Z-Image-Turbo Q6_K text-to-image, FLUX.2-klein-4B Q8_0 reference/edit, RealVisXL fallback).
Validated 2026-10-04 on the RTX 3060 12 GiB (`scripts/validate_studio.py`, `validation_results/studio/`).

## Using it

Double-click `CoverMorphStudio.exe`: the main window opens and the **AI 이미지 스튜디오** window opens on top
(also from the green top-bar button). No PowerShell or environment variables: model / sd.cpp / output folders are
set in the **설정** tab and saved to `config/creator_settings.json` (the youtubesum bridge reads the same file).

| Tab | What it does |
|---|---|
| 만들기 | YouTube / Shopify (히어로·컬렉션·상품 라이프스타일·프로모션·모바일) / 사용자 지정, editable canvas size, layout preview (text / subject / CTA regions), prompt presets, 채널/스토어, 인물 수, prompt (Korean/Japanese OK), 제목·부제·버튼 text layers, up to 4 references with roles, 상품 형태 보존 + 상품 크기, 품질, PC 사용, seed, 후보 수 |
| 대기열 | 상태·목적·프롬프트/제목·엔진·품질·진행률·경과·출력 폴더; 시작, 현재 작업 후 일시정지, 재개, 선택 취소, 대기 항목 삭제, 완료 항목 열기, 실패 재시도; live GPU / RAM / commit line |
| 후보 비교 | all candidates (never hidden), large / 340 px / 180 px (YouTube) previews, face crop, product crop, warnings, engine, seed, time, license, original / translated / engine prompt; 채택 (export), 다시 생성, seed만 변경, 프롬프트 수정, 편집으로 보내기, 기존 후보와 비교 (A/B) |
| 설정 | folders, JPG quality, 자원 여유 시 자동 시작, 자리 비움 idle minutes, recheck interval, engine status check |

Model names are never a choice. UI labels map to engine settings:

| UI | Internal |
|---|---|
| 빠른 미리보기 / 일반 / 최고 품질 | PREVIEW / BALANCED / BEST |
| 작업 중 PC 우선 (default) / 균형 / 자리 비움/최고 품질 | interactive_low_memory / balanced_idle / night_best |
| 인물 / 상품 / 스타일 / 구도 / 배경 | PERSON / PRODUCT / STYLE / COMPOSITION / BACKGROUND |

## Presets (`covermorph/creator_presets.py`, override/add in `config/creator_presets.json`)

| Purpose | Default canvas | Alternates | Text region | CTA |
|---|---|---|---|---|
| YouTube 썸네일 | 1280×720 | | left 44% | — |
| Shopify 히어로 배너 | 1800×700 | custom | left 40% | yes |
| Shopify 컬렉션 배너 | 1600×900 | 정사각형 1200×1200 | left 42% | — |
| Shopify 상품 라이프스타일 | 1600×1200 | | top 20% | — |
| Shopify 프로모션 타일 | 1080×1080 | | top 26% | yes |
| Shopify 모바일 배너 | 1080×1350 | | top 24% | yes |
| 사용자 지정 | any 256–4096 | | | |

Engines generate within a pixel budget per memory mode (PC 우선 ≤0.98 MP, 균형 ≤1.15 MP, 자리 비움 ≤1.4 MP,
measured peaks 7.3 / 10.4 GiB) and Lanczos-scale to the native canvas.

Prompt presets: Tokyo Chill solo woman / solo man / couple (explicit `appearance_hint: Japanese`), OLD POP mature
couple / mature solo (no appearance hint), Shopify clean studio / editorial / soft daylight / luxury / seasonal.
Appearance words are only added when a preset defines them; nothing is inferred from pixels.

## Reference roles (FLUX.2 wording)

product: shape, material, color, packaging, visible branding · person: face, hairstyle, clothing, pose unless the
prompt changes them · style: lighting, color, mood only · composition: placement and negative space, not identity ·
background: environment inspiration without copying its foreground subjects. Roles are stored with copies of the
reference files in the job folder (`refs/`, `job.json`).

## 상품 형태 보존 (product preservation)

1. The product is cut out of the reference (alpha PNG, or GrabCut on a plain backdrop: logos/prints kept,
   see-through holes such as a mug handle kept open).
2. **Exact composite**: Z-Image generates the setting (neutral wording, see below) and the reference's own product
   pixels are placed in the subject region with a contact shadow. Shape/logo are exact by construction; lighting is
   only approximated. These candidates rank first.
3. 일반/최고 품질 add **FLUX.2 lighting correction** of the best composite; 최고 품질 also adds **FLUX.2 product
   reference generations**. These redraw the product, so they always carry "AI가 다시 그린 상품 — 형태·부품 수·로고를
   참조와 직접 비교하세요", get a lower score and are never ranked above a composite.
4. Automatic checks on redrawn products: color drift (ΔE of the product-colored region) and logo/text OCR
   (EasyOCR in a short-lived CPU process, only if its weights are installed). A silhouette / hole-count check was
   tried on 8 earlier FLUX.2 mugs and could not tell one handle from two, so it is not used.

## Prompt lessons from this run (all fixed and measured)

| Problem seen | Cause | Fix |
|---|---|---|
| A camera appeared in Shopify hero / collection / mobile scenes | "Shot on a full-frame camera with a 50mm lens" in scenes without people | camera wording only for people scenes |
| Bottles/cameras kept appearing | the "no camera, bottle, …" sentence (Z-Image draws listed nouns) | positive wording: "surfaces are bare and uncluttered" |
| Product pasted on a blank white board / a bottle behind the mug | "product photography background plate", "nothing stands on the surface", "lifestyle product photograph" envelope | neutral `background_scene` envelope + "a wide, clear stretch of bare surface": 6/6 bare tables |
| OLD POP "50대 일본인 부부" came out partly Western | the translator dropped 일본인 | nationality/age in the instruction + few-shot + deterministic safety net |
| +4.9 GB commit and a CUDA context left after a product job | EasyOCR in-process; BEST product step also loaded RealVis | OCR worker process; FLUX.2-only product step |

## RTX 3060 validation (final code)

| # | Job | Canvas | Mode | Candidates (engine, s/image, GPU peak) | Job time | GPU after (baseline) | Result |
|---|---|---|---|---|---|---|---|
| 1 | Tokyo Chill woman | 1280×720 | 일반 / PC 우선 | Z-Image ×2, 34.5 s, 6.9 GiB | 72 s | 339 (339) | natural Japanese woman, side profile, title space |
| 2 | Tokyo Chill man | 1280×720 | 일반 / PC 우선 | Z-Image ×2, 34.8 s, 6.9 GiB | 72 s | 339 (339) | platform scene, good faces |
| 3 | Tokyo Chill couple | 1280×720 | 일반 / PC 우선 | Z-Image ×2, 34.8 s, 6.9 GiB | 72 s | 339 (339) | cafe-window couple; QA notes faces >28% of height |
| 4 | OLD POP couple (Korean prompt) | 1280×720 | 일반 / PC 우선 | Z-Image ×2, 34.6 s, 6.9 GiB | 86 s (incl. ~10 s translation) | 338 (338) | Japanese couple in their fifties, autumn bench |
| 5 | Shopify hero, no product | 1800×700 | 일반 / PC 우선 | Z-Image ×2, 35.4 s, 7.0 GiB | 75 s | 338 (338) | airy room, ceramic vases, no invented product |
| 6 | Shopify collection lifestyle | 1600×900 | 일반 / PC 우선 | Z-Image ×2, 35.5 s, 7.0 GiB | 76 s | 338 (338) | bright living room, no product |
| 7 | Shopify product-reference lifestyle | 1600×1200 | 최고 품질 / 자리 비움 | 2 composites (Z-Image 47 s), FLUX.2 lighting fix 47 s, FLUX.2 product ×2 34 s; peak 9.8 GiB | 274 s | 338 (338), commit +5 MB | composites exact; FLUX.2 versions natural with a single handle, flagged for manual check |
| 8 | Shopify promo tile + product | 1080×1080 | 일반 / PC 우선 | 2 composites 35 s, FLUX.2 lighting fix 39 s | 125 s | 338 (338) | mug on a festive table; composed "가을 한정 20% 할인" + "지금 구매" |
| 9 | Shopify mobile banner | 1080×1350 | 일반 / PC 우선 | Z-Image ×2, 35.5 s, 7.2 GiB | 75 s | 338 (338) | autumn window nook; composed Korean title/subtitle/CTA |

Exports: every job writes `export/<name>_textless.jpg|png` at the native canvas, plus `_composed` when title /
subtitle / CTA are set.

Queue (`validation_results/studio/queue 큐/`): 5 mixed jobs (YouTube, hero, product composite, mobile, OLD POP;
빠른 미리보기). Pause during job 1 → job 1 finished, 4 stayed pending (PASS); new queue/runner objects from disk
("restart") kept paused + 4 pending (PASS); 재개 completed all 5 (PASS); GPU after every job 338 MiB = baseline
(worst delta 0 MiB, PASS); commit change per job −70…+216 MB. Each FLUX.2 preview job took ~19 s.

## Limitations

* Composite lighting is approximate (shadow only); the FLUX.2 lighting-corrected candidate looks more natural but
  redraws the product and must be checked by eye.
* The product color check can false-alarm under strong warm light (one regenerated mug: "color region not found").
* OCR compares text only when EasyOCR weights are already installed; it never downloads them.
* The RealVis fallback runs inside the app process; when it is used, a ~150 MiB CUDA context stays until the app
  exits (the default engines run as separate processes and return everything).
* Idle detection uses Windows `GetLastInputInfo` (keyboard/mouse only) and is optional for 자리 비움 mode.
* Wide Shopify canvases in 작업 중 PC 우선 are generated at ≤0.98 MP and upscaled; use 자리 비움 for more detail.

# CoverMorph Studio v1.0.0-rc1 — release validation

Branch `v1.0-production-hardening` (base `shopify-workflow-dev`). No new models: Z-Image-Turbo Q6_K (T2I),
FLUX.2-klein-4B Q8_0 (reference/edit), RealVisXL fallback. Measured 2026-10-04 on the user's PC
(RTX 3060 12 GiB, i9-12900, 32 GB RAM, Windows 10). GPU baseline with nothing running: 338 MiB.

## What changed for v1.0

| Area | Result |
|---|---|
| Durable settings | `%LOCALAPPDATA%\CoverMorphStudio\settings.json` (schema 1, atomic writes, corrupt file kept as `settings.corrupt-*.json` and defaults used, newer-schema keys preserved). Queue and logs live there too; default output `Documents\CoverMorphStudio`. Old dist-local settings/queue migrated once. A rebuild/update never touches them (tested: rebuild simulation, corrupt file, Unicode path, migration). |
| First run | 4-step wizard: model folder (checked: Z-Image / FLUX.2 / RealVis / translator, GPU, free disk) → YouTube / Shopify / both → quality → PC usage. Re-run from 설정. |
| 원본 상품 보존 | 상품 처리 menu: **원본 그대로 합성** (product pixels only scaled; edge feather + contact shadow), **자연광 보정 합성** (background brightness + directional shadow; product pixels unchanged), **AI 재구성** (FLUX.2, explicit warning), 보존 안 함. Pixel equality of the product in both composite modes is a unit test. Products stand on the detected surface edge; position/size can be adjusted afterwards without AI. |
| Product mask | auto mask, invert, grow/shrink, feather, add/remove brush, outline warning; must be checked once per reference before a product job; stored per reference and copied into the job. |
| Errors | codes + plain Korean message + what to do; technical details in the log behind 자세히 보기; preflight (model folder, engines, disk ≥ 2 GB, output writable, readable references); failed jobs never stop the queue; corrupt queue quarantined. |
| Diagnostics | 진단 정보 복사: version/build, Windows, GPU/VRAM, modes, readiness, sanitized paths, last error, last job, queue summary; prompts only on opt-in, never images. |

## Phase 7 — 20 real jobs (`scripts/validate_v1.py`, `validation_results/v1 실작업/`)

All through the studio's queue executor; isolated data folder; each job made 2 candidates (최고 품질 job: 5),
top candidate exported (textless + composed when text was set). GPU after every job: **338 MiB (= baseline)**.

| # | Job | Canvas | Mode / 상품 처리 | Engine | Time | GPU peak | Commit Δ | Verdict | Ship? |
|---|---|---|---|---|---|---|---|---|---|
| 1 | Tokyo Chill woman | 1280×720 | 일반 | Z-Image | 68 s | 6.7 GiB | +1.7 GB* | natural side profile, title space | yes |
| 2 | Tokyo Chill man (river) | 1280×720 | 일반 | Z-Image | 69 s | 6.7 GiB | −1.5 GB* | night river, good face | yes |
| 3 | Tokyo Chill couple (umbrella) | 1280×720 | 일반 | Z-Image | 70 s | 6.7 GiB | +1.8 GB* | both faces good | yes |
| 4 | Tokyo Chill rainy night (Korean, "사람 없음") | 1280×720 | 일반 | Z-Image | 80 s | 6.7 GiB | −2.2 GB* | strong neon reflections; small distant pedestrians despite "no people" | usable |
| 5 | Tokyo Chill cafe (Japanese prompt) | 1280×720 | 일반 | Z-Image | 79 s | 6.7 GiB | 0 | reading woman by window | yes |
| 6 | Tokyo Chill train platform | 1280×720 | 일반 | Z-Image | 68 s | 6.7 GiB | 0 | sunset platform | yes |
| 7 | OLD POP mature woman (records) | 1280×720 | 일반 | Z-Image | 67 s | 6.7 GiB | −1.0 GB* | good face; holds the record awkwardly | usable |
| 8 | OLD POP mature man (Korean, 60대 일본인) | 1280×720 | 일반 | Z-Image | 77 s | 6.7 GiB | 0 | Japanese man in his 60s, snowy window | yes |
| 9 | OLD POP mature couple dancing | 1280×720 | 일반 | Z-Image | 69 s | 6.7 GiB | 0 | warm embrace, plausible hands | yes |
| 10 | OLD POP seasonal (cherry blossoms) | 1280×720 | 일반 | Z-Image | 69 s | 6.7 GiB | 0 | Showa street, no people | yes |
| 11 | Shopify hero, editorial | 1800×700 | 일반 | Z-Image | 73 s | 7.0 GiB | 0 | ceramic vases, text + CTA composed | yes |
| 12 | Shopify hero + bottle | 1800×700 | 원본 그대로 | Z-Image + composite | 73 s | 7.0 GiB | +0.5 GB | bottle on the counter, LUMA label exact | yes |
| 13 | Shopify collection daylight | 1600×900 | 일반 | Z-Image | 71 s | 7.0 GiB | 0 | bright living room | yes |
| 14 | Shopify collection square (Korean) | 1200×1200 | 일반 | Z-Image | 82 s | 6.9 GiB | 0 | autumn wood living room | yes |
| 15 | Product lifestyle — mug, strict | 1600×1200 | 원본 그대로 | Z-Image + composite | 76 s | 7.0 GiB | 0 | mug on the table (after the placement fix) | yes |
| 16 | Product lifestyle — box, natural light | 1600×1200 | 자연광 보정 | Z-Image + composite | 72 s | 7.0 GiB | −0.5 GB | box on the desk, soft directional shadow; product a little brighter than the lamp-lit room | usable |
| 17 | Product lifestyle — plant, corrected mask | 1600×1200 | 원본 그대로 | Z-Image + composite | 72 s | 7.0 GiB | 0 | full pot on the windowsill | yes |
| 18 | Product lifestyle — bottle, reference-edit comparison | 1600×1200 | AI 재구성, 최고 품질 | Z-Image + FLUX.2 | 256 s | 9.8 GiB | 0 | composites keep LUMA; FLUX.2 regenerations look natural but **lost the LUMA label** — flagged automatically ("상품 글자/로고가 다릅니다: 'LUMA' → '없음'", ΔE 41–46) | composites yes, AI no |
| 19 | Promo tile + mug | 1080×1080 | 원본 그대로 | Z-Image + composite | 72 s | 6.9 GiB | 0 | festive table, "겨울 한정 20% 할인" + CTA | yes |
| 20 | Mobile banner | 1080×1350 | 일반 | Z-Image | 70 s | 7.0 GiB | 0 | snowy reading nook, Korean title/subtitle/CTA | yes |

\* Commit Δ is system-wide (other apps were running); per job it swung both ways and never accumulated.

Summary: **16 ship-ready, 3 usable with minor issues, 1 comparison job whose AI candidates were correctly flagged.**
Found and fixed during this run: a product floating mid-air when the table had no visible front edge
(surface search limited to the lower half of the subject area; jobs 12/15/16/17/19 re-run on the final code).

The reference-edit comparison (#18 vs #12/#15) is the reason 원본 그대로 합성 is the default for products:
FLUX.2 redraws the product and can drop logos/labels; the composite keeps them by construction.

## Phase 8 — queue endurance on the packaged EXE (`scripts/validate_endurance.py`)

10 mixed jobs (YouTube, hero, strict product, Korean OLD POP, mobile, couple, collection, natural product,
rainy night, promo), data and output folders under `validation_results/v1 내구성 テスト/` (Korean+Japanese path).

| Run | What happened | Result |
|---|---|---|
| A | start → 현재 작업 후 일시정지 while job 1 ran → app closed | job 1 done, 9 pending, paused saved — PASS |
| B | reopen (same states) → 재개 → app closed while q03 (strict product) was running | q03 back to **pending**, nothing left "running", no sd-cli left — PASS |
| C | a helper process held GPU memory before the app opened → queue showed **"GPU 여유 4004 MiB (필요 6500 MiB)"** and waited; memory released after 20 s → resumed; harness killed sd-cli during q07 → `BACKEND_CRASH|이미지 엔진 프로세스가 정상적으로 끝나지 않았습니다.`, queue continued → 실패 재시도 → done | **10/10 done, no lost jobs** — PASS |
| after each run | sd-cli / llama-completion / CoverMorphStudio processes left: 0; GPU 338 MiB (Δ 0) | PASS |

## Phase 9 — package

`python scripts/build_release.py` → `dist\CoverMorphStudio\` (one folder) + `dist\CoverMorphStudio-v1.0.0-rc1-win64.zip`
with `QUICKSTART_KO.txt`; refuses to package config/outputs/logs/models; build id embedded (진단 정보 복사).
Models stay external (~16 GB). Settings in `%LOCALAPPDATA%` survive rebuilds/updates.
Hashes and the final commit are in `dist\release-v1.0.0-rc1.json` and in the handoff report.

## Limitations (none blocking)

* The automatic product mask can miss product parts that match the backdrop color (white pot on light grey);
  the one-time mask check is required for that reason and the brush fixes it in seconds.
* 자연광 보정 합성 never relights the product, so a studio-lit product can look slightly brighter than a dim scene;
  AI 재구성 relights but may redraw logos/labels (flagged).
* Z-Image occasionally ignores "no people" for distant pedestrians; one of two candidates may have awkward hands/props.
* ZIP is ~2.8 GB because the RealVis fallback needs PyTorch/CUDA; the default engines themselves are external.
* The EXE is unsigned (Windows SmartScreen may ask once).
* Interrupted/failed attempts leave their partial job folders in the output (no data is deleted automatically).

## Merge decision

**READY WITH MINOR LIMITATIONS** — every READY criterion is met (settings persistence, no quality regression,
strict product preservation, queue endurance, packaged EXE, 20-job validation mostly ship-ready, no data-loss bug);
the items above are known, documented and non-blocking. Not merged automatically: the merge to main is the user's call.

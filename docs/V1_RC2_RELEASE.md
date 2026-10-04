# CoverMorph Studio v1.0.0-rc2 — release notes and validation

Branch `v1.0-finalization` (base `v1.0-production-hardening`). No new models. Measured 2026-10-04 on the user's
RTX 3060 12 GiB. Review set: `D:\03_youtubesum\USER_REVIEW_V1_FINAL\FINAL_REVIEW_SHEET_RC2.jpg` (4090×5140).

## What changed since rc1

| Area | Change |
|---|---|
| Product default | **자연광 보정 합성 (추천)** is the default product workflow. **원본 그대로 합성** stays the absolute-preserve option. **AI 재구성 · 상품 형태/라벨 변형 가능** is never the default; its candidates carry a Korean warning, are ranked below composites, and OCR/colour checks flag label loss (the RC2 rebuild turned "LUMA" into an unreadable label — flagged). |
| Grounding | Two-layer contact shadow (tight contact line + soft ambient pool) in both composite modes; natural-light mode adds a directional cast shadow and blends only the semi-transparent anti-aliased rim toward the surroundings. |
| 자연광 강도 | 약하게 / 기본 (default) / 강하게 — background brightness pull, shadow opacity, rim blend. |
| Pixel safety | Opaque product pixels are identical to the photographed product in strict and in all three natural strengths: unit test + 16 real jobs (bottle, mug, box, irregular plant × 4 modes, 37k–153k pixels each). |
| Tokyo Chill couple preset | Medium-close two-shot (both faces readable, natural interaction, off-center, city still visible). Same seeds: old wording gave two women on one seed and stiff forward stares; new wording gave a man and a woman interacting on both seeds. |
| OLD POP presets | Lived-in, story-telling wording (natural skin with fine lines, personal objects, practical light, "not a stock photo"); added mature man and seasonal presets. Visibly less stock-like on all four same-seed tests. No nationality is added (see limitations). |
| Plain UI words | Normal screens show 사실적 생성 / 레퍼런스·편집 / 대체 엔진 and 처리 descriptions; model names only in 설정 (engine status, wizard check), 진단 정보 and the review's 기술 정보 block. |
| youtubesum discovery | CoverMorph records its EXE path in `%LOCALAPPDATA%\CoverMorphStudio\settings.json` when opened; youtubesum (v0.6-pro-editor-dev dd7b213) finds it automatically, or uses a program chosen once with "이미지 프로그램 설정…" (saved in `%LOCALAPPDATA%\YouTubeDynamicThumbnailStudio\image_bridge.json`). `IMAGE_PROGRAM_EXE` still works as a developer override. |

## youtubesum → CoverMorph end to end (packaged EXEs, no image-program/model env vars)

youtubesum EXE `--self-test-editor <sample> --bridge-generate` drives the real Pro Editor (real Tk events):
program discovery → 배경 생성 via the Image Bridge subprocess (CoverMorph rc2, Z-Image) → sidecars loaded →
title/badge/episode layers kept → title moved → save → reopen → export.

| Run | Program found by | Generate | Checks |
|---|---|---|---|
| Tokyo Chill person (Korean prompt) | CoverMorph 자동 연결 | 46.8 s | 20/20 PASS |
| Tokyo Chill couple | CoverMorph 자동 연결 | 39.0 s | 20/20 PASS |
| OLD POP couple | CoverMorph 자동 연결 | 35.7 s | 20/20 PASS |

All three exported 3 candidates at 1280×720; reopened documents and background were identical. youtubesum EXE also
opened by a plain no-argument launch. Shopify is not exposed in youtubesum; it was verified in CoverMorph's studio.
Outputs: `validation_results/rc2_youtubesum_e2e 유튜브/`.

## Endurance (CoverMorph rc2 EXE, `scripts/validate_endurance.py`, Unicode data/output paths)

A pause after current → app closed (1 done, 9 pending, paused) · B reopen → 재개 → closed during a strict product job
(job back to pending) · C GPU memory held before start → "GPU 여유 4004 MiB (필요 6500 MiB)" wait → released →
forced sd-cli kill → `BACKEND_CRASH|이미지 엔진 프로세스가 정상적으로 끝나지 않았습니다.` → 실패 재시도 → **10/10 done**.
After every run: 0 sd-cli / llama-completion / app processes, GPU 338 MiB (Δ 0). **PASS.**

## Tests

CoverMorph 229 passed / 3 skipped. youtubesum 82 tests OK (run without the user's IMAGE_PROGRAM_EXE /
COVERMORPH_MODELS_DIR variables; the one image-bridge test fails only when those variables are set, before and
after this change).

## Limitations

* OLD POP presets carry no nationality; on the same seeds the revised wording drifted to Western-looking people.
  For a Japanese channel write it in the prompt (e.g. "일본인 부부") or set `appearance_hint` in
  `config/creator_presets.json`.
* youtubesum's own A-candidate crop can place the title over faces (it reports "충돌 N건" and offers re-layout).
* The user profile still has `IMAGE_PROGRAM_EXE`, `COVERMORPH_MODELS_DIR`, `IMAGE_BRIDGE_TIMEOUT` as persistent
  user variables from development; they still work (env wins) but are no longer needed.
* Natural-light composites never relight the product itself; AI 재구성 can change labels (flagged).
* ZIP ~2.8 GB (RealVis fallback needs PyTorch); EXE unsigned.

# Quality Engine V2 — RTX 3060 comparison and selected defaults

Measured 2026-10-04 on the user's PC: RTX 3060 12 GiB (driver 616.92), i9-12900, 32 GB RAM,
commit limit ~62 GB, Windows 10. Other desktop apps (CapCut) were running; nothing was closed.
Engines were run **one at a time** (never two models resident).

## Engines

| Engine | Weights (file) | Quant | License | Commercial | Backend |
|---|---|---|---|---|---|
| Z-Image-Turbo | `leejet/Z-Image-Turbo-GGUF` `z_image_turbo-Q6_K.gguf` (sha256 `319f627b…`) | Q6_K | Apache-2.0 | yes | stable-diffusion.cpp `master-929-3f8527a` CUDA12, `--offload-to-cpu --diffusion-fa`, 8 steps, cfg 1.0 |
| FLUX.2-klein-4B | `leejet/FLUX.2-klein-4B-GGUF` `flux-2-klein-4b-Q8_0.gguf` (`0bba6951…`) | Q8_0 | Apache-2.0 | yes | same, 4 steps, cfg 1.0, euler |
| text encoder (both) | `unsloth/Qwen3-4B-GGUF` `Qwen3-4B-Q8_0.gguf` (`eed55523…`) | Q8_0 | Apache-2.0 | yes | |
| VAEs | FLUX.1 ae (`afc8e282…`), FLUX.2 ae (`d64f3a68…`) | bf16 | Apache-2.0 | yes | |
| RealVisXL V5.0 (control/fallback) | `SG161222/RealVisXL_V5.0` fp16 | fp16 | openrail++ | yes | Diffusers, DPM++ 2M Karras, 30 steps, cfg 5, IP-Adapter Plus for references |

Not used as defaults (licenses): FLUX.1-dev/Krea (non-commercial), Qwen-Image (research license),
InstantID/InsightFace (non-commercial face models).

`scripts/prepare_quality_v2_models.py` downloads all of the above into `<models>/quality_v2` and
verifies every SHA-256 (`--verify` to re-check).

## Time / memory (1280x720 unless noted)

| Engine | Seconds / image | GPU peak (total in use) | Job's GPU share | System commit after job |
|---|---|---|---|---|
| Z-Image-Turbo Q6_K | 34.9 | 7.3 GiB | +6.5 GiB | ~0 (process exits) |
| FLUX.2-klein Q8_0, text-to-image | 18.0 | 7.3 GiB | +6.5 GiB | ~0 |
| FLUX.2-klein, 1 reference / edit | 26.7 | 7.3 GiB | +6.5 GiB | ~0 |
| Shopify 1792x768 (Z-Image / FLUX.2) | 47.4 / 31.6 | 10.4 GiB | +9.6 GiB | ~0 |
| RealVisXL V5 (first load + image) | 52.6 | 11.9 GiB | +11.1 GiB | **+16.6 GB** while loaded |
| RealVisXL V5 (warm) | 29.1 | 11.9 GiB | | |

GPU memory went back to the exact pre-job baseline after every sd.cpp job (queue test below).

## Visual comparison (same scenes, same seeds, compiled prompts stored)

Folders: `validation_results/quality_v2/<scene>/` — `<engine>_full.png`, `_face.png`, `_340.png`,
`_180.png`, `_metrics.json` (original + compiled prompt, timing, VRAM, commit, QA) and
`compare_full/face/340/180.png`. References in `validation_results/quality_v2/_refs/`.

| Scene | Z-Image-Turbo | FLUX.2-klein | RealVisXL V5 |
|---|---|---|---|
| tc1 Tokyo solo woman side profile | best: film-like skin/hair, clear left title space | very good, readable shop signs | good face; harder, less natural light |
| tc2 Tokyo solo man | natural, good | good, slightly posed | good |
| tc3 Tokyo couple | best: natural couple, clean | good | readable fake signage on shops |
| tc4 rainy night, no people | clean empty street | clean | a person appeared despite "no people" |
| op5 OLD POP couple "in their fifties" | ages right | looks ~70, smiling/posed | standing couple, small faces |
| op6 OLD POP mature solo | good | good | good |
| sh7 Shopify lifestyle hero | good | good | good |
| sh8 product reference (mug) | — (no refs) | exact color + logo, scene changed; **duplicated handle** on some seeds | product kept but background not changed (prompt ignored) |
| rf9 person reference | — | same face/hair/sweater in a new scene | similar person + an unwanted second person |
| rf10 composition reference | — | palette + right placement, not the small-figure scale | copied layout most literally, subject unclear |
| ed11 edit: keep woman, change background | — | same woman, snowy night, title space | not supported |

Bridge end-to-end (`validation_results/bridge_v2/`): Korean prompt "비 오는 도쿄 거리의 젊은 여성 옆모습"
→ Z-Image canvas (35 s, 7.0 GiB) → edit "keep the people, change the background to a snowy night street"
→ FLUX.2 kept the same woman (hair, coat, bag) in a snowy street (27 s).

## Selected defaults

| Role | Engine | Why |
|---|---|---|
| Default text-to-image | **Z-Image-Turbo Q6_K** | Most natural/realistic people, best prompt adherence (age, empty scenes, textless), 7.3 GiB |
| Default reference / edit | **FLUX.2-klein-4B Q8_0** | Beats IP-Adapter visibly on subject preservation and instruction adherence; only engine that edits |
| Fallback | **RealVisXL V5** (existing SDXL path) | Used when sd.cpp/weights are missing or fail; `engine=legacy` forces it |

| Mode | No reference | With reference / edit | Memory profile |
|---|---|---|---|
| PREVIEW | FLUX.2 ×1 (~18 s) | FLUX.2 ×1 | interactive_low_memory |
| BALANCED | Z-Image ×2 | FLUX.2 ×2 | interactive_low_memory |
| BEST | Z-Image ×2 → FLUX.2 ×2 | FLUX.2 ×3 → RealVis+IP-Adapter ×1 (edit: FLUX.2 only) | night_best |

BEST runs engines sequentially and unloads before the next; every candidate is returned (scored on
technical QA only — face found/size/sharpness, exposure, duplicates — never appearance or identity).
BEST uses FLUX.2 instead of RealVis as the second text-to-image engine because RealVis produced readable
fake signage and ignored "no people" in this comparison.

The bridge (`--action generate/edit`) answers with one canvas: `engine=auto` (default) uses V2 when
installed, 1 candidate (`candidates` 1–4 to change; extras saved to `<project>/candidates/`),
`quality_mode` preview|balanced|best (old `quality_profile` fast→preview, quality→balanced).

## Low-memory policy

| Profile | Start a job only if GPU free ≥ | Commit free ≥ | RAM available ≥ |
|---|---|---|---|
| interactive_low_memory | 6.5 GiB | 12 GB | 6 GB |
| balanced_idle | 8 GiB | 10 GB | 5 GB |
| night_best | 9 GiB | 8 GB | 4 GB |

* One model/process at a time; sd.cpp exits after every job (no persistent pipeline).
* `--offload-to-cpu` always; interactive mode generates wide Shopify sizes at ≤1280x768 pixels and
  Lanczos-upscales (1792x768 peaked at 10.4 GiB).
* Below threshold the job stays queued with the reason recorded; it is not started.

## Queue (covermorph/job_queue.py)

Persistent JSON (atomic temp+replace), pause-after-current, resume, cancel one/all pending, job left
"running" by a crash returns to pending on restart. Pause is job-level (no mid-denoise checkpoint).

Real GPU test (`scripts/validate_quality_queue.py`, Korean/Japanese prompts, Unicode folder):
pause-after-current PASS, persisted across restart PASS, cancel pending + resume PASS,
GPU back to baseline (522 → 522 MiB) after each job PASS, low resources keep job queued PASS.

## Known limitations

* FLUX.2-klein can duplicate product parts (mug handle) on some seeds; the compiler's
  "same viewing angle / silhouette / number of parts" rule fixed 2/4 seeds. Product jobs carry a warning;
  use BALANCED/BEST for 2+ candidates.
* Composition references transfer placement/palette but not subject scale.
* The QA sharpness floor (tuned on RealVis) flags Z-Image's shallow depth-of-field faces as "soft" although
  they look sharp; it only affects ranking, never hides outputs.
* Readable fake signage can still appear in busy city scenes (all engines; least with Z-Image).
* sd-cli cannot open files under Korean/Japanese folder names (measured: "file not found"). The backend
  passes it 8.3 short paths or ASCII junctions under `%LOCALAPPDATA%\CoverMorph\sdcpp\links` and stages
  references/output in ASCII temp folders; a real FLUX.2 run from `D:\03_image\모델 モデル test` passed.

# Claude Code implementation brief — Quality Engine V2

Repository: 1976haru/image
Target working branch after current local person-quality work is safely committed: person-quality-dev
This file lives on quality-stack-v2-spec only as a transferable implementation brief.
Do not touch main.

## Goal

Stop treating one model as the answer.

The user cares primarily about:
1. visible image quality,
2. prompt following,
3. reference-image fidelity,
4. realistic people,
5. practical use on RTX 3060 12 GiB,
6. commercial/monetized workflows (YouTube + Shopify assets),
7. low enough memory pressure to coexist with normal PC use.

Implement a multi-engine quality stack and benchmark it on the user's actual RTX 3060.

## Recommended engine stack

### Engine A — Z-Image-Turbo: primary text-to-image candidate
Official model family: Tongyi-MAI/Z-Image-Turbo
License: Apache-2.0
Strengths:
- 6B architecture
- photoreal generation focus
- strong instruction following
- 8-NFE turbo generation
- commercially friendly license
- stable-diffusion.cpp supports Z-Image and quantized GGUF
- stable-diffusion.cpp documentation shows Z-Image can run with very low VRAM using GGUF/offload

Use cases:
- Tokyo Chill new backgrounds
- OLD POP backgrounds
- Shopify hero/collection/lifestyle images without strict reference identity

Implementation preference:
- use stable-diffusion.cpp subprocess backend first for low memory
- evaluate Q5_K/Q6_K quality if available; Q4 only as memory fallback
- Qwen3-4B text encoder quantized
- offload-to-cpu + diffusion flash attention
- final 16:9 and Shopify aspect ratios

### Engine B — FLUX.2-klein-4B: primary reference/edit engine
Official model: black-forest-labs/FLUX.2-klein-4B
License: Apache-2.0
Strengths:
- native T2I + image editing
- multi-reference support (up to 4 refs in official guidance)
- reference-based character/product/style consistency
- 4B
- designed for consumer hardware
- official model card says about 13GB VRAM; user's 12 GiB is slightly below, so MUST use offload/quantized path
- stable-diffusion.cpp supports FLUX.2 klein, GGUF and reference-image editing

Use cases:
- person reference + scene reference
- product reference + lifestyle background
- Shopify product/collection banners
- edit existing image while preserving subject
- character consistency

Implementation preference:
- stable-diffusion.cpp GGUF backend first
- Q5/Q6 if memory allows, Q4 fallback
- --offload-to-cpu and diffusion flash attention
- 4-step distilled model for interactive mode
- optional base 4B only for offline/night queue experiments

### Engine C — RealVisXL V5.0: proven SDXL fallback
Model: SG161222/RealVisXL_V5.0
License: openrail++
Strengths:
- already visibly better than SDXL Base on this machine
- existing Diffusers/SDXL bridge path is proven
- fast relative to heavier modern models
- works with existing IP-Adapter code

Use cases:
- fallback when new engines fail
- comparison baseline
- offline second opinion for portraits

Keep current photoreal FAST/QUALITY path.

### Do NOT make these defaults for monetized workflows
- FLUX.1-dev / FLUX.1-Krea-dev: non-commercial model license
- Qwen-Image-2.1: current official release uses Qwen Research License, non-commercial without separate license
- InstantID/InsightFace stack: model/checkpoint licensing includes research/non-commercial restrictions

They may be research-only optional backends, clearly labeled, but never automatic defaults for YouTube/Shopify commercial output.

## Architecture

Create a backend interface:

ImageBackend
- name
- license_id
- commercial_ok
- supports_t2i
- supports_single_reference
- supports_multi_reference
- supports_edit
- supports_transparency
- estimated_vram
- prepare()
- status()
- generate()
- edit()
- unload()

Backends:
- ZImageCppBackend
- Flux2KleinCppBackend
- RealVisDiffusersBackend

The bridge remains backward-compatible.

## Quality modes

PREVIEW:
- one fast engine only
- low-memory
- 1 candidate
- auto unload after result

BALANCED:
- primary engine chosen by task
- 2 candidates/seeds
- quality checks
- auto unload

BEST:
- sequential ensemble, NOT parallel in memory
- run two engines one at a time
- unload first before loading second
- candidate pool:
  - no refs: Z-Image-Turbo + RealVisXL
  - refs/edit: FLUX.2-klein + RealVisXL/IP-Adapter fallback
- generate 2 candidates per engine when practical
- automatic technical scoring
- show all candidates to user; do not let the scorer hide outputs

Sequential ensemble is mandatory on this machine because the user runs other desktop apps. Do not keep two diffusion models resident simultaneously.

## Candidate technical scoring

Score only technical quality:
- face detected when expected
- face size
- face sharpness
- gross facial distortion heuristics
- duplicate person warning
- exposure
- clipping
- subject-safe text space
- prompt/reference compliance where measurable

Do NOT score attractiveness, race, age desirability, or identity quality from inferred demographic traits.

Final selection remains user's choice.

## Reference input design

Support explicit reference roles:
- PERSON
- PRODUCT
- STYLE
- COMPOSITION
- BACKGROUND

For FLUX.2 multi-reference, construct prompts that explicitly assign each image role.

Example:
Image 1 = person reference
Image 2 = clothing/style reference
Image 3 = composition reference

"Use the same person from image 1, adopt the clothing mood from image 2, and follow the subject placement/negative space of image 3..."

Never silently mix roles.

## Prompt compiler

Create engine-specific prompt compilers.

### Z-Image compiler
- detailed natural-language scene prompt
- strong subject-first wording
- camera/lens/lighting
- explicit clean negative space
- textless/branding constraints
- use negative prompt if backend supports it

### FLUX.2 compiler
- positive descriptions only
- Subject + Action/Pose + Medium + Context + Lighting + Camera
- narrative prose
- explicit reference-index roles
- no negative-prompt dependence

### RealVis compiler
- SDXL photoreal prompt + negative prompt
- preserve current tested defaults

## User master prompt compatibility

The system must accept:
- direct prompt
- channel preset
- master prompt
- reference files

Do not overwrite user prompts.
Build:
user prompt + channel quality envelope + engine-specific rewrite

Store both original and compiled prompts in manifest.

## Shopify support

Add generation purposes:
- youtube_thumbnail_background
- shopify_hero_banner
- shopify_collection_banner
- shopify_product_lifestyle
- shopify_promo_tile
- shopify_mobile_banner

Add configurable canvas presets; do not hardcode only one Shopify theme size.

For product references:
- preserve exact product shape/color/logo as much as the model permits
- prefer FLUX.2 reference/edit backend
- warn when generated product deviates from reference
- never fabricate product claims/text in the image

## Memory policy

Add resource profiles based on current free system commit + GPU memory.

INTERACTIVE_LOW_MEMORY:
- stable-diffusion.cpp quantized backend preferred
- one candidate at a time
- immediate model/process exit after each job
- no persistent pipeline
- target VRAM comfortably below 10 GiB
- avoid large RAM offload when system commit is high

BALANCED_IDLE:
- more GPU/RAM allowed
- still one backend resident at a time

NIGHT_BEST:
- sequential ensemble
- can use heavier quantization/settings
- user is away

Before every job:
- inspect Windows available RAM/commit
- inspect GPU free/used VRAM
- if below threshold, leave job queued instead of starting

## Queue / pause / resume foundation

Implement now enough infrastructure for:
- queued jobs
- pause-after-current
- resume
- cancel pending
- completed jobs persist
- model process exits between jobs in low-memory mode

Do not attempt impossible mid-denoise checkpoint resume.
Pause/resume is job-level:
- finish current image
- stop before next job
- resume later

Persist queue atomically so app restart does not lose it.

## Evaluation matrix

Use the SAME fixed test set and prompts:
1. Tokyo Chill solo woman side profile
2. Tokyo Chill solo man
3. Tokyo Chill couple
4. Tokyo Chill rainy night
5. OLD POP mature couple
6. OLD POP mature solo
7. Shopify generic lifestyle hero
8. Shopify product-reference banner

For each engine applicable:
- 1280x720 or target Shopify output
- face/product crop
- 340px preview for YouTube
- timing
- GPU peak
- RAM/commit delta
- reference fidelity notes
- prompt adherence notes

Do not compare with different prompts unless also reporting the compiled prompt.

## Acceptance gate

Primary generation engine must visibly beat current RealVisXL on at least:
- prompt adherence OR realism
- face detail at thumbnail scale
- usable text-space composition

Reference engine must visibly beat current IP-Adapter path on:
- reference subject preservation
- edit instruction adherence
- composition control

If Z-Image or FLUX.2 does not beat the existing pipeline, keep it optional and do not switch defaults.

## Integration

Keep the existing:
- thumbnail bridge contract
- youtubesum sidecars
- atomic output
- unicode paths

Add manifest fields:
- backend
- model
- quantization
- model_license
- commercial_use_flag
- original_prompt
- compiled_prompt
- reference_roles
- quality_mode
- memory_profile
- timing
- peak_vram
- system_commit_before/after

## Final report

Separate:
A. Z-Image real run
B. FLUX.2-klein real run
C. RealVis control
D. reference comparison
E. Shopify test
F. low-memory/queue test

Report exact:
- weights/repositories
- quantization
- hashes if practical
- license
- generation time
- VRAM
- system memory
- visual comparison paths
- final chosen defaults

Do not claim success merely because a model runs.

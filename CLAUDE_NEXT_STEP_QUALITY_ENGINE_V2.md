# NEXT STEP — Quality Engine V2 kickoff

## First: protect current local work

You are likely currently on `person-quality-dev` with local commits/changes that may not yet exist on origin.

Before any new implementation:

1. `git status`
2. `git branch --show-current`
3. confirm branch is `person-quality-dev`
4. commit any intended source/test/doc changes
5. push `person-quality-dev`
6. confirm working tree is clean
7. DO NOT reset/clean/rebase away local work

Do not switch to `quality-stack-v2-spec` as the working branch.

Read the spec directly with:

```powershell
git fetch origin
git show origin/quality-stack-v2-spec:CLAUDE_QUALITY_ENGINE_V2.md
```

Then continue implementation on `person-quality-dev`.

---

# Immediate execution order

Do not attempt all models at once. Work in gates.

## Gate 0 — environment + storage audit

Report:
- detected GPU
- total/free VRAM
- Windows RAM/commit
- free disk space on D:
- current models directory
- current stable-diffusion.cpp presence/build status
- current CUDA build status

Do not download anything yet.

## Gate 1 — Z-Image feasibility

Goal:
prove or disprove Z-Image-Turbo on this RTX 3060 with a low-memory backend.

Tasks:
- inspect official model/backend requirements
- prepare stable-diffusion.cpp CUDA backend if not already present
- choose the highest practical GGUF quantization that fits the machine
- prefer Q6/Q5; fall back to Q4 only if necessary
- keep model files external to git
- one real Tokyo Chill human test
- one real scenery test

Record:
- exact model/quantization
- generation resolution
- time
- peak VRAM
- system commit delta
- face quality crop
- 340px/180px preview
- prompt adherence
- text-space usability

Gate decision:
- if visibly better than RealVisXL or materially more prompt-compliant at similar quality, keep as candidate
- otherwise mark optional and proceed

## Gate 2 — FLUX.2 klein 4B feasibility

Goal:
prove or disprove it as the reference/edit engine.

Tasks:
- use official/commercial-compatible 4B klein model
- low-memory/quantized path suitable for 12 GiB GPU
- test one PERSON reference
- test one PRODUCT reference
- test one COMPOSITION reference
- test one edit preserving subject while changing background/text space

Record:
- reference fidelity
- prompt/edit adherence
- runtime
- VRAM/RAM
- visible artifacts

Gate decision:
- must visibly beat current IP-Adapter reference path to become default
- otherwise remain optional

## Gate 3 — controlled comparison

Same scenes/prompts:
- Z-Image
- FLUX.2 klein when applicable
- RealVisXL current best

Validation set minimum:
1. Tokyo solo woman side-profile
2. Tokyo solo man
3. Tokyo couple
4. OLD POP mature couple
5. Shopify generic lifestyle
6. Shopify product-reference

Keep source prompts and compiled prompts in the report.

## Gate 4 — choose defaults

Only after visual comparison:

- default T2I engine
- default reference/edit engine
- fallback engine
- PREVIEW/BALANCED/BEST mapping

Do not choose defaults from speed or unit tests alone.

## Gate 5 — queue/low-memory foundation

After default engines are selected:
- persistent queue
- pause-after-current
- resume
- cancel pending
- one model/backend resident at a time
- unload/exit model process after each job in interactive low-memory mode
- resource threshold check before starting next job
- if memory low, keep job queued instead of crashing

---

# Progress reporting format — mandatory

At every meaningful checkpoint, report exactly these six lines before continuing:

```text
전체 진행률: NN%
현재 단계: ...
완료: ...
진행 중: ...
남은 것: ...
현재 리스크/막힘: ...
```

Also report a stage-specific success probability:

```text
현재 단계 성공 가능성: NN%
전체 프로젝트 성공 가능성: NN%
```

Percentages are estimates, not test results.

---

# Stop conditions requiring user action

Stop and ask before:
- closing user applications
- deleting files
- downloading a multi-GB model when there is insufficient disk space
- changing pagefile/system settings
- replacing the user's current proven model files
- accepting a license that requires explicit user acceptance/payment

You may install/build normal open-source runtime dependencies inside the project environment when needed.

---

# Final rule

The project only passes this milestone if actual output quality improves visibly.

A backend that merely runs is not a success.

Final deliverables:
- side-by-side comparison folders
- model/license table
- memory/time table
- selected defaults
- queue/low-memory test
- commit SHA
- EXE path + SHA-256
- remaining limitations

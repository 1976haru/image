# Claude Code implementation brief — Person Quality v1

Repository: `1976haru/image`
Branch: `person-quality-dev`
Base: `thumbnail-bridge-dev`
Do not touch `main`.

The previous milestone achieved REAL AI bridge PASS on RTX 3060, but the actual generated human quality is not acceptable for production thumbnails. Treat this as a visual-quality failure.

---

# 1. Quality target

For generated human subjects, target:
- realistic skin, not waxy/plastic
- natural eyes and eyelids
- believable nose/mouth proportions
- clean hair strands and silhouette
- no melted fingers/hands when visible
- no asymmetrical or distorted face
- no over-smoothed face
- no muddy low-detail face
- strong subject/background separation
- face remains readable at 340px thumbnail preview
- image should look like a cinematic photograph, not generic base-SDXL art

Acceptance is visual, not only automated.

---

# 2. Replace SDXL base as the default human-quality model

Keep SDXL base 1.0 available as a compatibility fallback, but do not use it as the preferred final human generator.

Implement a model profile system:

- `sdxl_base` — compatibility/fallback
- `photoreal_sdxl` — default for people
- future profiles allowed

First candidate to evaluate:
- `SG161222/RealVisXL_V5.0`
- Diffusers-compatible SDXL fine-tune
- license metadata: openrail++
- source model card must be recorded

Do not silently download large model weights during ordinary generation.
Use the existing external models directory and explicit prepare/download flow.

If another photoreal SDXL model is tested, record its exact repository and license separately.

Do not use a model with unclear redistribution/use terms as the default shipped model.

---

# 3. Model selection policy

Bridge request should be able to specify:
- `quality_profile`
- `model_id`
- `person_quality=true/false`

Default policy:
- if scene contains people => `photoreal_sdxl`
- scenery-only => current/general profile is allowed
- reference identity request => photoreal model + identity-preserving path if ready

Expose selected model/profile in project_manifest.json.

---

# 4. Portrait-first composition strategy

The current failure is partly because faces are often too small in a 1344x768 generation.

For human-centric thumbnails:
- do not generate tiny faces and hope they survive resizing
- enforce a minimum subject scale for PERSON candidate
- prefer medium shot / medium close-up for A candidate
- face target should usually occupy at least ~12–18% of frame height for solo subject, more when appropriate
- for two-person scenes, keep both faces large enough to remain readable

Implement composition profiles:
- SOLO_CLOSE
- SOLO_MEDIUM
- COUPLE_MEDIUM
- COUPLE_EMOTIONAL
- SCENERY_WITH_PERSON

Generate A candidate with people larger than B/C by design.

---

# 5. Face detection upgrade

The current Haar cascade is too weak for side-profile/small faces.

Add a better optional face detector using OpenCV YuNet / FaceDetectorYN.

Requirements:
- keep Haar fallback
- support frontal and moderate-profile faces better than Haar
- model file remains external or bundled only if license allows
- record detector used in manifest
- Unicode paths must work

Use face boxes for:
- quality inspection
- face detail pass
- safe zones
- subject scale validation

Do not infer gender or identity from face detection.

---

# 6. Face detail second pass

Implement an optional high-quality face-detail stage after the main generation.

Preferred approach:
1. detect face box
2. expand crop with context (hair/chin/neck)
3. upscale working crop to a useful resolution
4. perform a low-denoise img2img/inpaint-style refinement using the SAME photoreal model family where practical
5. blend the refined face region back with feathering
6. preserve expression/composition as much as possible

Target:
- improve micro-detail without changing the person into a different person

Suggested controls:
- face_detail_enabled
- denoise_strength around a conservative range
- crop expansion ratio
- detail steps
- max faces to refine

Do not over-refine tiny/background faces.

If a real face-detail backend cannot preserve the face reliably, return the original instead of making it worse.

---

# 7. Identity/reference preservation

The current generic IP-Adapter Plus is not sufficient for strong identity preservation.

Add an optional identity profile using InstantID only if all dependencies/models are present and the license terms are acceptable for the user's workflow.

Candidate:
- `InstantX/InstantID`
- model card license: Apache-2.0

Requirements:
- optional, not mandatory for normal generation
- reference image explicitly supplied by user
- do not infer identity from arbitrary files
- status must say whether identity backend is ready
- if not ready, fall back to generic reference mode with a warning

Keep reference model weights outside git.

---

# 8. Prompt quality

For people, build a specialized prompt template around the user scene prompt.

Positive concepts:
- cinematic photography
- natural skin texture
- detailed eyes
- realistic hair
- natural facial proportions
- subtle expression
- realistic lighting
- shallow depth of field when appropriate
- high-end editorial photography

Negative concepts:
- deformed face
- asymmetrical eyes
- malformed hands
- extra fingers
- fused fingers
- plastic skin
- waxy skin
- over-smoothed face
- doll-like face
- blurry face
- low-detail face
- duplicate person
- extra person
- text
- logo
- watermark

Do not make every image hyper-sharp or HDR.
Preserve the user's cinematic Tokyo Chill / Old Pop aesthetic.

---

# 9. Scheduler/settings evaluation

For each photoreal candidate model, test a small matrix, not dozens of random runs.

Compare:
- 20–24 steps
- 28–32 steps
- guidance appropriate to the chosen model card
- current balanced memory profile

Keep seed fixed when comparing model/settings.

Do not assume more steps means better face quality.

---

# 10. Two-stage quality mode for RTX 3060

Add:
- FAST
- QUALITY

FAST:
- single pass
- good for iteration

QUALITY:
- photoreal base generation
- face detection
- optional face-detail pass
- final resize/sharpen
- sidecars

Show estimated/actual time in bridge response.

The user has an RTX 3060 12 GiB. Optimize QUALITY so it is usable, even if total time becomes ~45–90 seconds per image.

---

# 11. Output post-processing

Final 1280x720 should avoid:
- oversharpen halos
- oversaturated orange skin
- excessive contrast
- crushed blacks
- cyan/orange AI look

Add restrained final processing:
- mild local contrast
- subtle sharpening only
- skin-safe saturation
- preserve natural white balance

Do not apply beauty smoothing.

---

# 12. Visual QA gate

After generation, compute basic technical checks:
- at least expected number of faces when the scene requires people
- minimum face size
- face sharpness
- gross exposure
- duplicate face/person warning if detectable

If face quality is clearly below threshold:
- warn
- optionally auto-regenerate once with a new deterministic seed in QUALITY mode
- never infinite-loop

Do not reject purely because the person is not conventionally attractive. Check technical image quality only.

---

# 13. Validation set

Create a fixed validation set of at least:

Tokyo Chill:
1. solo woman, evening Tokyo street, medium close-up
2. solo man, train platform, medium shot
3. young couple, cafe/window, two faces visible
4. young couple, night street, cinematic backlight
5. side-profile woman, rainy street

OLD POP:
6. mature couple, autumn park
7. mature woman, first snow
8. mature man, cafe/window

For each compare:
- current SDXL base
- photoreal profile
- photoreal + face detail

Same or controlled seeds where practical.

Save:
- full 1280x720
- face crop
- 340px preview
- 180px preview
- timing
- VRAM
- detector confidence
- face sharpness metric

---

# 14. Human visual scoring

Create a simple reviewer sheet, not an AI self-congratulatory score.

Columns:
- face realism 1–5
- eyes 1–5
- hair 1–5
- anatomy/hands 1–5
- subject sharpness 1–5
- thumbnail readability 1–5
- overall 1–5

Do not declare success unless the photoreal pipeline is visibly better than SDXL base across the validation set.

---

# 15. youtubesum integration

Do not break the current bridge protocol.

Add optional request fields under `options`:
- quality_profile
- person_quality
- face_detail
- identity_mode

If absent, preserve backward compatibility.

Expose in manifest:
- generation_model
- quality_profile
- face_detector
- face_detail_applied
- identity_backend
- quality_warnings

The Pro Editor does not need to understand model internals.

---

# 16. Licensing and model handling

Record model metadata in README:
- model repository
- license identifier
- whether weights are included (should normally be no)
- external model directory
- preparation steps

Do not commit model weights.

Before recommending a model for monetized/public workflows, clearly surface its license in the UI/docs.

---

# 17. Final acceptance

Required:
- all old bridge tests PASS
- new person-quality tests PASS
- packaged EXE PASS
- real RTX 3060 generation PASS
- current bridge/youtubesum handshake PASS
- visual comparison outputs produced

Final report:
- exact models
- exact settings
- GPU/VRAM
- FAST timing
- QUALITY timing
- VRAM peaks
- face detector used
- face detail applied?
- identity backend ready?
- comparison table
- remaining artifacts
- final commit SHA
- EXE path/SHA-256

Do not call this milestone PASS if the only improvement is metric/test success but the faces still look visibly poor.

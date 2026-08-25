# Codex Implementation Prompt — GolfBallFinder for iPhone 16 Pro

## Mission

You are the implementation agent for **GolfBallFinder**, an AGPL-3.0-only open-source iPhone application whose only essential job is to help a user find a golf ball in grass or deep rough using the rear camera.

The user may have difficulty visually locating a ball. The app must therefore prioritize **finding the ball reliably, quickly, and with a conspicuous indication**, not visual polish or feature breadth.

Target hardware: **iPhone 16 Pro**.
Target deployment: **physical iPhone 16 Pro**. The primary workstation may be Windows with **no physical Mac**. In that case, use hosted macOS/Xcode CI and deliver builds through TestFlight.
Primary constraint: **perform inference locally on-device; do not introduce cloud inference into the MVP.** Cloud macOS is permitted only for compilation, Core ML export, code signing, and distribution.

This repository is intentionally structured so you can proceed without re-designing the product. Read the repository first, then implement, compile, test, and fix forward.

---

## Repository source of truth

Read these before modifying architecture:

1. `README.md`
2. `docs/DEVELOPMENT_SPEC.md`
3. `docs/DATASET_PLAN.md`
4. `docs/TEST_PLAN.md`
5. `docs/RESEARCH_NOTES.md`
6. `THIRD_PARTY_NOTICES.md`
7. `AGENTS.md` if present

The starter implementation is under `GolfBallFinder/`; training/export utilities are under `training/` and automation helpers are under `scripts/`.

If the current machine is Windows, also read `docs/WINDOWS_CLOUD_BUILD.md`, run `scripts/bootstrap_windows.ps1`, and maintain `codemagic.yaml`. Do not block on the absence of Xcode locally. Use the cloud compile workflow as the Apple-toolchain gate.

---

## Product definition

### Core user flow

1. User opens the app near the approximate landing point.
2. The rear camera opens immediately.
3. User slowly sweeps the phone across grass/rough.
4. On-device AI analyzes the live camera stream.
5. When a plausible golf-ball candidate appears, the app focuses additional inference on that region.
6. A detection is not considered confirmed from a single frame. It must pass a short temporal/spatial consistency test.
7. When confirmed, the app displays a large high-visibility ring and direction cue, and emits haptic/audio feedback.
8. The confirmed position remains visually stable enough for the user to walk toward it.

### Explicit non-goals for MVP

Do **not** add these unless required to unblock the core task:

- accounts, login, analytics SDKs
- cloud inference or server APIs
- scorecard/GPS/course-management features
- social features
- subscriptions/payments
- elaborate settings screens
- LiDAR-based ball detection as the primary detector
- general object recognition
- public App Store production release (TestFlight packaging for private device testing is in-scope)

The app succeeds only if it helps locate the ball.

---

## Technical architecture — do not change without evidence

### iOS

- Swift + SwiftUI
- AVFoundation for live camera acquisition
- Core ML / Vision-compatible on-device inference
- Ultralytics iOS SDK for the first MVP inference path
- iOS 17+ deployment target
- physical target device: iPhone 16 Pro

### Model

Initial detector class set: exactly one semantic target, `golf_ball` (model metadata may initially expose another label such as `ball`; normalize in app/training where practical).

Preferred training path:

- YOLO26n or YOLO11n for new training runs
- 640×640 detector input for initial experiments
- Core ML export for device deployment
- start with FP16; benchmark INT8 only after FP16 correctness is established

A public golf-ball YOLOv8n checkpoint is provided as a **seed/bring-up option**, not as the expected final production model. Its job is to get a golf-ball-specific model into the iOS loop quickly. Replace it with a model trained/fine-tuned on field data as soon as practical.

### Device compute

Prefer Apple Neural Engine-capable Core ML execution. Avoid making the GPU both the live-preview renderer and exclusive inference resource unless measurements prove it is better on the target device.

Never assume simulator performance represents the iPhone.

---

## Critical detection strategy

This is a small-object problem. A golf ball becomes only a few pixels wide at useful search distances after a whole camera frame is resized to a 640×640 detector input. Therefore the MVP must preserve the repository's **adaptive tiled inference** strategy.

### ScanScheduler behavior

Use one inference job at a time unless profiling proves parallel inference is beneficial.

Default cycle:

1. infer on the full frame for fast broad coverage;
2. if no durable candidate exists, scan overlapping tiles over subsequent frames;
3. when a candidate is detected, temporarily lock inference to an expanded ROI around it;
4. if the candidate disappears for long enough, release the lock and resume broad/tiled scanning.

Do not naively run four or more 640×640 tiles on every camera frame. That increases latency, thermal load, and battery consumption unnecessarily.

### Candidate confirmation

Single-frame detections are **candidates**, not confirmed balls.

Default temporal policy is implemented as approximately:

- sliding 5-frame history;
- require at least 3 spatially consistent positive observations;
- centers must be close enough in normalized image coordinates;
- candidate confidence threshold may be lower than final confidence because temporal confirmation is used to recover precision.

Tune these values using field evidence rather than aesthetics.

### Candidate ROI lock

When a candidate appears:

- expand its normalized bounding box to a useful ROI;
- crop that region from subsequent full-resolution frames;
- feed the crop to the detector so the suspected ball occupies more input pixels;
- map detection coordinates back to full-frame normalized coordinates;
- keep the visual overlay in full-frame coordinates.

### Tracking

The first implementation may use the stabilizer/ROI lock without a separate Vision tracker. After baseline correctness, evaluate `VNTrackObjectRequest` or another lightweight tracker to reduce repeat detector work and stabilize the overlay. Keep the detector as the authority for reacquisition.

---

## Accuracy strategy — this is more important than UI work

The primary risk is **false positives and misses in real rough**, not model inference plumbing.

### Hard negatives required

Collect and train on negative scenes containing common confounders, including:

- mushrooms
- white/yellow flowers
- dry leaves and pale leaf edges
- stones/pebbles
- tees and ball markers
- litter
- dew/specular highlights
- patches of sand
- white shoes/clothing entering frame
- sunlit grass tips
- small round seed heads

Negative images with **no golf-ball annotation** are useful. If the training format/model expects images with no boxes, retain them rather than discarding them.

### Positive diversity required

Include golf balls that are:

- clean and dirty
- white and common colored variants if relevant
- in full sun, shade, mixed light, backlight
- wet
- blurred by camera motion
- near frame borders
- at multiple distances
- partially hidden by roughly 10%, 30%, 50%, and 70% vegetation occlusion
- visible only as an arc/partial sphere through blades of grass

Do not promise detection when the ball is completely visually occluded. RGB camera inference cannot recover an object with no visible evidence.

### Dataset split rule

Never randomly split adjacent frames extracted from one video into train/validation/test.

Split by **recording session / scene / video** so near-duplicate frames do not leak across datasets.

---

## Evaluation metrics

Do not optimize only mAP. The product metric is field-search success.

Track at least:

- frame-level precision/recall and mAP50/mAP50-95 for model comparison;
- search-session success: ball found within a 10-second sweep;
- success by occlusion bucket;
- false alert rate per 10 seconds of negative scanning;
- time from first visible evidence to confirmed UI indication;
- device inference latency and end-to-end UI latency;
- thermal behavior over a sustained 5-minute scan;
- battery impact qualitatively during field tests.

Initial practical targets, to be refined with data:

- clearly visible ball: >= 95% search-session success;
- moderate rough / ~50% occlusion: >= 80% search-session success;
- false confirmed alert: < 1 per 10 seconds of deliberate negative scan, with a longer-term goal substantially lower;
- overlay response feels immediate after temporal confirmation;
- no network dependency.

Do not falsify measurements. Record the exact model, thresholds, device, resolution, and test set used.

---

## Public research and reusable technology already identified

Use these as engineering inputs, not as claims that the problem is solved.

### 1. Ultralytics YOLO iOS SDK

Repository:
`https://github.com/ultralytics/yolo-ios-app`

Why useful:

- existing Swift integration for YOLO/Core ML;
- live camera inference path;
- Core ML model handling;
- bounding-box results and performance instrumentation;
- avoids spending the MVP on generic camera/model plumbing.

Important license note:

- the public SDK/repository is AGPL-3.0 at the pinned version used by this project;
- Policy A is selected: GolfBallFinder and its selected fine-tuned checkpoint use the public AGPL-3.0-only path;
- before external TestFlight distribution, publish the matching Complete Corresponding Source and checkpoint as
  required by `docs/PUBLIC_RELEASE_CHECKLIST.md`.

The architecture must therefore keep `GolfBallDetector` behind a small interface so the SDK can later be replaced without rewriting the camera/UI/search logic.

### 2. SAHI — Slicing Aided Hyper Inference

Repository:
`https://github.com/obss/sahi`

Paper:
`https://arxiv.org/abs/2202.06934`

Why useful:

- establishes the value of sliced/tiled inference for small-object detection;
- Python SAHI itself should not be embedded in iOS;
- implement the relevant concept natively in Swift: overlapping image crops, per-tile inference, remapping to global coordinates, and deduplication if multiple tiles are evaluated in one aggregation window.

SAHI is a conceptual/reference dependency here, not a runtime dependency.

### 3. Public seed golf-ball model

Hugging Face model:
`https://huggingface.co/notjulietxd/golf-ball-tracker`

Published model card reports approximately:

- YOLOv8 nano architecture;
- 559 real + 500 synthetic training images;
- 640×640 input;
- mAP50 around 0.812 and mAP50-95 around 0.586;
- Apache-2.0 repository/model-card licensing information.

The repository includes `scripts/fetch_seed_model.py`, which pins the currently researched `best.pt` SHA-256 to reduce silent model drift.

Treat third-party PyTorch checkpoints as third-party executable serialization. Only load a checkpoint whose origin and checksum you have consciously accepted. If policy/security requirements become stricter, do not load it; train from an official Ultralytics base model instead.

This checkpoint is not evidence of field-grade performance in deep rough. Use only for bring-up and comparison.

### 4. Roboflow Universe golf-ball datasets

Useful public examples found during research include:

- smartphone-oriented golf-ball dataset/model (~109 images, YOLO11n published metrics around mAP50 83.8%, precision 83.3%, recall 76.2%);
- larger golf-ball datasets in the low-thousands of images;
- examples labeled CC BY 4.0.

Do not bulk-merge datasets without checking each dataset's exact license, class ontology, annotation quality, duplication, and train/test provenance. Record source attribution in the dataset manifest.

Public datasets are seeds. The final model must be adapted to iPhone 16 Pro footage and the user's actual rough/grass conditions.

### 5. Apple frameworks

Relevant Apple technologies:

- AVFoundation `AVCaptureVideoDataOutput` for camera frames;
- Core ML for local execution across CPU/GPU/Neural Engine;
- Vision `VNCoreMLRequest` for model requests if/when the Ultralytics adapter is replaced;
- Vision object tracking APIs for post-detection tracking experiments;
- iPhone 16 Pro A18 Pro / Neural Engine and camera system are sufficient to justify an on-device-first design.

LiDAR may later be explored as an auxiliary depth/size sanity signal. It is explicitly not the primary MVP detector.

---

## Existing starter implementation

The starter code already provides these conceptual modules. Preserve responsibility boundaries:

- `CameraController`: AVFoundation capture session, frame delivery, zoom, inference scheduling
- `GolfBallDetector`: model loading and detector adapter
- `ScanScheduler`: full-frame/tile/candidate-ROI selection
- `DetectionStabilizer`: short temporal/spatial confirmation
- `Geometry`: crop/remapping functions
- `DetectionOverlay`: high-visibility screen-space indication
- `FeedbackManager`: haptic/audio confirmation
- `AppConfig`: thresholds and tunables

If the starter code does not compile against the exact pinned SDK/Xcode version, fix it. Do not discard the architecture solely because an SDK API changed.

---

## Required work sequence

### Phase 0 — environment and repo validation

1. Confirm macOS and current Xcode are available.
2. Run `xcodebuild -version`.
3. Install XcodeGen if needed.
4. Create Python venv and install `training/requirements.txt`.
5. Review the seed-model security/license note before running it.
6. Run `./scripts/bootstrap_mac.sh` if its assumptions match the machine; otherwise perform its steps manually and update the script.
7. Generate the Xcode project from `project.yml`.

Do not stop at describing commands: execute everything you can in the environment and report exact blockers.

### Phase 1 — compile and unit tests

1. Resolve Swift package dependencies.
2. Compile for an available iOS simulator.
3. Run unit tests.
4. Fix all compile errors, concurrency warnings that are correctness-relevant, geometry bugs, and SDK API mismatches.
5. Keep pure logic under unit test: crop mapping, scheduler state transitions, and temporal confirmation.

Passing simulator compile is necessary but not sufficient.

### Phase 2 — model bring-up

If no custom field model is available:

1. obtain the pinned seed model using `scripts/fetch_seed_model.py` only if accepted;
2. export it to Core ML using `training/export_coreml.py`;
3. place/copy `GolfBall.mlpackage` into `GolfBallFinder/Resources/`;
4. regenerate/rebuild if XcodeGen resource references require it;
5. verify model load state and surface model-load failures in-app instead of silently scanning with no model.

If the seed cannot export with the current Ultralytics/coremltools versions, investigate and pin a compatible toolchain or use a freshly trained YOLO26n/YOLO11n model. Document the exact resolution.

### Phase 3 — physical iPhone 16 Pro run

1. Connect/select the iPhone 16 Pro.
2. Set the user's Development Team/signing configuration without committing personal identifiers.
3. Install and launch.
4. Approve camera permission.
5. Verify:
   - preview orientation and aspect fill;
   - no significant image/overlay coordinate mismatch;
   - detector loads;
   - inference is local;
   - zoom toggle works;
   - scan/tile schedule progresses;
   - candidate ROI lock and release works;
   - haptic/audio fire only on confirmed transition, not every frame.

### Phase 4 — baseline field evaluation

Create a repeatable test collection before threshold tuning.

Record at least:

- close clean balls on short grass;
- 1–3 m sweeps on rough;
- multiple partial occlusions;
- negative scenes rich in mushrooms/leaves/stones/highlights;
- light/shade transitions;
- ordinary hand sweep motion.

Store source videos outside Git if large. Keep metadata/manifest in Git.

Run the seed/base model and record actual failure cases.

### Phase 5 — custom training

1. Extract frames at a non-redundant cadence with `training/extract_frames.py`.
2. Annotate ball boxes tightly, including partially visible balls where enough visual evidence exists.
3. Preserve negative frames.
4. Split by session/video.
5. Populate a real `dataset.yaml` from the example.
6. Train with `training/train.py`.
7. Evaluate with `training/evaluate.py` on held-out scenes.
8. Export best checkpoint to Core ML.
9. Put it on-device and repeat exactly the same field benchmark.

Choose a new model only if it beats the previous one on held-out and field/session metrics, not merely training loss.

### Phase 6 — precision/recall refinement

In this order:

1. add hard negatives from actual false alerts;
2. add missed-ball positives from actual false negatives;
3. tune candidate confidence threshold;
4. tune 3-of-5 temporal confirmation and spatial tolerance;
5. tune tile overlap/coverage and ROI expansion;
6. test 1× versus 2× capture/search modes;
7. only then consider a second-stage crop classifier if false positives remain problematic.

Do not add a second model before evidence shows temporal confirmation + hard-negative retraining is insufficient.

### Phase 7 — optional tracking optimization

After the detector is reliable:

- benchmark Vision object tracking from a confirmed box;
- use detector periodically for validation/reacquisition;
- ensure tracking never converts a stale false detection into a permanent lock;
- measure end-to-end latency/thermal impact before keeping the optimization.

---

## UI specification

Keep UI minimal and accessibility-oriented.

### Scanning state

- camera preview fills screen;
- small status indicator: `SCANNING`;
- optional subtle visual indication that inference is active;
- no dense diagnostics in normal mode.

### Candidate state

- visible but non-alarming candidate ring/box;
- status may show `CHECKING`;
- no success haptic yet.

### Confirmed state

- very thick high-contrast ring around ball;
- direction cue if the ball is near/off the central field;
- large `BALL FOUND` state;
- one strong haptic event;
- short system sound, respecting device/user constraints where possible;
- do not require reading small confidence text.

Debug-only overlays may show confidence, FPS/inference time, scan region, and normalized coordinates. Keep them behind a compile-time/debug flag or unobtrusive toggle.

---

## Camera specification

Initial target:

- rear wide camera;
- approximately 1280×720 video-data output where supported;
- portrait UI;
- 1× is the broad-search default;
- provide a simple 1×/2× control for experiments;
- avoid expensive conversions such as JPEG/UIImage per frame;
- operate on `CVPixelBuffer`/`CIImage` paths.

The exact camera format can be adjusted after profiling. Record any change and its detection/thermal consequences.

---

## Performance rules

- never queue unbounded inference work;
- if an inference is running, drop/defer intermediate camera frames rather than building backlog;
- prioritize freshest frame;
- keep inference off the main thread;
- keep UI state publication on the main actor/thread;
- use one inference per scheduler step initially;
- collect inference timing from real hardware;
- investigate thermal throttling during sustained scanning.

If the model runs much faster than needed, spend headroom on higher-value small-object coverage, not maximal FPS.

---

## Geometry correctness requirements

Coordinate transforms are critical.

All detections should have a well-defined normalized coordinate convention. The existing code uses normalized full-frame rectangles. Ensure these transformations are unit tested:

- tile-local box -> full-frame box;
- ROI-local box -> full-frame box;
- Vision/Core ML coordinate origin conversions if APIs use bottom-left coordinates;
- camera-buffer aspect ratio -> SwiftUI aspect-fill preview coordinates;
- preview mirroring/orientation behavior.

A correct detector with a misaligned overlay is a product failure.

---

## Logging / reproducibility

For every benchmarkable model, keep a small manifest containing:

- model name/hash
- base architecture
- training code/version
- dataset version/sources
- train/val/test scene IDs
- input size
- export precision (FP16/INT8)
- iOS app commit
- target device/iOS version
- confidence/stabilizer/scheduler parameters
- test results

Do not commit large raw videos or proprietary course footage unless intentionally desired.

---

## Licensing and privacy requirements

- Camera frames remain local in MVP.
- Do not add telemetry or upload code.
- Public dataset/model licenses must be recorded before incorporation.
- Preserve third-party notices.
- Follow the Ultralytics AGPL path and keep the Complete Corresponding Source and selected fine-tuned model publicly
  available for externally distributed builds; follow `docs/PUBLIC_RELEASE_CHECKLIST.md`.
- Keep the Ultralytics adapter replaceable without weakening the current AGPL obligations.
- Do not accidentally commit signing certificates, provisioning profiles, Apple IDs, personal team IDs, or private paths.

---

## Acceptance criteria for MVP

MVP is not complete merely because Xcode builds.

Minimum completion means:

1. app installs on the user's iPhone 16 Pro;
2. launch opens rear camera after permission;
3. a local Core ML golf-ball-capable model loads with no network requirement;
4. full-frame + tiled/ROI scan scheduler runs without frame backlog;
5. a visible golf ball can produce a correctly located candidate overlay;
6. a candidate must pass temporal/spatial confirmation before success feedback;
7. confirmed detection produces conspicuous visual + haptic feedback;
8. multiple negative objects do not trivially trigger confirmed detection in a short controlled test;
9. pure geometry/scheduler/stabilizer tests pass;
10. README contains the exact reproduce/build/install procedure that actually worked on the target machine.

If field-grade precision is not yet reached, ship/report the functioning MVP plus measured failure cases and proceed into the data loop. Do not hide uncertainty.

---

## First commands to attempt on a Mac

```bash
cd GolfBallFinderCodex
chmod +x scripts/*.sh
./scripts/bootstrap_mac.sh

# If the bootstrap completed:
xcodebuild -version
xcodegen generate
open GolfBallFinder.xcodeproj
```

For training after a dataset exists:

```bash
source .venv/bin/activate
python training/train.py \
  --data training/dataset.yaml \
  --base yolo26n.pt \
  --epochs 120 \
  --imgsz 640

python training/evaluate.py \
  --weights runs/golfball/<run-name>/weights/best.pt \
  --data training/dataset.yaml

python training/export_coreml.py \
  --weights runs/golfball/<run-name>/weights/best.pt \
  --output GolfBallFinder/Resources/GolfBall.mlpackage
```

Adjust paths to the actual run directory generated by the installed Ultralytics version.

---

## How to behave as Codex

- Work directly in the repository.
- Inspect existing code before creating replacements.
- Execute tests/builds whenever the environment permits.
- Fix concrete errors instead of only explaining them.
- Keep changes small and reviewable.
- Add tests for non-UI logic when changing geometry/scheduler/stabilizer behavior.
- Do not introduce cloud dependencies.
- Do not add product features outside the stated goal.
- If a current SDK API differs from this prompt, use the installed SDK as source of truth, adapt the adapter, and document the difference.
- If target hardware is unavailable, finish everything that can be validated on simulator/unit tests and give exact physical-device commands and unresolved hardware-only checks.
- Never claim a field accuracy percentage that was not actually measured.
- When blocked, state the exact blocker, evidence, and smallest action needed; continue with all independent work.

### Primary engineering decision rule

When choosing between two implementations, prefer the one that improves **real ball-finding probability per unit latency/thermal cost** on iPhone 16 Pro.

That is the product.

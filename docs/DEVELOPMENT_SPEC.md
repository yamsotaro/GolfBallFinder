# Golf Ball Finder — Development Specification

Version: 0.1 MVP specification  
Target device: iPhone 16 Pro  
Primary goal: reliably help a user visually locate a golf ball in grass or rough by pointing to it in a live camera view, with all inference on-device.

## 1. Product problem

A ball has landed in an approximate area but is difficult to visually distinguish from grass, leaves, mushrooms, stones, highlights, flowers, tees, litter, and other small round/light objects. The app must reduce search time by continuously analyzing the rear-camera image and clearly indicating a stable ball candidate.

The app is not a shot-flight tracker. It starts after the user has walked to the approximate landing area. It does not need cloud AI, maps, accounts, social features, scorekeeping, or a polished settings system.

## 2. Definition of success

The utility metric is not generic COCO mAP. The primary field metric is: **within a deliberate 5–10 second phone scan of the approximate landing area, does the app identify a visible golf ball without repeatedly alerting on non-balls?**

### MVP engineering acceptance

- Installs and runs on a physical iPhone 16 Pro.
- Opens directly into the rear camera after permission is granted.
- Performs inference without a network connection after the model is bundled.
- Shows no cloud login, no account UI, no location request, and no upload path.
- Uses a large indication suitable for reduced visual acuity: ring + arrow + text/state + haptic + sound.
- A single-frame false positive cannot trigger the confirmed-found feedback.
- The app can switch between 1x and 2x camera zoom.
- Unit tests cover crop-coordinate mapping, adaptive scan scheduling, and temporal confirmation.
- Debug observation includes model inference milliseconds and active scan source.
- Field Diagnostics remains separate from the normal finder UI and writes local-only JSONL/evidence for verified
  finds, false positives, and misses.

### Alpha field targets (targets, not guaranteed facts)

Measure these with a held-out set of real iPhone recordings/scenes:

- Ball visible >=50%: >=90% scene-level discovery within 10 seconds.
- Ball visible roughly 25–50%: >=80% scene-level discovery within 10 seconds.
- Confirmed false alert rate: <=1 per 60 seconds of negative-only deliberate scanning.
- p50 model inference on iPhone 16 Pro: <30 ms per invocation.
- Detection indication latency after a repeatable candidate is visible: <300 ms is preferred.
- Ten-minute continuous scan should not crash or become unusably thermally throttled.

These thresholds are tunable after first field data. Do not claim them as achieved until measured.

## 3. Non-goals for MVP

Do not add these unless they directly unblock ball discovery:

- cloud inference;
- user accounts;
- backend/database;
- maps/GPS landing prediction;
- LiDAR-based ball classification;
- ball flight trajectory reconstruction;
- scorecard features;
- App Store commercialization work;
- elaborate settings;
- multi-platform support;
- AR anchors/world tracking.

## 4. Target hardware facts relevant to the design

Apple's iPhone 16 Pro technical specification lists A18 Pro, a 6-core CPU, 6-core GPU, 16-core Neural Engine, 48MP Fusion camera at 24mm, 12MP 2x at 48mm, 5x telephoto, and LiDAR. For this product, the 16-core Neural Engine and camera are directly relevant; LiDAR is intentionally excluded from the MVP because the target is too small/occluded for it to be the primary classifier.

Use the 1x wide/Fusion camera for broad search and expose a simple 2x zoom toggle for close inspection. Do not assume 5x is useful in near-field rough without testing minimum focus distance, stabilization, and field of view.

## 5. Runtime technology choice

### Chosen MVP stack

- Language/UI: Swift 5.10 + SwiftUI.
- Camera: AVFoundation (`AVCaptureSession`, `AVCaptureVideoDataOutput`).
- Image path: `CVPixelBuffer -> CIImage`, without JPEG/UIImage round-trips per frame.
- Detector API: `UltralyticsYOLO` Swift package v8.9.13.
- Runtime: Core ML through the Ultralytics package.
- Compute: `useGpu: true`; on current Ultralytics iOS behavior this requests `.cpuAndNeuralEngine` on iOS 16+ rather than consuming the GPU used by camera/UI composition.
- Base model candidate: YOLO26n for custom training; YOLO11n is the fallback comparison.
- Input size: 640x640.
- Model resource name: `GolfBall.mlpackage` / bundle name `GolfBall`.
- Camera capture preset: 1280x720.
- Processed inference cadence: cap around 20 invocations/s initially.

### Why this stack

Ultralytics has a native Swift/Core ML iOS package with camera and custom model support. Its v8.9.13 release standardizes detection input at 640x640 and includes fixes for Float16 detection tensors. Its published on-device benchmark is on iPhone 17 Pro/A19 Pro, not iPhone 16 Pro, so those numbers are evidence that the design class is real-time capable, not a promised iPhone 16 Pro benchmark. Physical iPhone 16 Pro measurements are mandatory.

Core ML supports CPU/Neural Engine compute selection. Vision/Core ML and AVFoundation are Apple's native low-latency path. The app deliberately does not send frames over a network.

## 6. Small-object strategy

The ball may occupy only a small number of pixels in a 720p camera frame. Full-frame-only detection is therefore insufficient as the sole strategy.

Use a SAHI-inspired sliced-inference concept, but implement the lightweight crop scheduler natively in Swift rather than embedding Python SAHI on iOS.

### Search scheduler

The implementation currently uses five overlapping normalized tiles, each 0.62 x 0.62:

- top-left;
- top-right;
- bottom-left;
- bottom-right;
- center.

While no candidate is present:

```text
processed frame 0: full frame
processed frame 1: tile 0
processed frame 2: full frame
processed frame 3: tile 1
...
```

This keeps one inference invocation per processed frame rather than 5–10 simultaneous inferences.

### ROI lock

On any credible candidate:

1. Expand the detected bounding box around its center by approximately 5x, enforcing a minimum ROI side of 0.34 of the image.
2. For the following frames, repeatedly infer on that ROI.
3. Map the crop-relative box back to full-frame normalized coordinates.
4. Smooth ROI movement to avoid crop jitter.
5. If the ROI loses the candidate for more than three ROI scans, release it and resume broad search.

The intention is to convert one weak small-object hit into several higher-pixel-density confirmations.

## 7. False-positive strategy

The model threshold is intentionally low (`~0.12`) so partially hidden balls are not discarded too early. Precision is recovered outside the model through temporal/spatial confirmation.

### Confirmation policy

Initial defaults:

- stabilizer window: 5 processed inference results;
- required spatially compatible hits: 3;
- minimum per-candidate confidence: 0.14;
- minimum average confidence across clustered hits: 0.20;
- maximum normalized center movement between compatible hits: 0.20;
- confirmed state persistence: 5 processed frames.

These are starting values only. Tune them from recorded false positives and misses.

### Hard negatives required in training

Include negative images containing, at minimum:

- mushrooms;
- white/yellow flowers;
- pale leaves;
- dry leaves with circular highlights;
- white stones/pebbles;
- tees and ball markers;
- litter/paper/plastic;
- dew/water highlights;
- sand edges;
- white shoes/clothing entering frame;
- bright specular highlights;
- dandelions;
- small round seed heads.

The correct training label is only `golf_ball`. Hard-negative images should contain no label boxes.

## 8. Model/data strategy

### Bootstrap only

A public Hugging Face model `notjulietxd/golf-ball-tracker` is provided as a bootstrap checkpoint. Its model card reports YOLOv8-nano, 559 real + 500 synthetic images, 640 input, mAP50 81.2%, and Apache-2.0. Its own limitations say it uses mixed sports-ball data and needs real golf-ball data. Use it only to validate the full camera->Core ML->overlay pipeline quickly.

The bootstrap script pins the published SHA256 of `best.pt` to reduce accidental model drift.

### Target model

Train a new one-class detector on:

1. permissibly licensed/public golf-ball data that passes manual quality inspection;
2. the user's own iPhone 16 Pro rough/grass recordings;
3. negative-only scenes;
4. hard negatives collected from real failures.

Prefer YOLO26n first. Also train/benchmark YOLO11n if export/runtime compatibility or accuracy is materially better. Choose based on held-out field data and iPhone latency, not model generation number.

### Dataset leakage rule

Never randomly split adjacent frames from the same video across train/validation/test. Entire recording sessions, locations, or deliberate scene groups must be assigned to one split. Otherwise nearly identical frames make validation misleadingly optimistic.

## 9. Data collection protocol

Record many short clips rather than one long clip. Vary:

- rough depth: fairway-like, short rough, deep rough;
- ball visibility: 100%, 75%, 50%, 25%, arc/fragment only;
- distance: approximately 0.5–1m, 1–2m, 2–4m, >4m;
- lighting: overcast, hard sun, shade, backlight, late afternoon;
- grass: dry/wet, green/brown, mixed weeds;
- ball: clean/dirty, white/yellow where relevant, different dimple patterns;
- camera motion: slow sweep, walking, slight motion blur;
- lens: primarily 1x, some 2x;
- negative scenes: same environments with no ball.

Start around 1,000–3,000 diverse labeled positive/negative frames for the first serious model, then let field failures determine expansion. Diversity matters more than raw adjacent-frame count.

## 10. UI/UX specification

### Main screen

Full-screen camera. No landing page after permission has already been granted.

Top status capsule:

- `AI読み込み中`
- `探索中`
- `候補を確認中`
- `ボール発見`
- actionable error text if model/camera fails.

Bottom controls:

- 1x/2x toggle;
- `再探索` reset.

### Detection indication

Candidate:

- large yellow ring centered on the suspected ball;
- no sound/haptic confirmation.

Confirmed:

- large green ring;
- arrow icon above the target;
- `ボール発見` state;
- success haptic;
- short system beep.

Do not rely on color alone; text, shape, arrow, haptics, and sound provide redundant cues.

### Accessibility

Use large geometry rather than a tiny detector bounding box. Keep controls high contrast and simple. Dynamic Type support is desirable, but the camera marker must remain visually dominant.

## 11. Privacy and security

- No cloud inference.
- No frame upload.
- No account.
- No location request.
- No microphone request.
- No photo-library permission for MVP.
- The model is bundled into the app for runtime offline use.
- Network is only used during development to fetch dependencies/model files.
- If a future debug failure-capture feature is added, make storage local and opt-in.

## 12. Code architecture

```text
GolfBallFinderApp
  -> ContentView
       -> CameraPreview (AVCaptureVideoPreviewLayer)
       -> DetectionOverlay
       -> CameraController
            -> AVCaptureSession / VideoDataOutput
            -> ScanScheduler
            -> GolfBallDetector
                 -> UltralyticsYOLO / Core ML
            -> DetectionStabilizer
            -> FeedbackManager
```

Key boundaries:

- `GolfBallDetector` owns all Ultralytics-specific code. This is the future replacement seam.
- `ScanScheduler` knows no ML framework.
- `DetectionStabilizer` is deterministic/pure enough to unit test.
- UI consumes only `FinderState` and normalized coordinates.

## 13. Current repository implementation

The starter implementation already includes:

- 720p portrait AVFoundation capture;
- direct `CVPixelBuffer -> CIImage` inference;
- model loading from `GolfBall.mlpackage`;
- low detector thresholds;
- full-frame/overlapping-tile alternation;
- ROI candidate lock;
- crop-to-full coordinate mapping;
- temporal confirmation;
- custom large overlay;
- haptic + system sound;
- 1x/2x zoom;
- reset search;
- unit tests for critical pure logic;
- latest-frame-only admission with busy/cadence drop counters and thermal-adaptive target FPS;
- camera interruption, permission denial/recovery, foreground/background, portrait rotation, and 1x/2x handling;
- optional Field Diagnostics with bounded logging and opt-in local JPEG evidence;
- Python train/evaluate/export scripts;
- SHA256-pinned public seed model fetch;
- XcodeGen project generation.

## 14. Performance work order

Do not prematurely micro-optimize. Measure in this order on iPhone 16 Pro:

1. model load time;
2. inference p50/p95 in live camera;
3. effective processed FPS;
4. thermal behavior over 10 minutes;
5. full-frame versus tile latency;
6. 1x versus 2x detection success;
7. FP16 versus int8 export;
8. YOLO26n versus YOLO11n.

Ultralytics' current iOS performance notes show `.cpuAndNeuralEngine` is preferred in its camera pipeline and that camera preprocessing can dominate if capture resolution is unnecessarily high. Keep 720p until field evidence justifies higher resolution.

If the ball remains too small, test in this order:

- smaller/stronger overlapping tiles;
- more time in tile mode relative to full mode;
- 2x lens/zoom workflow;
- higher camera capture resolution while keeping model input 640;
- training with sliced crops;
- a second-stage ball/not-ball classifier on the detected patch.

## 15. Optional second-stage verifier

Do not add before measuring false positives. If temporal confirmation plus hard-negative training is insufficient, add a tiny crop classifier:

```text
YOLO candidate -> crop 96/128/160 px -> golf_ball vs not_golf_ball classifier -> confirm
```

The verifier should be trained primarily on detector false positives. It must not become a required cloud call.

## 16. Vision tracking

Apple provides `VNTrackObjectRequest` for tracking a previously identified bounding box across video frames. This is a valid optimization/visual-stability option, but the current MVP uses repeated ROI detector inference because it simultaneously tracks and re-classifies the object. Add Vision tracking only if it measurably improves overlay smoothness or allows detector frame skipping without sacrificing confirmation reliability.

## 17. LiDAR position

LiDAR exists on iPhone 16 Pro but is not part of initial detection. A golf ball is very small and can be behind grass, making LiDAR unreliable as the primary recognizer. Future experiments may use depth only for plausibility checks such as rejecting impossible apparent sizes/distances.

## 18. Licensing boundary

Ultralytics iOS is AGPL-3.0. The current app is explicitly a personal prototype. Before App Store/commercial/closed-source distribution, make a licensing decision:

- comply with AGPL obligations;
- purchase an appropriate Ultralytics license;
- or replace `GolfBallDetector` with another Core ML model/runtime path.

SAHI is MIT, but only its inference concept is reimplemented; the Python package is not embedded in the iOS app. Dataset licenses must be recorded per source.

## 19. Deployment to personal iPhone

Xcode can install to a personal device with an Apple Account. Apple's current Personal Team documentation says free provisioning profiles expire after seven days, requiring rebuild/reinstall. For routine use without that limitation, Apple Developer Program enrollment is more convenient. The app does not need to be publicly listed in the App Store to be developed/tested on the user's device.

## 20. Definition of done for first useful personal build

The first useful build is complete when:

1. `./scripts/bootstrap_mac.sh` completes on the development Mac.
2. `xcodegen generate` creates the project.
3. `GolfBallFinder` builds with no compiler errors.
4. The app is signed and launched on iPhone 16 Pro.
5. Airplane mode does not stop detection after launch/model load.
6. An obvious golf ball on short grass is detected and confirmed.
7. A single-frame false candidate does not trigger haptic/sound.
8. 1x/2x switch works.
9. Unit tests pass.
10. A first field-test log records failures in at least 20 positive scenes and 10 negative-only scan scenes.
11. Those failures become the next hard-negative/positive training batch.

## 21. Field diagnostics data boundary

The main UI remains camera/status/zoom/reset plus one diagnostics entry button. Diagnostics samples are persisted at
no more than 2 Hz even though live values update after every completed inference. Manual feedback stores exactly the
latest retained frame; it never creates a pending camera/inference queue. Files remain in the app's Documents
directory and are never uploaded. Each record carries session/scene, app build, model SHA when bundled, timestamp,
thermal state, confidence, scan mode, normalized bbox, latency/FPS, confirmation timing, zoom, and frame-drop counts.

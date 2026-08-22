# Golf Ball Finder — iPhone 16 Pro on-device AI prototype

> **Windows-only workstation is supported.** You do not need to own a physical Mac. Use `scripts/bootstrap_windows.ps1` for local ML/data work and `codemagic.yaml` for hosted macOS/Xcode compilation, signing, and TestFlight delivery. See `docs/WINDOWS_CLOUD_BUILD.md`.

A minimal iOS app for finding a golf ball in grass/rough with the rear camera. The camera stream is analyzed locally with Core ML. The MVP uses Ultralytics' native iOS/Core ML package and adds an application-specific search pipeline: full-frame scan, overlapping sliced scan, ROI lock, temporal confirmation, large visual indication, haptics, and sound.

## Recommended path from Windows (no physical Mac)

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\bootstrap_windows.ps1
```

Then push the repository to GitHub and use the included Codemagic workflows. `ios-compile-check` runs the Swift
unit tests and an unsigned Xcode compile. `ios-model-compile-check` additionally downloads the SHA256-pinned seed,
exports `GolfBall.mlpackage`, runs tests/build, and fails unless `GolfBall.mlmodelc` is present in the Simulator app.
It also inspects the generated `project.pbxproj` to require the model in the application target's Sources phase and
requires `ModelManifest.json` in the application target's Resources phase, then requires a `coremlcompiler`/Core ML
compilation operation in a clean Xcode build log before checking both resources in the bundle.
The lightweight model-free workflow uses `project.compile-check.yml`; the full `project.yml` deliberately requires
the exported model and manifest. Neither workflow needs Apple signing. `ios-testflight` repeats the model/export/test
path, assigns a unique build number, signs the IPA, and uploads it to App Store Connect. Full setup:
`docs/WINDOWS_CLOUD_BUILD.md`.

The final app still runs AI inference locally on the iPhone; the cloud Mac is only a build/signing machine.

## Alternative: local Mac path

Prerequisites: Xcode, an Apple Account signed into Xcode, Homebrew, Python 3, and an iPhone 16 Pro for real performance testing.

```bash
cd GolfBallFinderCodex
./scripts/bootstrap_mac.sh
```

The bootstrap script:

1. installs XcodeGen if needed;
2. creates `.venv` and installs ML tooling;
3. downloads a SHA256-pinned public golf-ball seed checkpoint from Hugging Face (third-party PyTorch serialization; review the security/license note before use);
4. exports it to `GolfBallFinder/Resources/GolfBall.mlpackage`;
5. generates `GolfBallFinder.xcodeproj`.

Both bootstrap scripts also run the Python utility/configuration tests:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s Tests/Python -v
```

Then open the project, choose your Signing Team, select the connected iPhone, and Run.

> The seed checkpoint is only an end-to-end bootstrap model. It was not trained specifically for deep rough. Product accuracy depends on fine-tuning with iPhone footage from rough/grass plus hard negatives.

## Repository map

- `GolfBallFinder/` — SwiftUI + AVFoundation + local YOLO/Core ML inference.
- `training/` — training, evaluation, frame extraction, Core ML export.
- `scripts/` — Windows/Mac bootstrap, configuration, and seed-model retrieval.
- `Tests/` — deterministic geometry, scan scheduler, temporal confirmation tests.
- `Tests/Python/` — Windows-runnable utility, YAML/CI, and release-configuration tests.
- `docs/DEVELOPMENT_SPEC.md` — product/technical specification.
- `docs/CODEX_PROMPT.md` — a self-contained prompt to give Codex.
- `docs/DATASET_PLAN.md` — capture/annotation/hard-negative plan.
- `docs/RESEARCH_NOTES.md` — researched technologies, OSS, datasets, licensing.
- `docs/TEST_PLAN.md` — simulator/unit tests and physical-device field tests.
- `docs/VALIDATION_REPORT.md` — checks already completed here vs. Apple-toolchain/device gates still required.
- `docs/AUDIT_REPORT.md` — current Windows audit findings, fixes, validation evidence, and remaining gates.
- `docs/WINDOWS_CLOUD_BUILD.md` — Windows + hosted macOS + TestFlight deployment with no physical Mac.
- `FIELD_TEST_PLAN.md` — directly executable iPhone field protocol and KPI definitions.
- `NEXT_HUMAN_STEPS.md` — private Apple/Codemagic/TestFlight actions after enrollment approval.

## Current MVP architecture

```text
AVCaptureSession 720p portrait
        |
        v
ScanScheduler
  |-- full frame
  |-- overlapping tile (alternating)
  `-- ROI lock after candidate
        |
        v
UltralyticsYOLO -> Core ML -> CPU + Neural Engine preferred
        |
        v
low-threshold candidates
        |
        v
DetectionStabilizer (3 hits / 5 processed frames)
        |
        +--> candidate: yellow ring
        `--> confirmed: green ring + arrow + haptic + beep
```

Only one model invocation is scheduled per processed frame. While searching, full-frame and sliced crops alternate. A candidate switches the next frames to an enlarged ROI so the same suspected ball is repeatedly inspected at higher effective pixel density.

## Build without the bootstrap script

```bash
brew install xcodegen
python3 -m venv .venv
source .venv/bin/activate
pip install -r training/requirements.txt
python scripts/fetch_seed_model.py
python training/export_coreml.py --weights training/models/seed_golf_ball_yolov8n.pt
xcodegen generate
open GolfBallFinder.xcodeproj
```

The Core ML export also writes `GolfBallFinder/Resources/ModelManifest.json` with the source checkpoint hash,
tool versions, input size, and precision. The app reads its checkpoint SHA256 into each field-diagnostics record, so
the model workflow treats the manifest as a required top-level bundle resource. Keep it with field results so model
comparisons remain reproducible.

## Train the actual model

1. Record many short iPhone videos of golf balls in representative grass/rough and separate negative-only scenes.
2. Extract spaced frames while keeping recording sessions separate:

```bash
source .venv/bin/activate
python training/extract_frames.py ~/Videos/rough_session_*.MOV --interval 0.75
```

3. Annotate one class only: `golf_ball`. Keep hard-negative images with no boxes.
4. Split train/val/test by recording session, never by adjacent frames.
5. Copy `training/dataset.yaml.example` to `training/dataset.yaml` and edit the path.
6. Train:

```bash
python training/train.py \
  --data training/datasets/golf_ball_v1/dataset.yaml \
  --manifest training/datasets/golf_ball_v1/dataset_manifest.csv \
  --base yolo26n.pt --epochs 120 --imgsz 640
```

7. Evaluate:

```bash
python training/evaluate.py \
  --weights runs/golfball/golf_ball_yolo26n/weights/best.pt \
  --data training/datasets/golf_ball_v1/dataset.yaml \
  --manifest training/datasets/golf_ball_v1/dataset_manifest.csv \
  --split test --output runs/golfball/golf_ball_yolo26n/test_metrics.json
```

8. Export and replace the bundled model:

```bash
python training/export_coreml.py \
  --weights runs/golfball/golf_ball_yolo26n/weights/best.pt
xcodegen generate
```

Re-run on the iPhone and collect false positives/false negatives for the next iteration.

The Field Diagnostics sheet records inference/thermal/scan/bbox timing as JSONL. A human can mark a correct find,
`これはボールではない`, or a miss; the app stores the latest frame and metadata locally under Files > On My
iPhone > Golf Ball Finder > FieldDiagnostics. Nothing is uploaded automatically.

For a reproducible session-level dataset:

```bash
cp training/dataset_manifest.example.csv training/dataset_manifest.csv
# Edit source_dir/split/content_type, keeping every recording session in exactly one split.
python training/prepare_dataset.py \
  --manifest training/dataset_manifest.csv \
  --output training/datasets/golf_ball_v1
python training/validate_dataset.py \
  --data training/datasets/golf_ball_v1/dataset.yaml \
  --manifest training/datasets/golf_ball_v1/dataset_manifest.csv
```

`training/summarize_field_logs.py` converts completed positive/negative JSONL scenes into field-result JSON.
`training/compare_models.py` compares those results only when protocol, frozen manifest hash, and iPhone device
match. Its ranking starts with 10-second discovery, then false confirmed alerts/min, then latency.

Use `docs/FIELD_TEST_LOG_TEMPLATE.csv` for the first repeatable device/field pass. Do not fill target values into
the result columns until they have actually been measured on the iPhone 16 Pro.

## Important license note

The MVP depends on Ultralytics' AGPL-3.0 iOS SDK. That is acceptable for this personal prototype workflow, but closed-source/commercial distribution requires a separate licensing decision. See `THIRD_PARTY_NOTICES.md`. The design intentionally isolates the detector behind `GolfBallDetector.swift` so the runtime can later be replaced.

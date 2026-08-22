# Repository audit report

Audit date: 2026-08-22  
Host: Windows, Python 3.14.6, no Swift/Xcode toolchain, no physical Mac  
Scope: repository source, Windows bootstrap, Python/model utilities, Swift static checks, XcodeGen/Codemagic/TestFlight configuration

## Source-of-truth review

Read in full before changing implementation:

- `AGENTS.md`;
- the root Windows one-shot prompt (missing initially; restored as a non-duplicating entry point);
- `README.md`, every file under `docs/`, `project.yml`, and `codemagic.yaml`;
- every file under `training/` and `scripts/`;
- all application and Swift unit-test sources.

The supplied workspace initially had no `.git` directory. Consequently, commit history, prior diffs, remote URL,
branch protection, and the ability to start a connected Codemagic build could not be audited from this copy.

## Findings and changes

| Priority | Finding | Resolution in this audit |
|---|---|---|
| Critical | No App Icon asset was configured, creating an App Store Connect/TestFlight validation risk. | Added an opaque 1024x1024 `AppIcon` asset catalog and enabled it in `project.yml`. |
| Critical | Every signed build used `CURRENT_PROJECT_VERSION: 1`, so subsequent uploads could be rejected as duplicate builds. | `scripts/configure_bundle_id.py` now validates/injects Codemagic `BUILD_NUMBER`; the signed workflow supplies it. |
| High | `ios-compile-check` compiled but did not run the critical Swift unit tests. | Both Codemagic workflows now select an available iPhone simulator and run `xcodebuild test`. |
| High | macOS Core ML dependencies were broad and could resolve to a PyTorch version beyond coremltools 9.0's supported export line. | Pinned the Mac exporter to coremltools 9.0, PyTorch 2.7.0/torchvision 0.22.0, NumPy 1.26.4, and compatible direct dependencies. Apple Silicon/Python 3.11 resolution passed a pip dry run. |
| High | `GolfBallDetector` declared unchecked sendability while model readiness/error state was accessed from multiple queues without synchronization. | Added locked model state; inference can only receive a model after successful load. |
| High | Detections below the stabilizer candidate threshold could still consume the ROI lock budget. | ROI locking now requires `candidateMinConfidence`, while the detector's low threshold remains intact for configured candidate processing. |
| High | Frame extraction silently reused a folder based only on video stem, permitting session mixing/overwrites. Invalid intervals and failed JPEG writes were not fatal. | Added path-hashed session IDs, positive finite interval validation, explicit overwrite behavior, write checks, nonzero failure status, and `session.json` metadata. |
| Medium | Windows bootstrap stated Python 3.10+ but did not enforce it or run utility tests. | Added version enforcement, project-local caches, pinned direct dependencies, and Python regression tests. |
| Medium | Core ML export did not write the model/checkpoint provenance required by the spec. | Export now writes `ModelManifest.json` with checkpoint SHA256, class names, input size, precision, tool versions, source revision, and timestamp. |
| Medium | The project had no maintained Windows-runnable tests for configuration/data scripts. | Added 13 `unittest` cases covering release configuration, YAML/workflows, App Icon/privacy/field-log structure, frame-session safety, interval/empty-video validation, and hashing. |
| Medium | Initial field results had no checked-in schema. | Added `docs/FIELD_TEST_LOG_TEMPLATE.csv`; result cells remain deliberately unclaimed. |
| Low | TestFlight export-compliance prompts were not explicitly described by the app plist. | Declared `ITSAppUsesNonExemptEncryption: false`; the app contains no non-exempt encryption feature. |
| Low | The app's no-upload/no-tracking behavior was specified only in prose. | Added an app privacy manifest declaring no tracking, collected-data types, tracking domains, or required-reason API categories. |
| High | The unsigned cloud gate did not exercise seed fetch, Core ML export, or verify the compiled model resource. | Added `ios-model-compile-check`, including pinned SHA verification, export, XCTest/build, and `.mlmodelc`/manifest bundle assertions. |
| High | A target-level `resources` key unsupported by XcodeGen left the generated model and ordinary bundle resources unregistered. Xcode could still compile the Swift target without producing a bundled model or manifest. | Registered `GolfBall.mlpackage` individually as `sources`, registered the diagnostics manifest/privacy manifest individually with `buildPhase: resources`, required XcodeGen 2.38.0+, and added generated-project phase validation plus build-product enumeration. |
| High | No field diagnostics or recoverable failure evidence existed. | Added bounded JSONL metrics, scene timing, local-only latest-frame evidence, and separate Field Diagnostics UI for verified finds, false positives, and misses. |
| High | Camera lifecycle and thermal load were incomplete. | Added interruption/runtime/background/foreground observers, permission recovery, portrait rotation, latest-frame-only admission/drop counters, and 20/15/8/4 FPS thermal policy. |
| High | Training split and comparison rules were prose-only. | Added manifest-driven preparation, dataset/label/leakage/duplicate/Hard Negative validation, provenance hashes, and field-KPI model comparison. |

## Validation evidence

Passed on Windows after the changes:

- `scripts/bootstrap_windows.ps1`: virtual environment, pinned dependency install, seed download, SHA256 verification,
  and Python tests;
- Python `unittest`: 29/29 passing, including session split/Hard Negative validation, field-log KPI summary,
  comparable model ranking, XcodeGen model-source configuration, and generated-pbxproj phase validation;
- Python `compileall` plus `--help` smoke checks for all CLIs;
- PowerShell parser for `bootstrap_windows.ps1` and Bash syntax check for `bootstrap_mac.sh`;
- Bash syntax checks for all 23 Codemagic script blocks;
- YAML parse for XcodeGen, Codemagic, and dataset examples; JSON parse for asset catalogs;
- seed checkpoint load: task `detect`, only class `golf_ball`, pinned SHA256
  `45e8f8bd8975dc7f437919a11c3f6ee1fe7c8ae40b0f49910d0677d1c0326791`;
- synthetic OpenCV video extraction: 10 decoded frames at 10 FPS and 0.2-second interval produced five frames plus
  `session.json`;
- Swift tree-sitter syntax parse: all 20 app/test `.swift` files passed;
- macOS arm64 / CPython 3.11 pip dry-run: the pinned Core ML export dependency set resolved.

The tree-sitter result is syntax-only. It does not claim Swift type-checking, iOS SDK compatibility, or linking.

## Remaining gates and risks

1. **Cloud Apple-toolchain gate:** the owner reports `ios-compile-check`, XcodeGen, package resolution, Simulator
   XCTest/build, and the `ios-model-compile-check` Core ML export/compilation all succeeded. Re-run
   `ios-model-compile-check` for the manifest Resources-phase correction; its previous run stopped only at the
   missing manifest bundle assertion.
2. **Core ML resource gate:** the reported run proves `GolfBall.mlmodelc` compilation and bundling. The corrected
   workflow must still prove `ModelManifest.json` is copied to the app bundle.
3. **Signing/App Store Connect:** supply the private Codemagic integration and `BUNDLE_ID` externally. No key,
   certificate, provisioning profile, team ID, or `.p8` belongs in Git.
4. **Physical iPhone gate:** camera permission, preview orientation/aspect fill, overlay alignment, 1x/2x behavior,
   offline model load, feedback transition, latency, thermal behavior, and battery remain device-only.
5. **Accuracy gate:** no representative field dataset or measured field results are present. The seed model is only
   a pipeline bootstrap. Hard negatives, held-out scene splits, real iPhone footage, and the field log must drive the
   next model iteration. No mAP or discovery target is claimed as achieved.
6. **Repository transport:** restore or initialize the intended Git history, configure a private remote, and connect
   that remote to Codemagic before a cloud workflow can be triggered from this workspace.

## Next implementation order

1. Push the current revision to the intended private Git remote and run unsigned `ios-model-compile-check`.
2. Fix any exact Core ML/Xcode/Ultralytics API or concurrency diagnostics from that run; do not proceed on a red gate.
3. After Apple approval, configure the external Apple/Codemagic values and run `ios-testflight`; keep automatic TestFlight submission off
   for the first signing diagnosis, then enable it if desired after the IPA upload is confirmed.
4. Install the processed internal build on iPhone 16 Pro and execute `FIELD_TEST_PLAN.md`.
5. Record at least 30 positive and 10 one-minute negative-only scenes, then train/fine-tune against actual misses and hard
   negatives before tuning thresholds or adding a second-stage verifier.

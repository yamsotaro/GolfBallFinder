# Validation report for generated starter repository

Generated/checked: 2026-08-22.

## Checks completed in the generation environment

- Python source syntax: `python3 -m py_compile training/*.py scripts/*.py` — PASS.
- Shell syntax: `bash -n scripts/*.sh` — PASS.
- `project.yml` YAML parse — PASS.
- `training/dataset.yaml.example` YAML parse — PASS.
- Every Swift source/test file was parsed with `swiftc -frontend -parse` (Swift 6.2.1 parser) — PASS.
- CLI examples were cross-checked so training uses `--base`, evaluation/export use `--weights`.
- Ultralytics iOS API shape used by `GolfBallDetector` was checked against the pinned v8.9.13 source (`YOLO` async load, `isLoaded`, `setThresholds`, `CIImage` call, `YOLOResult.boxes`, `Box.xywhn`, `Box.conf`, `YOLOResult.inferenceMs`).

## Apple-toolchain checks reported complete before the current feature revision

The project owner reports that Codemagic `ios-compile-check`, XcodeGen, Swift Package resolution, iPhone Simulator
unit tests, and iOS Simulator compilation succeeded. This closes the original starter/audit compile gate for that
exact uploaded revision.

## Checks still required after the current feature revision

The generation environment is not macOS and does not have the iOS SDK, Xcode signing system, Core ML compilation toolchain, or the user's iPhone 16 Pro. Therefore these remain required gates on the user's Mac/Codex environment:

1. Re-run unsigned `ios-model-compile-check`. The latest reported run completed seed export, XCTest, Simulator
   build, Core ML compilation, and the explicit `GolfBall.mlmodelc` bundle assertion. It stopped only because
   `ModelManifest.json` was absent: the target-level `resources` key had not registered it with XcodeGen. The revised
   gate now verifies the model Sources phase, manifest/privacy Resources phase, Core ML compiler operation, all
   `.mlmodelc` products, all bundled JSON paths, and both required bundle results.
2. Re-run `ios-compile-check` for the diagnostics/camera/training revision; Windows Swift parsing is syntax-only.
3. After Apple approval, run signing/upload and install on iPhone 16 Pro.
4. Check camera orientation/overlay alignment and lifecycle on hardware.
5. Measure real-device latency, thermal, battery, discovery, and false alerts.

Codex should treat any failure at these gates as an implementation bug/toolchain compatibility task and fix the repository rather than merely documenting it.

## Windows audit refresh — 2026-08-22

Completed in the current Windows workspace:

- `scripts/bootstrap_windows.ps1` created `.venv`, installed the pinned direct dependencies, downloaded the seed
  checkpoint, and verified its pinned SHA256.
- The seed checkpoint loaded through Ultralytics as a one-class `golf_ball` detection model.
- Python source compilation and YAML parsing passed before the audit changes; the maintained Python test suite is
  now the regression gate for utility/configuration behavior.
- App Store build-number injection, App Icon asset configuration, simulator-test CI steps, safe frame session IDs,
  and export manifests were added during the audit.
- After the Core ML target-registration fix, Windows bootstrap passed again, including seed SHA verification and
  33 Python tests. Swift tree-sitter syntax parsing passed for 22 Swift app/test files, and all 23 Codemagic script
  blocks passed Bash syntax checking.

## Color Assist Windows refresh — 2026-08-23

The default-off Color Assist revision adds Core Image transforms, tile-cycle ordering, diagnostics fields/debug
previews, and the Python reference evaluator. Windows tests cover OFF baseline preservation, offline white-ball
saliency rank, mixed-arm rejection, configuration invariants, and Python/Swift syntax. Core Image runtime kernels,
Simulator XCTest/type-checking, iPhone processing latency, thermal impact, and field KPI changes still require the
unsigned Codemagic workflows and then the physical iPhone 16 Pro A/B protocol.

The earlier Swift parser result applies to the generated starter revision, not automatically to later edits.
Because Swift/Xcode is unavailable on this Windows host, every later Swift edit still requires the relevant
Codemagic unsigned workflow before that edit may be described as compiling. Physical-device and field claims remain
unverified until `FIELD_TEST_PLAN.md` is executed.

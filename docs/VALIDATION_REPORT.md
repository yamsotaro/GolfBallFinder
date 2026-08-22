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

## Checks not possible in this environment

The generation environment is not macOS and does not have the iOS SDK, Xcode signing system, Core ML compilation toolchain, or the user's iPhone 16 Pro. Therefore these remain required gates on the user's Mac/Codex environment:

1. `xcodegen generate` using the installed XcodeGen version.
2. Swift type-check/link against the actual iOS SDK and Ultralytics v8.9.13 package.
3. Python dependency installation on the selected macOS/Python version.
4. Third-party seed checkpoint download/hash verification and Core ML export.
5. Simulator build/test.
6. Development signing and install on iPhone 16 Pro.
7. Camera orientation/overlay alignment check on hardware.
8. Real-device latency, thermal, battery, and detection-quality measurements.

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

The earlier Swift parser result applies to the generated starter revision, not automatically to later edits.
Because Swift/Xcode is unavailable on this Windows host, the updated Swift sources still require the
`ios-compile-check` Codemagic workflow before they may be described as compiling. Physical-device and field claims
remain unverified until the iPhone 16 Pro test matrix is executed.

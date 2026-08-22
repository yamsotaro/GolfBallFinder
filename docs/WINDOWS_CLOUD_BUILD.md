# Windows-only workstation + cloud macOS + iPhone

This is the recommended workflow when you do not own or use a physical Mac.

## What is and is not possible

You do **not** need a physical Mac. You do need Apple's macOS/Xcode toolchain somewhere for an iOS build. In this repository that role is delegated to Codemagic's hosted Mac runner.

Daily workflow:

```text
Windows + Codex + Python + Git
        |
        v
GitHub repository
        |
        v
Codemagic hosted macOS/Xcode
  - Core ML export
  - XcodeGen
  - Swift Package resolution
  - code signing
  - IPA build
        |
        v
App Store Connect / TestFlight
        |
        v
iPhone 16 Pro
```

## Recommended distribution path: TestFlight

For a device you use yourself, TestFlight avoids USB deployment and avoids owning a Mac. It does require Apple Developer Program membership.

A TestFlight build can be installed on the iPhone using Apple's TestFlight app. The application still performs inference locally on the iPhone; TestFlight is only the delivery mechanism.

## One-time Apple setup

1. Enroll the Apple Account used for this project in Apple Developer Program.
2. In Certificates, Identifiers & Profiles, create an explicit App ID / Bundle ID, for example:
   `com.<yourname>.golfballfinder`
3. In App Store Connect, create a new iOS app record using exactly that Bundle ID.
4. In App Store Connect > Users and Access > Integrations, create an App Store Connect API key with the
   `App Manager` role required by Codemagic publishing. Download the `.p8` file once and keep it secure.
5. Install TestFlight on the iPhone 16 Pro.

Do not commit the `.p8` key or any Apple credential to Git.

## One-time Codemagic setup

1. Push this repository to a private GitHub repository.
2. Create a Codemagic account and add the GitHub repository.
3. In Codemagic Team settings > Developer Portal, add the App Store Connect API key. Give the integration the exact reference name:

   `golfballfinder-appstore`

4. Let Codemagic create/manage an Apple Distribution certificate if prompted.
5. In Codemagic environment variables create a group named:

   `golfballfinder_config`

6. Add:

   `BUNDLE_ID = com.<yourname>.golfballfinder`

   Use the exact Bundle ID registered with Apple.

## First cloud run

Run workflow `ios-compile-check` first. It needs no signing, runs the pure Swift unit tests on an available iPhone
simulator, and proves that the project and package dependencies compile on a real Apple toolchain.

Then run `ios-model-compile-check`, which is also unsigned. It creates a pinned Python 3.11 environment, downloads
the public seed `.pt`, verifies its fixed SHA256, exports `GolfBall.mlpackage`, runs simulator tests/build, and checks
that Xcode compiled `GolfBall.mlmodelc` plus `ModelManifest.json` into `GolfBallFinder.app`. Run this while Apple
Developer approval is pending; it isolates model/export/resource errors from later signing errors.

The model must be declared as an individual target `source` in `project.yml`. XcodeGen 2.38.0 or newer recognizes
`.mlpackage` as a source type; the spec explicitly sets `buildPhase: sources` so it is placed in the application
target's `PBXSourcesBuildPhase`, allowing Xcode's standard Core ML build pipeline to produce `.mlmodelc`. Declaring
files under a target-level `resources` key is not part of XcodeGen's target schema and does not register them. Bundle
resources must also be target `sources` entries, with `buildPhase: resources`. The workflow therefore validates the
generated model `PBXFileReference`/`PBXBuildFile`/Sources phase and the manifest/privacy
`PBXFileReference`/`PBXBuildFile`/Resources phase, uses separate DerivedData for XCTest and the final clean build,
requires a Core ML compiler operation in the final build log, and lists every generated `*.mlmodelc` directory and
every bundled JSON file before checking `GolfBall.mlmodelc` and top-level `ModelManifest.json` in the app.

`project.yml` deliberately requires both the exported model and `ModelManifest.json`. The model-free
`ios-compile-check` generates from `project.compile-check.yml`, an overlay that removes those two generated inputs
while retaining the application sources and privacy manifest. Do not use that overlay for model or TestFlight builds.
The manifest is not needed to execute Core ML inference, but Field Diagnostics loads it from the bundle root and
adds `checkpoint_sha256` to records. It is therefore required for traceable field evidence and model comparisons.

After Apple Developer approval and private integration setup, run `ios-testflight`.

The signed workflow:

1. applies the Bundle ID and Codemagic's monotonically increasing build number;
2. creates the Core ML seed model in the Mac runner if `GolfBall.mlpackage` is not already in the repository;
3. records the checkpoint hash/export tool versions in `ModelManifest.json`;
4. generates the Xcode project and runs unit tests;
5. resolves UltralyticsYOLO;
6. fetches/creates App Store signing files;
7. creates a signed `.ipa` with the required App Icon.

For the very first build, `codemagic.yaml` intentionally has automatic TestFlight submission disabled (`submit_to_testflight: false`). This makes first-time App Store Connect configuration easier to diagnose. Once the IPA upload path is confirmed, set:

```yaml
submit_to_testflight: true
```

or upload the generated signed build using the configured Codemagic publishing integration.

`auth: integration` uploads the IPA to App Store Connect. `submit_to_testflight` controls the optional
post-processing submission action; internal testers can use the uploaded build after App Store Connect processing
and internal-group assignment. Enable the flag after the first upload path is confirmed if you want that action
automated.

## Internal TestFlight use

If you are the only tester, add your Apple Account as an internal tester in App Store Connect where applicable. External beta testing can trigger beta review; internal TestFlight is preferable for this project during development.

TestFlight builds remain available for a limited beta period, so keep producing newer builds as the model/application changes.

## Windows setup

PowerShell:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\bootstrap_windows.ps1
```

This creates `.venv`, installs Windows-compatible YOLO/data tooling, and downloads the seed `.pt` checkpoint. It intentionally does **not** install `coremltools`; Core ML export occurs on hosted macOS.

Train normally on Windows (NVIDIA CUDA is ideal if available), then commit/push the trained `best.pt` only if your repository/security policy allows large model artifacts. Prefer Git LFS for large model files. Another option is to keep model artifacts in release/object storage and have CI download a SHA256-pinned artifact.

## After a successful TestFlight build

On the iPhone:

1. Open TestFlight.
2. Install Golf Ball Finder.
3. Grant Camera permission on first launch.
4. Test outdoors with real grass/rough.
5. Record field results: found/not found, occlusion level, false alert, approximate range, 1x/2x, lighting, thermal behavior.

Use `docs/FIELD_TEST_LOG_TEMPLATE.csv` and copy the checkpoint SHA256 from the generated model manifest.

The AI inference remains on-device. No cloud inference is required while searching for a ball.

## Important limitation of a no-physical-Mac workflow

You lose Xcode's live attached debugger, Instruments, and direct device console while walking around with the camera. Compensate by keeping the app diagnostically observable:

- optional in-app diagnostics overlay for inference ms / processed FPS / thermal state;
- structured local session logs;
- export/share a small JSON/CSV field-test log;
- crash reports through TestFlight/App Store Connect.

These diagnostic features should remain optional and must not interfere with the minimal search UI.

This repository implements them in a separate `Field Diagnostics` sheet. JSONL and opt-in feedback JPEGs stay in
the app's Documents directory and are available in Files; there is no automatic upload.

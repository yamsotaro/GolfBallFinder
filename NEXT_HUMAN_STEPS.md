# Next Human Steps After Apple Developer Approval

These are the remaining actions that require the project owner's Apple/Codemagic accounts or the physical iPhone.
Do not put any credential, certificate, provisioning profile, `.p8` file, or secret value in Git.

## 1. Confirm unsigned model build first

Before configuring signing, push the intended commit and run Codemagic `ios-model-compile-check`. Keep its generated
`ModelManifest.json` and verify the workflow reaches the explicit `GolfBall.mlmodelc` bundle check. This workflow
does not require Apple Developer approval or signing. Confirm the log first reports a valid `PBXFileReference` and
application-target Sources phase for the model and Resources phase for `ModelManifest.json`, then shows a
`coremlcompiler`/Core ML compilation operation, the complete `*.mlmodelc` listing, the bundled JSON listing, and
successful checks for both `GolfBall.mlmodelc` and top-level `ModelManifest.json`.

## 2. Apple Developer and App Store Connect

1. Accept any current Apple Developer Program agreements and finish enrollment.
2. Register the explicit Bundle ID `com.yamsotaro.golfballfinder`.
3. Create the iOS app record in App Store Connect with exactly that Bundle ID and the app name/primary language.
4. In App Store Connect Users and Access > Integrations, create an API key with the minimum role Codemagic needs
   for signing/upload (normally App Manager for this workflow).
5. Download the `.p8` once and store it only in a password manager or the Codemagic integration. Never add it to
   this repository, chat, diagnostics folder, or ordinary environment file.
6. Complete tax/banking only if Apple requires it for the chosen distribution context; it is not an app runtime
   dependency.

## 3. Codemagic private configuration

1. In Team settings > Developer Portal, add the App Store Connect API integration using the exact reference name
   `GolfBallFinder Codemagic` expected by `codemagic.yaml`.
2. Create the protected variable group `golfballfinder_config`.
3. Add only `APP_STORE_APPLE_ID`, using the non-secret numeric Apple ID in App Store Connect > General > App
   Information. The Bundle ID is fixed in source as `com.yamsotaro.golfballfinder` and the workflow rejects any
   mismatch.
4. In Team settings > codemagic.yaml settings > Code signing identities, make sure an Apple Distribution certificate
   and App Store provisioning profile for `com.yamsotaro.golfballfinder` are available. Generate/fetch them through
   Codemagic using the configured Developer Portal integration; do not commit exported signing files.
5. Run `ios-testflight`. Confirm unit tests pass, then confirm the final IPA verification logs show the expected
   Bundle ID, the selected App Store build number, matching profile, Apple Distribution authority,
   `GolfBall.mlmodelc`, and `ModelManifest.json`.
6. `auth: integration` uploads the IPA and `submit_to_testflight: true` requests TestFlight post-processing. Resolve
   any export-compliance or beta metadata prompt in App Store Connect if Apple requires it.

If signing fails, compare Bundle ID, Team, App Store app record, certificate type, profile type, API-key role, and
unaccepted agreements. Do not “fix” signing by committing exported credentials.

## 4. TestFlight and iPhone 16 Pro

1. Add the owner's Apple Account to an internal TestFlight tester group.
2. Wait for App Store Connect processing; resolve export-compliance or beta metadata prompts if shown.
3. Install Apple's TestFlight app and then Golf Ball Finder on the iPhone 16 Pro.
4. Grant Camera permission. Deny it once on a disposable build if possible, confirm the app offers Settings recovery,
   then grant it.
5. Launch once, enable airplane mode, force-quit, relaunch, and verify the bundled model works offline.
6. Use Files > On My iPhone > Golf Ball Finder to verify FieldDiagnostics logs/evidence are recoverable.
7. Follow `FIELD_TEST_PLAN.md`; do not substitute Simulator observations for device measurements.

## 5. Device-only acceptance evidence

The owner must record, rather than assume:

- camera preview startup, portrait orientation, overlay/bbox alignment at screen edges;
- interruption, permission denial/recovery, background/foreground, and lock/unlock behavior;
- actual 1x/2x behavior and focus usability in rough;
- Core ML model load and inference while offline;
- inference p50/p95, effective FPS, candidate/scene confirmation latency;
- 10-second scene discovery and false confirmed alerts/min;
- success by occlusion bucket;
- 15-minute thermal/battery behavior;
- haptic/sound occurs once per confirmation and resets correctly.

Only after these measurements should thresholds, scan cadence, tiles, model precision, or a second-stage verifier be
tuned. Preserve every field result with app build number, commit, and model checkpoint SHA.

## 6. AGPL public-source gate before external TestFlight

Policy A is selected: GolfBallFinder and its fine-tuned model are distributed under AGPL-3.0-only while using the
Ultralytics AGPL path. Before inviting external testers, complete `docs/PUBLIC_RELEASE_CHECKLIST.md`, verify the source
repository and model release while signed out, and place `docs/TESTFLIGHT_BETA_DESCRIPTION.md` in the beta metadata.
Do not treat Apple approval or a successful upload as proof that the source/model publication gate has been met.

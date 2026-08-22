# Next Human Steps After Apple Developer Approval

These are the remaining actions that require the project owner's Apple/Codemagic accounts or the physical iPhone.
Do not put any credential, certificate, provisioning profile, `.p8` file, or secret value in Git.

## 1. Confirm unsigned model build first

Before configuring signing, push the intended commit and run Codemagic `ios-model-compile-check`. Keep its generated
`ModelManifest.json` and verify the workflow reaches the explicit `GolfBall.mlmodelc` bundle check. This workflow
does not require Apple Developer approval or signing. Confirm the log first reports a valid `PBXFileReference` and
application-target Sources phase, then shows a `coremlcompiler`/Core ML compilation operation and the complete
`*.mlmodelc` listing.

## 2. Apple Developer and App Store Connect

1. Accept any current Apple Developer Program agreements and finish enrollment.
2. Register a unique explicit Bundle ID, for example `com.<owner>.golfballfinder`.
3. Create the iOS app record in App Store Connect with exactly that Bundle ID and the app name/primary language.
4. In App Store Connect Users and Access > Integrations, create an API key with the minimum role Codemagic needs
   for signing/upload (normally App Manager for this workflow).
5. Download the `.p8` once and store it only in a password manager or the Codemagic integration. Never add it to
   this repository, chat, diagnostics folder, or ordinary environment file.
6. Complete tax/banking only if Apple requires it for the chosen distribution context; it is not an app runtime
   dependency.

## 3. Codemagic private configuration

1. In Team settings > Developer Portal, add the App Store Connect API integration using the reference name
   `golfballfinder-appstore` expected by `codemagic.yaml`.
2. Create the protected variable group `golfballfinder_config`.
3. Add `BUNDLE_ID` with the exact registered identifier. Do not store it by editing source for a one-off build.
4. Allow Codemagic to create/fetch an Apple Distribution certificate and App Store provisioning profile.
5. Run `ios-testflight`. Confirm its unit tests pass before the signing steps and that the model manifest in the build
   has the expected checkpoint SHA.
6. For the first signed diagnosis, leave `submit_to_testflight: false`. Confirm the generated IPA and App Store
   Connect upload path, then enable automatic TestFlight submission if desired.

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

## 6. Licensing decision before broader distribution

The current iOS runtime dependency is Ultralytics AGPL-3.0. Personal prototype/TestFlight work still needs to respect
its terms. Before any closed-source, commercial, client, or public App Store distribution, explicitly choose one:

- comply with the applicable AGPL obligations;
- obtain an appropriate Ultralytics commercial license;
- replace `GolfBallDetector` with a suitably licensed Core ML runtime/model path.

Do not treat Apple approval or a successful TestFlight upload as resolution of the OSS licensing question.

# GolfBallFinder Field Test Plan

Status: ready for first signed iPhone 16 Pro build; device results are not yet measured.  
Primary device: iPhone 16 Pro  
Primary runtime: bundled Core ML model, airplane-mode capable

## 1. Purpose and decision rule

The primary question is whether the app helps a person find a visible golf ball in real grass/rough within a
10-second deliberate scan without repeatedly confirming mushrooms, leaves, flowers, stones, glare, or other
non-balls. mAP is supporting information only.

Rank builds/models in this order:

1. scene discovery rate within 10 seconds (higher is better);
2. confirmed false alerts per minute of negative-only scanning (lower is better);
3. detection latency (lower is better);
4. success rate by occlusion bucket;
5. sustained thermal behavior and effective inference FPS.

Do not call the MVP accurate, real-time, thermally stable, or field-ready until this plan has produced measurements
on the physical iPhone 16 Pro.

## 2. Build identity and pre-flight

Before leaving for the field:

- Run Codemagic `ios-model-compile-check` on the exact commit. It must fetch the SHA256-pinned seed `.pt`, export
  `GolfBall.mlpackage`, run simulator tests, build unsigned, and verify `GolfBall.mlmodelc` in the `.app`.
- After Apple approval, build the same commit/model with `ios-testflight`.
- Record Git commit, App Store build number, app version, and `checkpoint_sha256` from `ModelManifest.json`.
- Install through TestFlight and open once with connectivity available.
- Enable airplane mode, relaunch, and confirm the bundled model still loads. Inference must not need a network.
- In iOS Settings, verify Golf Ball Finder has Camera permission. No Photos, Location, Microphone, or account is
  required.
- Open the ECG/waveform button and confirm Field Diagnostics shows a writable session ID.
- In Files, confirm `On My iPhone/Golf Ball Finder/FieldDiagnostics` exists after the first launch.

Record weather, grass condition, approximate ambient temperature, device case, starting battery, lens, and any
direct sun exposure. Do not compare thermal runs with materially different conditions without noting the difference.

## 3. Scene identifiers and bins

Use a unique Scene ID for every positive 10-second attempt, for example:

`courseA_20260830_v50_deep_shade_1x_001`

Required bins:

- visibility: `visible_100`, `visible_75`, `visible_50`, `visible_25`, `visible_10`;
- rough: short, medium, deep;
- light: sun, shade, backlight, overcast;
- lens: 1x or 2x;
- distance: `<1 m`, `1–2 m`, `2–4 m`, `>4 m`;
- outcome: verified true positive, false positive, or miss.

The app writes timestamps, inference latency, effective inference FPS, thermal state, confidence, scan mode,
candidate-to-confirmed time, scene-start-to-confirmed time, normalized bbox, zoom, and frame-drop counters.
Use the note field for rough/light/distance and the distractor type.

## 4. Positive-scene procedure (10 seconds)

Use at least 30 held-out scenes in the first serious pass, including at least five scenes at each visibility bucket
from 100% through 10% where a human can genuinely identify the ball. Do not reuse training scenes as held-out test
scenes.

For each scene:

1. Place or identify the ball and decide its visibility bucket before opening the diagnostic controls.
2. Stand at a realistic search distance and point away from the known location.
3. Open Field Diagnostics, select `ボールあり`, enter the unique Scene ID, select visibility, add conditions to the note, and tap
   `10秒シーン開始 / 再探索`.
4. Close diagnostics and scan naturally for at most 10 seconds. Use a repeatable slow sweep; do not hold the camera
   still directly over the known ball unless that matches normal use.
5. If the green confirmation points to the golf ball, immediately open diagnostics and tap `正しいボール発見`.
6. If a yellow/green indication points to a non-ball, tap `これはボールではない`; include the distractor type.
   Continue the attempt until 10 seconds or a correct confirmation.
7. If no correct confirmation occurs within 10 seconds, tap `見逃しを記録`. Keep the ball in the latest frame when
   possible so the saved JPEG can become a labeled positive example.
8. Tap `シーン終了`. Repeat selected scenes at both 1x and 2x using different Scene IDs. Treat them as separate
   attempts.

The automatic `confirmed` event is a detector state, not ground truth. Only the human-marked `true_positive` event
counts as a successful discovery.

## 5. Negative-only and Hard Negative procedure

Run at least ten one-minute negative-only scans, totaling at least ten minutes. Deliberately include mushrooms,
white/yellow flowers, pale stones, dry leaves, dew/glare, tees, litter, seed heads, sand edges, white shoes/clothing,
and strong reflections.

For each one-minute scan:

1. Verify no golf ball is present.
2. Select `ボールなし` and start a unique negative Scene ID such as `courseA_neg_mushroom_001`.
3. Scan naturally for exactly 60 seconds.
4. For every candidate or confirmed alert on a non-ball, tap `これはボールではない` and name the distractor.
5. Resume the same scene after each record; the app resets temporal confirmation but retains the Scene ID.
6. At exactly 60 seconds tap `シーン終了` so the denominator is recorded.
7. Retrieve the session's `evidence/false_positive_*.jpg` and matching `events.jsonl` records. These are the next
   high-value Hard Negative candidates; review them before adding empty YOLO label files.

Do not count yellow candidate flicker as a *confirmed* false alert, but preserve it as a false-positive training
sample when it is persistent or operationally distracting.

## 6. Thermal and lifecycle run

Perform one uninterrupted 15-minute scan with representative camera movement:

- record starting/ending battery and ambient conditions;
- keep Field Diagnostics closed during most of the run;
- at minutes 0, 5, 10, and 15 note thermal state, effective FPS, target FPS, and latency;
- verify target FPS steps down at `fair`, `serious`, and `critical` states;
- verify preview/UI remains usable and the app does not crash;
- lock/unlock the phone once, background/foreground once, and cause an interruption if safely reproducible;
- after resume, verify fresh detections occur and old confirmation state does not re-alert;
- test 1x→2x→1x and verify the displayed factor is the actual clamped device factor;
- rotate the phone physically while the app remains portrait-locked; verify preview, model coordinates, and overlay
  remain aligned.

The implementation discards frames while inference is busy and stores no pending inference frame. Compare
`framesDroppedBusy`, `framesDroppedCadence`, `cameraFramesReceived`, and `inferenceFramesAdmitted`; increasing drop
counters are expected, but increasing latency or stale overlays are not.

## 7. KPI calculations

Use only human-verified events and a frozen scene list.

- **10-second scene discovery rate** = positive scenes with a `true_positive` event at
  `scene_start_to_confirmed_ms <= 10000` / all attempted positive scenes.
- **False confirmed alerts/min** = human-rejected green confirmations / total negative-only scan minutes. Also report
  all rejected candidate+confirmed events separately.
- **Detection latency** = p50/p90 of `scene_start_to_confirmed_ms` for successful scenes. Separately report
  `candidate_to_confirmed_ms` to tune temporal confirmation.
- **Occlusion success** = successful scenes within 10 seconds / attempted scenes for each visibility bucket.
- **Thermal behavior** = time spent in each thermal state, effective FPS and inference latency by state, plus whether
  the app crashed or became operationally unusable.

Report sample counts beside every percentage. A 100% result on two scenes is not evidence of broad reliability.

Initial engineering targets remain hypotheses until measured:

- visible >=50%: at least 90% discovery within 10 seconds;
- visible about 25–50%: at least 80% within 10 seconds;
- at most one confirmed false alert per 60 seconds of negative-only scanning;
- preferred candidate-to-confirmed latency below 300 ms;
- no crash or unusable thermal degradation during the sustained run.

## 8. Export and dataset feedback loop

After each outing:

1. In Files, copy the complete `FieldDiagnostics/<session_id>` folder to Windows before removing the TestFlight app.
2. Keep `events.jsonl` and `evidence/` together; `evidence_filename`, event ID, Scene ID, build, and model SHA join them.
3. Review privacy/sensitive content before moving frames into any dataset. Nothing is uploaded automatically.
4. Put confirmed non-ball frames into a `hard_negative` session with empty label files.
5. Put miss frames into a positive session and annotate only a genuinely visible golf-ball extent.
6. Update `training/dataset_manifest.csv`; one session must belong to exactly one split.
7. Run `training/prepare_dataset.py`, then `training/validate_dataset.py`.
8. Train and evaluate on the unchanged frozen test sessions.
9. Generate one field result per build and compare at least two matched runs:

   ```bash
   python training/summarize_field_logs.py exported/session1/events.jsonl exported/session2/events.jsonl \
     --model-id golf_ball_build_001 \
     --protocol-id field-v1-frozen-scenes \
     --dataset-manifest training/datasets/golf_ball_v1/dataset_manifest.csv \
     --output field_build_001.json
   python training/compare_models.py field_build_001.json field_build_002.json \
     --output model_comparison.json
   ```

   The summarizer rejects scenes without both start/end events and rejects logs containing multiple checkpoint
   hashes.

Never move a field-test session between splits merely to improve a result, and never promote a model because mAP
improved while 10-second discovery or false confirmed alerts worsened.

## 9. Safety and failure handling

- Stop walking before operating diagnostics controls; do not stare at the phone while moving through rough terrain.
- Do not intentionally overheat the device. Stop the sustained run if iOS shows a temperature warning.
- If JPEG evidence fails, retain the JSONL metadata and note the failure; do not report the case as fully captured.
- Simulator latency, camera behavior, and thermal values do not count as physical-device evidence.

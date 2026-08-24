# Test plan

## Windows/Python checks

The Windows bootstrap runs these automatically after dependencies are installed:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s Tests/Python -v
```

The suite validates frame-extraction safety, bundle/build-number configuration, YAML structure, the required
Codemagic workflows, and App Icon configuration. Python syntax is checked separately with `py_compile`/`compileall`.

## Automated unit tests

Run after generating the Xcode project:

```bash
xcodebuild test \
  -project GolfBallFinder.xcodeproj \
  -scheme GolfBallFinder \
  -destination 'platform=iOS Simulator,name=iPhone 16 Pro'
```

If that simulator name is unavailable, use an installed iPhone simulator UDID.
The Codemagic workflows select the iPhone 16 Pro simulator when present and otherwise fall back to an available
iPhone simulator, so a renamed simulator image does not disable the test gate.

Covered pure logic:

- crop-relative -> full-frame normalized coordinate mapping;
- aspect-fill image -> screen coordinate mapping;
- ROI clamp/expansion;
- full/tile alternation;
- ROI lock after a candidate;
- ROI release after repeated misses;
- one-frame false positive remains only a candidate;
- three spatially consistent hits confirm a ball;
- intermittent false positives and spatially inconsistent tracks do not confirm;
- multiple candidates preserve the spatially consistent track;
- confirmation/reset and one-time feedback behavior;
- every screen edge, oversized/negative ROI clipping, and crop/aspect-fill coordinate edges;
- inference busy/cadence gating has no pending backlog and accepts fresh frames after camera timestamp reset;
- raw Core ML `[1, 5, 8400]` output is decoded by the pinned SDK into top-left normalized CGRects;
- malformed/non-finite/out-of-bounds and extreme-aspect boxes never enter candidate confirmation;
- the reported `x=0.996 y=0.017 w=0.004 h=0.980` box remains scanning even when repeated;
- synthetic full-frame, four-corner, edge, overlapping-tile, and near-edge ROI boxes map correctly;
- portrait aspect-fill overlay geometry preserves top-left origin and places valid boxes correctly;
- re-search clears candidate/found history, ROI, candidate timing, and restarts an active scene clock;
- candidate-to-confirmed timing uses only the confirming spatial track, not an earlier unrelated candidate;
- a synthetic 32BGRA camera buffer renders as RGBA/sRGB in the diagnostics-only Raw preview;
- thermal policy monotonically reduces target inference FPS.
- Color Assist OFF preserves the original round-robin order;
- prioritized cycles remain deterministic and visit all five tiles without starvation;
- priority changes apply at a tile-cycle boundary, not halfway through a cycle;
- all four Color Assist modes produce finite tile scores on Core Image;
- white-ball saliency can prioritize a synthetic white patch in grass and render Debug previews;
- Color Assist pauses at serious/critical thermal state while ROI behavior remains authoritative.

## Build gate

```bash
xcodebuild \
  -project GolfBallFinder.xcodeproj \
  -scheme GolfBallFinder \
  -sdk iphonesimulator \
  -configuration Debug \
  build CODE_SIGNING_ALLOWED=NO
```

## Physical device smoke test

1. Install on iPhone 16 Pro.
2. Grant camera permission.
3. Confirm camera preview starts.
4. Confirm model loads from the app bundle with airplane mode enabled.
5. Place a ball fully visible on grass at ~1m.
6. Scan slowly; confirm ring, arrow, haptic and beep.
7. Remove ball and scan for one minute; record confirmed false alerts.
8. Before any accuracy run, move a clearly visible target through center, four corners, and screen edges; verify the ring follows it in full, tile, and ROI modes.
9. Press Re-search while candidate/found and while ROI is active; verify state returns to scanning and timing fields clear/restart.
10. Toggle 2x and repeat.
11. Cover ~50% of the ball with grass and repeat.
12. Put a white mushroom/leaf/stone-like distractor in frame and repeat.

## Performance capture

Record at least:

- model load time;
- live `inferenceMs` p50/p95;
- processed FPS;
- device thermal state;
- battery delta over a 10-minute scan;
- full-frame, tile, ROI inference latency.
- Color Assist processing p50/p95, busy-frame drop delta, and effective FPS delta with the same scenes OFF/ON.

The separate Field Diagnostics sheet records these fields at a bounded 2 Hz sample cadence while updating live
metrics per completed inference. It also records scene starts, human-verified correct finds, false positives, and
misses. Use the root `FIELD_TEST_PLAN.md` as the device execution protocol.

Never substitute simulator performance for physical-device performance.

## Field test matrix

Minimum first serious field pass:

- 30 positive scenes across at least 3 environmental conditions;
- 10 negative-only scenes;
- at least 5 deep-rough/partial-visibility scenes;
- both 1x and 2x in selected scenes;
- collect screenshots/video or written failure descriptions for every miss/false alert.
- repeat a frozen subset with Color Assist OFF and ON using the same model, sweep, lens, and scene order.

Start from `docs/FIELD_TEST_LOG_TEMPLATE.csv`; preserve the exact app build number and checkpoint hash for each row.

## Regression rule

Before accepting a new model/config:

- evaluate on the same frozen held-out test sessions;
- do not tune using the test split;
- compare scene discovery, false alerts/minute, and latency together;
- reject a model that improves mAP while materially worsening field discovery or false-alert behavior.
- reject Color Assist if tile rank looks better but discovery, false alerts, latency, or thermal behavior worsens.

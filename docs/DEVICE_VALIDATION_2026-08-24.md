# iPhone 16 Pro device validation — 2026-08-24

## Observed result

The first physical-device run proved app launch, camera capture, Core ML execution, and live
inference. It did **not** prove detection accuracy. Color Assist was OFF.

| Metric | Measured value |
| --- | ---: |
| Inference latency | 5.5 ms |
| Effective inference FPS | 15.0 |
| Configured target FPS | 20 |
| Thermal state | nominal |
| Reported confidence | 0.370 |
| Reported scan mode | ROI |
| Reported normalized bbox | x 0.996, y 0.017, w 0.004, h 0.980 |
| Candidate to confirmed | 305,768 ms |
| Frames received / admitted | 3,790 / 1,895 |
| Dropped busy / cadence | 20 / 1,875 |

The reported box is a right-edge vertical sliver, not a plausible golf-ball box. The success UI
was therefore a false confirmation and must not be counted as a model-quality success.

## Audited model/output contract

`training/export_coreml.py` exports the pinned one-class YOLOv8 checkpoint at 640 x 640 with
`nms=False`. Loading the actual checkpoint on Windows produced a raw detector head with shape
`[1, 5, 8400]`:

- channels 0...3: center-x, center-y, width, height in model-input pixels;
- channel 4: `golf_ball` class confidence;
- no separate objectness channel.

The exact generated Core ML output feature name and data type are assigned by Core ML Tools and
must not be guessed. The generated package is intentionally absent from Git. The model and signed
Codemagic workflows now open the actual `GolfBall.mlpackage` specification, print every input and
output name/type/shape, store that result in `ModelManifest.json`, and fail unless the raw output is
`[1, 5, 8400]` with one 640 x 640 image input.

UltralyticsYOLO v8.9.13 consumes that raw tensor. Its traditional-output decoder converts center
xywh to corner xyxy, reverses the 640-square letterbox into detector-input pixels, clamps to the
detector input, and exposes `Box.xywhn` as a **top-left-origin normalized CGRect**. The app must not
center-decode `Box.xywhn` a second time.

## App coordinate contract

All rectangles after `GolfBallDetector` use normalized top-left coordinates:

1. Full frame: `full = local`.
2. Tile or ROI crop `(cx, cy, cw, ch)`:
   `fullX = cx + localX * cw`, `fullY = cy + localY * ch`,
   `fullW = localW * cw`, `fullH = localH * ch`.
3. Core Image crops convert top-left normalized Y to Core Image bottom-left pixels with
   `pixelY = imageMinY + (1 - cropMaxY) * imageHeight`.
4. Both the video-data connection and preview-layer connection request 90-degree portrait
   rotation. The delivered CIImage and SDK result are therefore treated as orientation `.up`.
5. The preview uses aspect fill. Overlay scale is
   `max(viewWidth/imageWidth, viewHeight/imageHeight)` with centered crop offsets.

For the reported right-edge sliver, aspect fill places the center beyond the visible right edge on
a portrait display, while its almost full-height box creates a huge ring. That explains why a
normal marker was not visible even though the state said found.

## Defenses and timing correction

Before ROI locking and again before temporal confirmation, a box must now have finite components,
positive width/height, normalized bounds (small floating-point tolerance only), at least a 2-pixel
side, normalized aspect ratio at most 4:1, and pixel-space aspect ratio at most 4:1. These constants
live in `AppConfig` for future measured tuning. The reported 245:1 normalized box is rejected.

The old Candidate-to-confirmed metric started when any candidate state first appeared and did not
restart when tracking jumped to an unrelated location. A sequence of unrelated false candidates
could therefore report several minutes. Confirmation latency now comes from the camera timestamps
of the hits in the single spatial cluster that actually confirmed. Manual re-search resets the
scheduler/ROI, stabilizer/history, candidate and confirmed diagnostics, candidate timing, and the
active scene timing window.

Confidence 0.370 was allowed by configuration: model 0.12, candidate 0.14, confirmed average 0.20,
with 3 hits in 5 frames. That path was internally consistent, but geometry had no plausibility gate.

## Debug preview

Capture explicitly requests 32-bit BGRA. Core Image interprets that pixel buffer; the code does not
manually reinterpret BGRA bytes before YOLO. Diagnostics-only preview images are now materialized as
RGBA8 in sRGB, and tests exercise a synthetic 32BGRA green buffer to detect red/blue channel swaps.
This changes only debug visualization. The raw CIImage/crop supplied to YOLO and Color Assist OFF
behavior are unchanged.

## Required follow-up

Run `ios-compile-check` and `ios-model-compile-check`, inspect the printed actual model output name
and type, then repeat the physical-device geometry protocol in `docs/TEST_PLAN.md`. Do not begin
model accuracy scoring until valid overlays and sane timing pass on full-frame, tile, and ROI scans.

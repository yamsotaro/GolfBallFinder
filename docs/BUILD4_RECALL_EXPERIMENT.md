# Build 4 recall experiment

External TestFlight submission is stopped until this experiment passes all gates. Build 3 device misses have blank
detector diagnostics, so this work targets detector recall for small and partially occluded white balls. It does not
change bbox geometry, overlay mapping, ROI behavior, Color Assist, or the raw RGB model input.

## Data and license decision

The reproducible source/count/license record is `training/build4_recall_manifest.json`. The accepted real source is
Open Images V7: 882 license-verified images, consisting of 382 positives and 500 hard negatives. Its annotations are
CC BY 4.0 and each selected image metadata row identifies CC BY 2.0; per-image attribution remains in
`training/datasets/public_mvp_v3/attribution.csv`.

The following candidates were investigated but were not mixed into Build 4:

- Roboflow Universe `golf-mxabz/golfball-4jxv1`: the public page identifies 17,460 object-detection images and CC BY
  4.0. Download required an authenticated account/API path unavailable to this run, so zero images were used.
- Kaggle sports-ball classification sets: licensing was shown on their dataset pages, but they do not provide the
  required golf-ball object boxes and some describe web-search collection without adequate per-image provenance.
  Zero images were used.
- Zenodo `Accurate Balls Detection`: the record did not specify a usable license. Zero images were used.
- Objects365: its official terms restrict the dataset to academic use and do not provide rights suitable for this
  external app. Zero images were used.

Never substitute an unofficial mirror or infer a license. A future authenticated download must be recorded with its
exact version URL, license, counts, and attribution before use.

## Build 4 challenge set

`training/build_recall_dataset.py` creates 324 deterministic, split-safe synthetic positives as an auxiliary layer on
top of 882 real images. Real data remains the majority. A synthetic ball crop and its grass background always come
from the same pre-existing split; parent session IDs, measured visibility, 640-input bbox scale, background/placement
grass scores, and challenge category are written to the generated manifest.

The generated synthetic distribution is:

| Dimension | Bin counts |
| --- | --- |
| Bbox scale at 640 | `<12`: 126; `12–20`: 94; `20–40`: 65; `>40`: 39 |
| Visible fraction | `75–100`: 79; `50–75`: 83; `30–50`: 81; `<30`: 81 |
| Output split | train: 240; val: 36; test: 48 |

Source crops must contain genuinely green vegetation; broad brown color alone is not accepted because it confused
roads, wood, skin, and clothing with dry grass. Brown/dead-grass color is instead added as a controlled augmentation.
Exact and perceptual collisions at dHash distance 6 or less are rejected against all real and already accepted
synthetic images. The standard validator then checks class IDs, bounds, zero-size boxes, missing pairs, corruption,
duplicates, and scene-level split isolation.

## Reproduction

On Windows, run the complete nano experiment with:

```powershell
.\scripts\run_build4_recall_pipeline.ps1 -Epochs 15 -Batch 8 -Device cpu
```

Add `-IncludeTileSweep` to evaluate the candidate using a complete full+5-tile and full+3x3 static scan cycle. Dataset
pixels and training outputs remain ignored by Git. The script does not export Core ML, replace the app model, submit a
TestFlight build, commit, or push.

Training keeps the export input at 640. Multi-scale augmentation varies training resolution only; the required Core ML
contract remains one RGB `image` input at 640x640 and raw output `[1, 5, 8400]` with center-x, center-y, width, height,
and class confidence channels and no independent objectness channel.

## Build 3 fixed baseline

At confirmed threshold 0.44 on the 170-image Build 4 test split, full-frame Build 3 produced precision 0.8889, recall
0.3390, 5 FP, and 1 confidence>=0.8 FP. Recall by 640-input bbox scale was 0.000 (`<12`), 0.222 (`12–20`), 0.423
(`20–40`), and 0.742 (`>40`). Its synthetic 30–50% visibility and `<30%` visibility recall were both zero.

Using the same checkpoint, data, and threshold, full+5-tile produced precision 0.4253/recall 0.3136/50 FP, while
full+3x3 produced 0.4500/0.3051/44 FP. Tile-edge fragments and locally magnified distractors outweighed any recall
benefit, so this evidence does not authorize changing the production scan layout.

## Release gate

The candidate must achieve overall precision at least 0.90, overall recall at least 0.85, `<12px` recall at least
0.80, and combined 30–75% visibility recall at least 0.80 without materially regressing hard-negative or
confidence>=0.8 FP behavior. These numeric gates are necessary but not sufficient: iPhone 16 Pro latency, 1x/2x
distance scenes, temporal confirmation, and representative real partial-occlusion scenes still require device tests.

## Completed offline result (2026-08-25)

All operating-point comparisons below use the same 170-image held-out test split, IoU 0.5, and unchanged confirmed
threshold 0.44. Synthetic challenge results are reported separately from the 56 real-positive test images and must not
be treated as measured field accuracy.

| Model / scan | Precision | Recall | FP | Confidence >= 0.8 FP | mAP50 | mAP50–95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Build 3 nano / full | 0.8889 | 0.3390 | 5 | 1 | 0.5221 | 0.2747 |
| Build 4 nano / full | 0.8431 | 0.7288 | 16 | 1 | 0.7770 | 0.4683 |
| Build 4 nano / full+5-tile | 0.5959 | 0.7373 | 59 | 7 | 0.7770 | 0.4683 |
| Build 4 nano / full+3x3 | 0.5755 | 0.6780 | 59 | 4 | 0.7770 | 0.4683 |
| Build 4 small probe / full | 0.7551 | 0.3136 | 12 | 1 | 0.5486 | 0.2570 |

Build 4 nano full-frame recall by input-scale bin is 0.618 (`<12`), 0.630 (`12–20`), 0.923 (`20–40`), and
0.774 (`>40`). Its synthetic visibility recall is 0.917 (`75–100`), 1.000 (`50–75`), 0.833 (`30–50`), and
0.917 (`<30`). The corresponding synthetic challenge recall is 0.800 for B/small, 0.917 for C/partial occlusion,
and 0.917 for D/deep rough. However, recall on the real-positive group is only 0.600 and seven operating-threshold
FPs occur in the independent hard-negative category. The sole confidence >= 0.8 FP is a large incorrect box in a
positive image containing two ground-truth balls; the high-confidence FP count therefore did not improve from Build 3.

The nano run used 15 CPU epochs and took 4,351 seconds (72.5 minutes). The selected checkpoint is 24,461,415 bytes,
3.01 million parameters, and 8.1 GFLOPs. The small comparison was intentionally a five-epoch capacity probe from the
official COCO-pretrained YOLOv8s checkpoint; it took 3,386 seconds (56.4 minutes), selected an 89,474,599-byte training
checkpoint, and has 11.13 million parameters and 28.4 GFLOPs. Because it was both much larger and worse after the
bounded probe, it was not extended or selected. This does not claim that a fully converged small model could never
improve the result.

Full-frame Build 4 nano is the best candidate from this experiment. Neither tile layout is authorized: 5-tile gained
one TP while adding 43 FPs, and 3x3 reduced recall while adding 43 FPs versus full-frame. The production scan strategy,
model, and thresholds remain unchanged. The validation threshold search suggested candidate 0.29 and confirmed 0.44,
but its confirmed validation point (precision 0.875, recall 0.716) also failed the gate, so no threshold change is
authorized.

Offline CPU inference averaged about 67 ms/image for nano and 166 ms/image for small in the comparable Ultralytics
test pass. These are not iPhone measurements. Using the prior Build 3 iPhone observation of roughly 6–10 ms only as a
baseline, nano still requires a fresh device measurement; small's 3.5x compute makes it the clear latency/FPS risk.
No same-scene 1x/2x frames were available, so the camera strategy was not measured. In ideal optics 2x would roughly
double object diameter in model pixels, but crop, stabilization, focus, exposure, and lens switching make a device A/B
mandatory before adopting any 1x/2x policy.

The selected nano PyTorch checkpoint preserves class `0: golf_ball` and the export contract: RGB `image`,
`[1, 3, 640, 640]` input, raw `[1, 5, 8400]` output, center-x/center-y/width/height/class-confidence channels, and no
independent objectness channel. Because the release gate failed, no Core ML export was promoted, no app model was
replaced, and no TestFlight build was submitted. The next iteration needs more independent real 2–3 m, partial-
occlusion, and hard-negative field-like data before repeating the same pipeline.

## Subsequent limited-beta promotion decision

After reviewing the failed external-beta gate, the Build 4 nano checkpoint was explicitly authorized for a limited
TestFlight device comparison. This exception does not change the measured metrics or claim that the external-beta gate
passed. The byte-identical, Git-ignored release copy is `training/models/public_mvp_v4.pt`; its SHA256, publication
state, exact contract, Build 3 comparison, and limitations are pinned in `training/public_mvp_release_v4.json`.
Production thresholds, bbox geometry/decoder, Color Assist baseline, and the adaptive full-frame/five-tile/ROI scan
strategy remain unchanged. Core ML export and TestFlight submission remain delegated to Codemagic after the public
release asset is uploaded and its URL/SHA environment variables are updated by a human.

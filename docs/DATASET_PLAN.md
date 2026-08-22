# Dataset and model-improvement plan

## Objective

Build a one-class `golf_ball` detector optimized for a handheld iPhone looking downward/obliquely into grass and rough. The detector should see difficult partial balls while the surrounding pipeline suppresses false positives.

## Sources

Use public data only after inspecting the images and recording license/attribution in a manifest. Public golf-ball datasets are useful seed data but are unlikely to match the exact phone/rough domain. The decisive dataset is self-recorded iPhone footage.

Suggested manifest columns:

```text
source_id,source_url,license,download_date,image_count,allowed_use,attribution,notes
```

## Capture batches

### Batch A — obvious positives

- Ball fully visible.
- 0.5–3m.
- Multiple angles.
- Short grass and fairway-like surfaces.
- 200–400 diverse frames.

Purpose: get the entire training/deployment loop working.

### Batch B — rough and occlusion

- 10–25%, 25–50%, 50–75% occlusion categories.
- Deep rough with only arcs/patches visible.
- Shadows and backlight.
- Dirty ball.
- 500–1,000 frames.

### Batch C — hard negatives

Record the same camera movement with no golf ball and deliberately include:

- mushrooms;
- flowers/dandelions;
- pale stones;
- dead leaves;
- water/dew highlights;
- tees/markers;
- trash;
- white shoes;
- seed heads;
- sand/soil bright patches.

Aim for at least 500 negative-only frames early.

### Batch D — field failures

Every confirmed false positive and every missed visible ball should become a labeled training example. This batch has the highest value after the first usable model.

## Annotation

- One class only: `golf_ball`.
- Tight axis-aligned box around the visible ball extent, not the hypothetical hidden full sphere.
- For severe occlusion, label the visible part only if a human reviewer can genuinely identify it as a ball from the frame/context.
- Do not label ambiguous pixels that even a human cannot identify.
- Negative images have empty/no label files.

## Split rules

Use whole sessions/scenes:

```text
course_A_morning_clip_01..10 -> train
course_B_afternoon_clip_01..04 -> validation
course_C_overcast_clip_01..05 -> test
```

Never distribute neighboring frames from one clip across multiple splits.

The maintained manifest is `training/dataset_manifest.csv` (copy the example). Required pipeline columns are:

```text
session_id,split,source_dir,content_type
```

`content_type` is `positive`, `hard_negative`, or `mixed`. Build and validate without random frame splitting:

```bash
python training/prepare_dataset.py --manifest training/dataset_manifest.csv --output training/datasets/golf_ball_v1
python training/validate_dataset.py \
  --data training/datasets/golf_ball_v1/dataset.yaml \
  --manifest training/datasets/golf_ball_v1/dataset_manifest.csv
```

The validator requires `images/<split>/<session_id>/...`, checks one-class YOLO boxes and boundaries, rejects a
positive label in a Hard Negative session, and detects exact image duplicates crossing splits. `train.py` and
`evaluate.py` call the same validation before model work and record dataset/manifest hashes.

`training/extract_frames.py` writes each source video to a readable, path-hashed session directory and records a
`session.json`. It refuses to mix with an existing extraction unless `--overwrite` is explicitly supplied; that
flag removes only generated `frame_*.jpg` files for the matching session and preserves unrelated reviewer notes.

## Metadata bins to track manually or in filenames

- visibility: `v100`, `v75`, `v50`, `v25`, `v10`;
- distance: `d0_1`, `d1_2`, `d2_4`, `d4plus`;
- light: `sun`, `shade`, `backlight`, `overcast`;
- rough: `short`, `medium`, `deep`;
- negative type when relevant.

This enables slice-specific recall analysis beyond a single mAP number.

## Training baseline

Start:

```bash
python training/train.py \
  --data training/datasets/golf_ball_v1/dataset.yaml \
  --manifest training/datasets/golf_ball_v1/dataset_manifest.csv \
  --base yolo26n.pt --epochs 120 --imgsz 640
```

Then compare one controlled YOLO11n run if useful. Keep the same dataset split and evaluation settings.

## Augmentation rules

Helpful starting augmentations:

- scale variation;
- modest HSV/brightness changes;
- horizontal flip;
- mild translation;
- mild perspective/rotation;
- mosaic during early training.

Do not use unrealistic transformations that turn grass/ball geometry into a different domain. After actual failure analysis, add custom synthetic occlusion only if it improves held-out real scenes.

## App-level evaluation protocol

For each held-out scene:

1. Place/identify the ball and mark ground truth.
2. Begin from a realistic user distance.
3. Start a timer and scan naturally for up to 10 seconds.
4. Record whether a confirmed alert occurs and time-to-first-confirmed alert.
5. Record whether the alert points at the actual ball.
6. Run separate no-ball scans for false-alert rate.

Primary outputs:

- scene discovery rate;
- time-to-first-confirmed detection p50/p90;
- false confirmed alerts per minute;
- recall by occlusion/distance/light/rough bin;
- iPhone inference p50/p95;
- thermal behavior.

Generate each model's device result from completed app JSONL scenes using `training/summarize_field_logs.py` (the
checked-in field evaluation JSON is the schema example). Compare only matched device, protocol, and frozen-manifest
runs with `training/compare_models.py`; its ordering prioritizes 10-second discovery, false confirmed alerts/min,
and latency before offline mAP.

## Color Assist reference and future image-input A/B

`training/color_assist.py` mirrors the iOS low-resolution formulas and evaluates round-robin OFF versus saliency
ordered ON using `training/color_assist_manifest.example.csv`. Use held-out field images with either a normalized
ball bbox or a human `ball_tile_index`. Report tile rank/top-1/top-3 plus reference-machine processing latency; do
not interpret these as YOLO accuracy or iPhone thermal results.

For the device experiment, use the same Raw RGB model/checkpoint for both arms and change only the experimental
Color Assist flag. Summarize separate completed JSONL runs so discovery within 10 seconds, time to first candidate,
false confirmed alerts/min, confirmation latency, occlusion success, Color Assist latency, and thermal state remain
comparable.

A future Filtered RGB model may be trained as a separate model/input variant using exported
`golf_contrast_rgb` images, but its dataset manifest, model ID, checkpoint, and evaluation arm must remain separate.
The current production detector input is Raw RGB only.

# Public-data MVP model pipeline

This pipeline builds the pre-field-data beta model from license-filtered public images. It does not replace the
post-beta loop based on human-reviewed iPhone field samples.

## Data source and license gate

`training/public_dataset_sources.yaml` is the checked-in source catalog. The current source is Open Images V7:

- annotations: CC BY 4.0;
- images: CC BY 2.0 as recorded in each selected image's Open Images metadata;
- attribution: required and written per image to the public metadata file
  `training/datasets/public_mvp_v3/attribution.csv`;
- source/license URLs and downloaded/used counts: written to the public metadata file
  `training/datasets/public_mvp_v3/source_manifest.json`.

Open Images warns users to verify image licenses. The builder therefore rejects any image whose metadata License URL
is not on the catalog allow-list and preserves author, original landing page, and license for audit. Do not add a
source with an unclear license.

## Build and validate the dataset

From the repository root on Windows:

```powershell
.\.venv\Scripts\python.exe training/build_public_dataset.py `
  --output training/datasets/public_mvp_v3 `
  --max-positives 800 `
  --max-negatives 500 `
  --scene-negatives 150
```

The builder downloads official annotations/metadata into `.cache`, verifies image licenses and decodability,
normalizes Golf ball boxes to YOLO class `0`, creates empty labels for negatives, removes exact and perceptual
duplicates, and applies a deterministic image/session-level 75/12.5/12.5 split. Dataset pixels, labels, and generated
training outputs are ignored by Git; the attribution/source-manifest metadata is deliberately exposed by the narrow
`.gitignore` exceptions.

Scene negatives use human-verified Grass, Grassland, Lawn, Meadow, and Pasture labels. The broader `Golf course`
image-level class is deliberately excluded: contact-sheet review found that such scenes can contain tiny real golf
balls even when no Golf ball box is present, which would create harmful false-negative training labels.

Render deterministic contact sheets before training:

```powershell
.\.venv\Scripts\python.exe training/render_dataset_review.py `
  --data training/datasets/public_mvp_v3/dataset.yaml `
  --manifest training/datasets/public_mvp_v3/dataset_manifest.csv `
  --split train --content-type positive --count 48 `
  --output tmp/public-positive-review.jpg
```

Repeat for `hard_negative` and for held-out `test`. This human review is required because a verified scene label is
not proof that every small object in an image was exhaustively annotated.

## Train and evaluate

Use the current seed as initialization, not as the beta accuracy baseline:

```powershell
.\.venv\Scripts\python.exe training/train.py `
  --data training/datasets/public_mvp_v3/dataset.yaml `
  --manifest training/datasets/public_mvp_v3/dataset_manifest.csv `
  --base training/models/seed_golf_ball_yolov8n.pt `
  --epochs 20 --imgsz 640 --batch 8 --device cpu --workers 0 `
  --name public_mvp_hardneg_v3 --seed 42
```

Run `training/evaluate.py` first on `val` with `--recommend-thresholds`, then evaluate seed and new checkpoints on
the untouched `test` split at the same selected operating threshold. Treat confidence >= 0.8 false positives as a
separate release gate. Offline image metrics do not measure iOS temporal confirmation or field discovery.

`training/train.py` saves each epoch by default. `training/select_checkpoint.py` evaluates every epoch on validation
and selects by the precision/recall gate plus confidence >= 0.8 FP count, rather than accepting Ultralytics' mAP-only
`best.pt`. `scripts/run_public_mvp_pipeline.ps1` performs this selection automatically and writes the release copy to
`training/models/public_mvp_best.pt`.

The recorded v3 selection uses model/candidate/confirmed thresholds `0.23 / 0.25 / 0.44`. These were selected on
validation with false positives prioritized; they are not a claim of field accuracy. The untouched public test split
did not meet the aspirational precision >= 0.90 and recall >= 0.80 gate, so iPhone field validation remains required.

## Core ML and CI handoff

Core ML export remains on hosted macOS. Publish only the selected checkpoint as the public GitHub Release asset
documented in `training/public_mvp_release_v3.json`, verify it while signed out, and place its public URL and SHA256 in
Codemagic group `golfballfinder_config`. Then run `ios-model-compile-check`. The workflow refuses a hash mismatch and
validates the exact one-input/one-output raw contract before Xcode compilation. After that passes, run
`ios-testflight` to export the signed IPA and submit it to TestFlight.

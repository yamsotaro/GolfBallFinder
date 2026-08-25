# Third-party notices

GolfBallFinder itself is licensed under AGPL-3.0-only; see `LICENSE`. Third-party works remain under their own
licenses. This file is an inventory and is not a substitute for the upstream license texts included with each
package or binary distribution.

## Ultralytics YOLO iOS / Ultralytics Python

- iOS repository: `https://github.com/ultralytics/yolo-ios-app`
- Python repository: `https://github.com/ultralytics/ultralytics`
- License used by this project: GNU AGPL-3.0
- This repository pins the iOS package to v8.9.13 and the training package versions in
  `training/requirements*.txt`.
- GolfBallFinder follows the AGPL path: the application source, build/export scripts, corresponding model release
  metadata, and fine-tuned checkpoint are intended for public distribution under AGPL-3.0-only. No Ultralytics
  Enterprise License is asserted.

## SAHI

- Repository: `obss/sahi`
- License: MIT
- We reuse the design concept (overlapping sliced inference), not SAHI's Python runtime on iOS.

## Direct build, training, and export dependencies

The lock inputs are the exact Swift package version in `project.yml` and the exact Python versions in
`training/requirements.txt` and `training/requirements-windows.txt`. The Xcode project is generated, so its generated
`Package.resolved` is not the source-of-truth lock. The declared primary licenses below were checked against
upstream/package metadata. Binary wheels can contain additional notices; preserve the license files shipped in each
installed distribution.

| Dependency | Role | Primary license / notice source |
| --- | --- | --- |
| Apple coremltools | Core ML export/inspection | BSD-3-Clause; `https://github.com/apple/coremltools` |
| PyTorch | YOLO training/export | BSD-3-Clause plus bundled third-party notices; `https://github.com/pytorch/pytorch` |
| torchvision | PyTorch vision operators | BSD-3-Clause; `https://github.com/pytorch/vision` |
| NumPy | Numeric processing | BSD-3-Clause plus bundled notices; `https://github.com/numpy/numpy` |
| opencv-python/OpenCV | Image processing | Apache-2.0 plus wheel third-party notices; `https://github.com/opencv/opencv-python` |
| PyYAML | Configuration parsing | MIT; `https://github.com/yaml/pyyaml` |
| Pillow | Image decoding/processing | MIT-CMU; `https://github.com/python-pillow/Pillow` |
| huggingface_hub | Seed checkpoint retrieval | Apache-2.0; `https://github.com/huggingface/huggingface_hub` |

Transitive development dependencies are not incorporated into the iOS application merely by running the training
toolchain. Their own terms still apply when installing or redistributing those packages.

## Bootstrap checkpoint

- Model repository: `https://huggingface.co/notjulietxd/golf-ball-tracker`
- File used: `best.pt`
- License stated by the model repository: Apache-2.0
- Pinned SHA256: `45e8f8bd8975dc7f437919a11c3f6ee1fe7c8ae40b0f49910d0677d1c0326791`
- Role: initialization checkpoint and seed baseline; it is not shipped as the selected iOS beta checkpoint.

The selected fine-tuned checkpoint has its own AGPL-3.0-only publication record in
`training/public_mvp_release_v3.json`. Preserve this upstream checkpoint notice and do not imply endorsement by its
author.

## Public golf-ball datasets/models

Public resources mentioned in `docs/RESEARCH_NOTES.md` have their own licenses. Do not merge data into a training
corpus until its license and attribution requirements are recorded in a dataset manifest.

### Open Images Dataset V7

- Dataset and download documentation: `https://storage.googleapis.com/openimages/web/download_v7.html`
- Annotations: CC BY 4.0.
- Selected image pixels: CC BY 2.0 as listed in each image's Open Images metadata.
- Attribution: required. The public snapshot `training/datasets/public_mvp_v3/attribution.csv` keeps author, original
  landing page, license URL, Open Images ID, content role, and content SHA256 for all 882 selected images. Aggregate
  counts, source URLs, split counts, and manifest hashes are in `training/public_mvp_v3.lock.json`; the generated
  source manifest is `training/datasets/public_mvp_v3/source_manifest.json`.
- The image pixels and generated YOLO dataset are not redistributed in Git. They are reproducibly downloaded from
  their recorded original/public URLs by `training/build_public_dataset.py` and validated against the published
  metadata.

Open Images notes that users remain responsible for verifying image licenses. The public-data builder only accepts
the explicit CC BY 2.0 metadata URL allow-list and fails closed for other or missing license values.

The training output and the fine-tuned checkpoint do not relicense Open Images photographs. Dataset attribution and
license records must remain available alongside any reproducibility materials.

# Third-party notices

This prototype is designed for personal experimentation and intentionally keeps third-party code as dependencies rather than copying it into the repository.

## Ultralytics YOLO iOS / Ultralytics Python

- Repository: `ultralytics/yolo-ios-app`
- iOS package license: AGPL-3.0
- The Python `ultralytics` package is also distributed under Ultralytics' AGPL/commercial licensing model.
- This repository pins the iOS package to v8.9.13.
- Personal experimentation is the intended MVP use. Before distributing a closed-source or commercial application, re-evaluate the license and either comply with AGPL obligations, obtain an Ultralytics commercial license, or replace this layer with a permissively licensed/custom Core ML runtime.

## SAHI

- Repository: `obss/sahi`
- License: MIT
- We reuse the design concept (overlapping sliced inference), not SAHI's Python runtime on iOS.

## Public golf-ball datasets/models

Public resources mentioned in `docs/RESEARCH_NOTES.md` have their own licenses (for example CC BY 4.0 or Apache-2.0). Do not merge data into a training corpus until its license and attribution requirements are recorded in a dataset manifest.

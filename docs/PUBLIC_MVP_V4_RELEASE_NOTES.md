# GolfBallFinder Public MVP v4

Release asset: `public_mvp_v4.pt`

SHA256: `1cf77c75ec1cd4e8f66e4abddee13d038dd7604a17ce16b8709ada7e89746426`

- Recall-focused update.
- Improved small and partially occluded golf-ball detection in the held-out Build 4 experiment.
- Retains the 640x640 nano model contract, current production confidence thresholds, and existing adaptive
  full-frame, five-tile, and ROI scan strategy.
- Known limitation: false positives increased versus v3, and real-world rough performance still requires further
  field data.
- This checkpoint is a limited TestFlight Build 4 candidate and did not pass the external beta quality gate.

The checkpoint is licensed `AGPL-3.0-only`. Full provenance, exact metrics, contract, limitations, and publication
state are recorded in `training/public_mvp_release_v4.json`. Do not mark the asset as published until the release URL
works without authentication and the downloaded asset independently matches the SHA256 above.

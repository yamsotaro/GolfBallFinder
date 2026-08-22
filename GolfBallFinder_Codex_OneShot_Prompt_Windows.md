# GolfBallFinder Codex one-shot prompt — Windows entry point

This repository supports development from Windows without a physical Mac. The normative implementation prompt is
[`docs/CODEX_PROMPT.md`](docs/CODEX_PROMPT.md), and the hosted Apple-toolchain path is
[`docs/WINDOWS_CLOUD_BUILD.md`](docs/WINDOWS_CLOUD_BUILD.md).

Before changing code, read both files completely together with `AGENTS.md`, `README.md`, every document under
`docs/`, `project.yml`, `codemagic.yaml`, `training/`, `scripts/`, and the Swift implementation/tests.

Execute all work possible on Windows: bootstrap the Python environment, run Python tests and static checks, review
Swift source and tests, and keep the `ios-compile-check` Codemagic workflow green. Use the `ios-testflight` workflow
for Core ML export, Xcode compilation, signing, App Store Connect upload, and TestFlight delivery. Do not commit
Apple credentials, signing files, `.p8` keys, or personal identifiers.

Product invariants remain those in `AGENTS.md` and `docs/CODEX_PROMPT.md`: local inference only, minimal UI,
adaptive full-frame/tiled/candidate-ROI scanning, temporal/spatial confirmation, a replaceable detector adapter,
freshest-frame processing without an unbounded inference queue, tested coordinate conversion, and measured—not
claimed—field accuracy driven by hard negatives and real iPhone footage.

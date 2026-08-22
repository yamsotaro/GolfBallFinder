# AGENTS.md

## Objective
Build and validate the smallest iPhone 16 Pro app that reliably finds golf balls in grass/rough using local camera inference.

## Environment
- Primary developer workstation may be Windows. Do not stop because Xcode is unavailable locally.
- Use Codemagic hosted macOS for Apple-toolchain compile/sign/TestFlight when no physical Mac exists.
- Read `docs/WINDOWS_CLOUD_BUILD.md` for that path.

## Mandatory reading
- `docs/CODEX_PROMPT.md`
- `docs/DEVELOPMENT_SPEC.md`
- `docs/DATASET_PLAN.md`
- `docs/TEST_PLAN.md`
- `THIRD_PARTY_NOTICES.md`

## Invariants
- Local inference only for MVP; no cloud/API upload.
- Keep UI minimal.
- Preserve adaptive full-frame -> tiled -> candidate-ROI scanning.
- Do not confirm a ball from one frame; use temporal/spatial confirmation.
- Keep detector behind an adapter so Ultralytics can later be replaced.
- No unbounded inference queue; process the freshest frame.
- Validate coordinate conversions with tests.
- Field data and hard negatives drive accuracy improvements.
- Do not claim unmeasured accuracy.

## Before finishing a change
1. Run relevant unit tests.
2. Build with local Xcode when available; otherwise keep Codemagic `ios-compile-check`/`ios-testflight` reproducible.
3. Update README/docs if reproduction steps changed.
4. Report any device-only validation still required.

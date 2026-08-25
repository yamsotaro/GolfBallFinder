# AGPL public release checklist

This checklist is the release gate for making GolfBallFinder public and inviting external TestFlight testers. It is
an engineering compliance checklist, not legal advice. Do not distribute a beta until the repository and model URLs
below are publicly reachable without credentials.

## Current audit status

The authorized local history rewrite replaced the author and committer email metadata in all eight existing commits
with the repository owner's GitHub-provided ID-based `noreply` address. The post-rewrite audit found zero personal
emails and no credential/private-key material in worktree files or reachable Git blobs. The actual old-to-new commit
mapping and old commit hashes are intentionally not stored in the public repository. The rewritten history has been
validated directly from current reachable commit metadata and blobs rather than from a repository mapping file.

## License and public locations

- Project license: GNU Affero General Public License v3.0 only (`AGPL-3.0-only`), exact text in `LICENSE`.
- Public source: `https://github.com/yamsotaro/GolfBallFinder`.
- Planned model asset:
  `https://github.com/yamsotaro/GolfBallFinder/releases/download/public-mvp-v4/public_mvp_v4.pt`.
- Model SHA256: `1cf77c75ec1cd4e8f66e4abddee13d038dd7604a17ce16b8709ada7e89746426`.
- Model provenance and evaluation: `training/public_mvp_release_v4.json`.
- Third-party and dataset notices: `THIRD_PARTY_NOTICES.md`.

## Complete Corresponding Source inventory

The public revision corresponding to a TestFlight build must include:

- `GolfBallFinder/`: iOS application source and non-generated resource documentation;
- `Tests/` and `Tests/Python/`: Swift and Python regression tests;
- `project.yml` and `project.compile-check.yml`: XcodeGen/package configuration, including the exact Swift package
  pin (the generated Xcode project's `Package.resolved` is not the source of truth);
- `codemagic.yaml`: unsigned, model, signing, export, and TestFlight workflows without secret values;
- `scripts/`: bootstrap, model retrieval, release audit, Bundle ID configuration, and pipeline orchestration;
- `training/*.py`, `training/*.ps1`, and `training/requirements*.txt`: data, training, evaluation, selection, export,
  and inspection programs with pinned inputs;
- `training/public_dataset_sources.yaml`, `training/public_mvp_v3.lock.json`,
  `training/datasets/public_mvp_v3/source_manifest.json`, and
  `training/datasets/public_mvp_v3/attribution.csv`: source policy, frozen counts/hashes, and all 882 per-image
  attributions;
- `training/public_mvp_release_v3.json` and `training/public_mvp_release_v4.json`: historical/current checkpoint SHA,
  settings, thresholds, measured metrics, limitations, and publication plan;
- `docs/`, `README.md`, `FIELD_TEST_PLAN.md`, and `NEXT_HUMAN_STEPS.md`: build, validation, and operating docs;
- `LICENSE` and `THIRD_PARTY_NOTICES.md`.

The public Git repository intentionally excludes downloaded dataset pixels/labels, `runs/`, caches, virtual
environments, generated Xcode projects, exported Core ML bundles, checkpoints, IPAs/archives, signing identities,
provisioning profiles, and all credentials. Those are either reproducible outputs or private material, not source
files that should be committed. The selected checkpoint is a separate public release asset because of its size.

## Required pre-public checks

Run from the repository root on Windows:

```powershell
.\.venv\Scripts\python.exe scripts\audit_public_release.py
.\.venv\Scripts\python.exe -m unittest discover -s Tests\Python -v
git diff --check
git status --short
git ls-files --cached --others --exclude-standard
```

Then inspect the entire output file list, all staged/unstaged diffs, GitHub repository visibility, branch protection,
and every release asset. Never paste a `.p8`, `.p12`, issuer ID, key ID, private key, App Store Connect token, or
Codemagic secret into Git, a release description, or a TestFlight field.

The audit checks every tracked/untracked publication candidate, sensitive filenames and private-key containers in
ignored trees, commit author/committer email metadata, and every blob in all reachable commits. If it reports a
history finding, do not make the repository public. Record only its commit, path, and finding type; rotate a credential
first when applicable and clean history with an explicitly reviewed procedure.

## Publication order

1. Review and commit the complete source revision without generated/private files.
2. Push it, change the repository to Public, and verify the source URL while signed out.
3. Create public release tag `public-mvp-v4`; upload `training/models/public_mvp_v4.pt` as
   `public_mvp_v4.pt`; use `docs/PUBLIC_MVP_V4_RELEASE_NOTES.md` and state AGPL-3.0-only, the source revision,
   provenance JSON path, and SHA256.
4. Download the asset while signed out and independently verify its SHA256.
5. Set Codemagic `MODEL_CHECKPOINT_URL` to the public HTTPS asset URL and keep
   `MODEL_CHECKPOINT_SHA256` equal to the published SHA. These are identifiers, not secrets.
6. Run `ios-compile-check`, then `ios-model-compile-check`, then `ios-testflight`.
7. Add the text from `docs/TESTFLIGHT_BETA_DESCRIPTION.md` to the beta description/review information and verify its
   links before inviting an external tester.

Do not mark `checkpoint.publication.published` true until steps 2 through 4 actually succeed. Update the manifest in
a follow-up source revision if the final tag, URL, or asset name differs.

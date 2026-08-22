#!/usr/bin/env python3
"""Download the public Apache-2.0 seed checkpoint from Hugging Face with a pinned SHA256."""
from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO = "notjulietxd/golf-ball-tracker"
FILENAME = "best.pt"
EXPECTED_SHA256 = "45e8f8bd8975dc7f437919a11c3f6ee1fe7c8ae40b0f49910d0677d1c0326791"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="training/models/seed_golf_ball_yolov8n.pt")
    args = p.parse_args()

    downloaded = Path(hf_hub_download(repo_id=REPO, filename=FILENAME))
    digest = sha256(downloaded)
    if digest != EXPECTED_SHA256:
        raise SystemExit(f"SHA256 mismatch. expected={EXPECTED_SHA256} actual={digest}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(downloaded, out)
    print(f"Verified seed model written to {out}")
    print("Note: this is a bootstrap checkpoint, not the target rough/grass production model.")


if __name__ == "__main__":
    main()

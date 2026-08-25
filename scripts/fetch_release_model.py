#!/usr/bin/env python3
"""Fetch a SHA256-pinned release checkpoint without storing credentials in Git."""
from __future__ import annotations

import argparse
import hashlib
import os
import re
import shutil
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen


SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def install_checkpoint(source: str, expected_sha256: str, output: Path) -> str:
    expected = expected_sha256.strip().lower()
    if not SHA256_PATTERN.fullmatch(expected):
        raise ValueError("MODEL_CHECKPOINT_SHA256 must be 64 lowercase hexadecimal characters")
    parsed = urlparse(source)
    if parsed.scheme != "https":
        raise ValueError("MODEL_CHECKPOINT_URL must use HTTPS")

    output.parent.mkdir(parents=True, exist_ok=True)
    if output.is_file() and sha256(output) == expected:
        return expected

    temporary = output.with_name(output.name + ".part")
    temporary.unlink(missing_ok=True)
    try:
        request = Request(source, headers={"User-Agent": "GolfBallFinder-model-fetch/1.0"})
        try:
            with urlopen(request, timeout=180) as response, temporary.open("wb") as file:
                if urlparse(response.geturl()).scheme != "https":
                    raise ValueError("Checkpoint download redirected away from HTTPS")
                shutil.copyfileobj(response, file, length=1024 * 1024)
        except (HTTPError, URLError, TimeoutError, OSError) as error:
            raise ValueError(f"Checkpoint download failed: {type(error).__name__}") from error
        actual = sha256(temporary)
        if actual != expected:
            raise ValueError(f"Checkpoint SHA256 mismatch: expected={expected} actual={actual}")
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    return expected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="training/models/release_golf_ball.pt")
    parser.add_argument("--url", default=os.environ.get("MODEL_CHECKPOINT_URL"))
    parser.add_argument("--sha256", default=os.environ.get("MODEL_CHECKPOINT_SHA256"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.url or not args.sha256:
        raise SystemExit(
            "Set MODEL_CHECKPOINT_URL and MODEL_CHECKPOINT_SHA256 in Codemagic's "
            "golfballfinder_config group"
        )
    try:
        digest = install_checkpoint(args.url, args.sha256, Path(args.out))
    except ValueError as error:
        raise SystemExit(str(error)) from error
    print(f"Verified release checkpoint SHA256: {digest}")
    print(f"Release checkpoint written to: {Path(args.out)}")


if __name__ == "__main__":
    main()

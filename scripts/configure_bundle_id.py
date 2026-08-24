#!/usr/bin/env python3
"""Set canonical release identifiers in XcodeGen config without requiring a Mac.

This repository has one signed product. Unsigned workflows may use their own
temporary project overlay, but this release helper refuses any other Bundle ID.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path


RELEASE_BUNDLE_ID = "com.yamsotaro.golfballfinder"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-id", required=True)
    parser.add_argument(
        "--build-number",
        type=int,
        help="Positive build number selected from App Store Connect history.",
    )
    parser.add_argument("--project", default="project.yml")
    args = parser.parse_args()

    bundle_id = args.bundle_id.strip()
    if not re.fullmatch(r"[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+", bundle_id):
        raise SystemExit(f"Invalid reverse-DNS bundle identifier: {bundle_id!r}")
    if bundle_id != RELEASE_BUNDLE_ID:
        raise SystemExit(
            f"Release Bundle ID must be {RELEASE_BUNDLE_ID!r}, got {bundle_id!r}"
        )
    if args.build_number is not None and args.build_number < 1:
        raise SystemExit("Build number must be a positive integer")

    path = Path(args.project)
    text = path.read_text(encoding="utf-8")
    updated, count = re.subn(
        r"(?m)^(\s*PRODUCT_BUNDLE_IDENTIFIER:\s*).+$",
        rf"\g<1>{bundle_id}",
        text,
        count=1,
    )
    if count != 1:
        raise SystemExit("Could not find exactly one PRODUCT_BUNDLE_IDENTIFIER in project.yml")

    if args.build_number is not None:
        updated, count = re.subn(
            r"(?m)^(\s*CURRENT_PROJECT_VERSION:\s*).+$",
            rf"\g<1>{args.build_number}",
            updated,
            count=1,
        )
        if count != 1:
            raise SystemExit("Could not find exactly one CURRENT_PROJECT_VERSION in project.yml")

    path.write_text(updated, encoding="utf-8")
    print(f"Configured bundle identifier: {bundle_id}")
    if args.build_number is not None:
        print(f"Configured build number: {args.build_number}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Set release identifiers in the XcodeGen configuration without requiring a Mac.

Used by cloud CI before `xcodegen generate`, and can also be run manually.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-id", required=True)
    parser.add_argument(
        "--build-number",
        type=int,
        help="Positive App Store build number (normally Codemagic BUILD_NUMBER).",
    )
    parser.add_argument("--project", default="project.yml")
    args = parser.parse_args()

    bundle_id = args.bundle_id.strip()
    if not re.fullmatch(r"[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+", bundle_id):
        raise SystemExit(f"Invalid reverse-DNS bundle identifier: {bundle_id!r}")
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

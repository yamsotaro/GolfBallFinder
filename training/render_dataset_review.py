#!/usr/bin/env python3
"""Render a deterministic contact sheet for human dataset quality review."""
from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path

import yaml
from PIL import Image, ImageDraw, ImageFont, ImageOps

try:
    from .validate_dataset import IMAGE_SUFFIXES
except ImportError:
    from validate_dataset import IMAGE_SUFFIXES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--attribution", type=Path)
    parser.add_argument(
        "--negative-class",
        action="append",
        default=[],
        help="For hard negatives, keep attribution rows containing this class (repeatable)",
    )
    parser.add_argument("--split", choices=["train", "val", "test"], default="train")
    parser.add_argument("--content-type", choices=["positive", "hard_negative"], required=True)
    parser.add_argument("--source-type", choices=["real", "synthetic"])
    parser.add_argument("--count", type=int, default=48)
    parser.add_argument("--columns", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def stable_rank(path: Path, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{path.as_posix()}".encode()).hexdigest()


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.data.read_text(encoding="utf-8"))
    root = Path(str(config["path"]))
    root = root if root.is_absolute() else (args.data.parent / root).resolve()
    session_types: dict[str, str] = {}
    session_source_types: dict[str, str] = {}
    with args.manifest.open(encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            if row["split"] == args.split:
                session_types[row["session_id"]] = row["content_type"]
                session_source_types[row["session_id"]] = row.get("source_type", "real") or "real"
    image_root = root / "images" / args.split
    allowed_sessions: set[str] | None = None
    if args.negative_class:
        if args.attribution is None:
            raise SystemExit("--negative-class requires --attribution")
        requested = set(args.negative_class)
        allowed_sessions = set()
        with args.attribution.open(encoding="utf-8", newline="") as file:
            for row in csv.DictReader(file):
                classes = set(filter(None, row.get("negative_classes", "").split("|")))
                if row.get("split") == args.split and requested & classes:
                    allowed_sessions.add(row["session_id"])
    images = [
        path
        for path in image_root.rglob("*")
        if path.suffix.lower() in IMAGE_SUFFIXES
        and session_types.get(path.relative_to(image_root).parts[0]) == args.content_type
        and (
            args.source_type is None
            or session_source_types.get(path.relative_to(image_root).parts[0]) == args.source_type
        )
        and (allowed_sessions is None or path.relative_to(image_root).parts[0] in allowed_sessions)
    ]
    images = sorted(images, key=lambda path: stable_rank(path.relative_to(root), args.seed))[: args.count]
    if not images:
        raise SystemExit("No matching images")

    tile_width, tile_height = 240, 200
    rows = (len(images) + args.columns - 1) // args.columns
    sheet = Image.new("RGB", (tile_width * args.columns, tile_height * rows), "#202020")
    font = ImageFont.load_default()
    for index, path in enumerate(images):
        with Image.open(path) as source:
            source = ImageOps.exif_transpose(source).convert("RGB")
            source.thumbnail((tile_width, tile_height - 24), Image.Resampling.LANCZOS)
            tile = Image.new("RGB", (tile_width, tile_height), "#101010")
            x_offset = (tile_width - source.width) // 2
            y_offset = (tile_height - 24 - source.height) // 2
            tile.paste(source, (x_offset, y_offset))
            draw = ImageDraw.Draw(tile)
            label_path = (
                root
                / "labels"
                / args.split
                / path.relative_to(image_root).with_suffix(".txt")
            )
            for line in label_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                _, cx, cy, width, height = (float(value) for value in line.split())
                x1 = x_offset + (cx - width / 2) * source.width
                y1 = y_offset + (cy - height / 2) * source.height
                x2 = x_offset + (cx + width / 2) * source.width
                y2 = y_offset + (cy + height / 2) * source.height
                draw.rectangle((x1, y1, x2, y2), outline="#00ff66", width=3)
            draw.text((4, tile_height - 20), path.stem[:28], fill="white", font=font)
        sheet.paste(tile, ((index % args.columns) * tile_width, (index // args.columns) * tile_height))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.output, quality=90)
    print(f"Rendered {len(images)} {args.content_type} images to {args.output}")


if __name__ == "__main__":
    main()

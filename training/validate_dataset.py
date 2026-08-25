#!/usr/bin/env python3
"""Validate a session-split one-class YOLO dataset before training or comparison."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml
from PIL import Image, ImageOps


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".heic", ".bmp", ".tif", ".tiff"}
VALID_SPLITS = {"train", "val", "test"}
VALID_CONTENT_TYPES = {"positive", "hard_negative", "mixed"}


class DatasetValidationError(ValueError):
    pass


@dataclass(frozen=True)
class SessionEntry:
    session_id: str
    split: str
    source_dir: str
    content_type: str


@dataclass
class DatasetSummary:
    sessions: int = 0
    images: int = 0
    positive_images: int = 0
    negative_images: int = 0
    boxes: int = 0
    hard_negative_images: int = 0


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_difference_hash(path: Path, hash_size: int = 16) -> int:
    """Fully decode an image and return a 256-bit perceptual difference hash."""
    try:
        with Image.open(path) as image:
            image.load()
            image = ImageOps.exif_transpose(image).convert("L")
            image = image.resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
            pixels = image.tobytes()
    except (OSError, ValueError, ModuleNotFoundError) as error:
        raise DatasetValidationError(f"Corrupt or undecodable image: {path}: {error}") from error
    value = 0
    for row in range(hash_size):
        offset = row * (hash_size + 1)
        for column in range(hash_size):
            value = (value << 1) | int(pixels[offset + column] > pixels[offset + column + 1])
    return value


def load_session_manifest(path: Path) -> dict[str, SessionEntry]:
    if not path.is_file():
        raise DatasetValidationError(f"Session manifest not found: {path}")
    entries: dict[str, SessionEntry] = {}
    with path.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        required = {"session_id", "split", "source_dir", "content_type"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise DatasetValidationError(f"Manifest missing columns: {sorted(missing)}")
        for line_number, row in enumerate(reader, start=2):
            session_id = (row.get("session_id") or "").strip()
            split = (row.get("split") or "").strip()
            source_dir = (row.get("source_dir") or "").strip()
            content_type = (row.get("content_type") or "").strip()
            if not session_id or "/" in session_id or "\\" in session_id:
                raise DatasetValidationError(f"Invalid session_id at manifest line {line_number}: {session_id!r}")
            if split not in VALID_SPLITS:
                raise DatasetValidationError(f"Invalid split at manifest line {line_number}: {split!r}")
            if content_type not in VALID_CONTENT_TYPES:
                raise DatasetValidationError(
                    f"Invalid content_type at manifest line {line_number}: {content_type!r}"
                )
            if session_id in entries:
                raise DatasetValidationError(f"Duplicate session_id in manifest: {session_id}")
            entries[session_id] = SessionEntry(session_id, split, source_dir, content_type)
    if not entries:
        raise DatasetValidationError("Session manifest contains no sessions")
    return entries


def _dataset_root(config_path: Path, config: dict[str, Any]) -> Path:
    raw = Path(str(config.get("path", "")))
    if not str(raw):
        raise DatasetValidationError("dataset.yaml requires a path")
    return raw.resolve() if raw.is_absolute() else (config_path.parent / raw).resolve()


def _label_path(dataset_root: Path, split: str, image_relative: Path) -> Path:
    return dataset_root / "labels" / split / image_relative.with_suffix(".txt")


def validate_label_file(path: Path) -> int:
    if not path.exists() or not path.read_text(encoding="utf-8").strip():
        return 0
    boxes = 0
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        parts = line.split()
        if len(parts) != 5:
            raise DatasetValidationError(f"{path}:{line_number}: expected 5 YOLO fields")
        try:
            class_id_value = float(parts[0])
            values = [float(value) for value in parts[1:]]
        except ValueError as error:
            raise DatasetValidationError(f"{path}:{line_number}: non-numeric label") from error
        if not class_id_value.is_integer() or int(class_id_value) != 0:
            raise DatasetValidationError(f"{path}:{line_number}: only class 0 (golf_ball) is allowed")
        if not all(math.isfinite(value) for value in values):
            raise DatasetValidationError(f"{path}:{line_number}: non-finite box value")
        x, y, width, height = values
        if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < width <= 1 and 0 < height <= 1):
            raise DatasetValidationError(f"{path}:{line_number}: normalized box values are out of range")
        tolerance = 1e-6
        if x - width / 2 < -tolerance or x + width / 2 > 1 + tolerance:
            raise DatasetValidationError(f"{path}:{line_number}: box crosses horizontal image boundary")
        if y - height / 2 < -tolerance or y + height / 2 > 1 + tolerance:
            raise DatasetValidationError(f"{path}:{line_number}: box crosses vertical image boundary")
        boxes += 1
    return boxes


def validate_dataset_config(config_path: Path, manifest_path: Path) -> DatasetSummary:
    if not config_path.is_file():
        raise DatasetValidationError(f"Dataset config not found: {config_path}")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise DatasetValidationError("Dataset config must be a YAML mapping")
    if config.get("names") != {0: "golf_ball"}:
        raise DatasetValidationError("Dataset must contain exactly class 0: golf_ball")

    sessions = load_session_manifest(manifest_path)
    root = _dataset_root(config_path, config)
    summary = DatasetSummary()
    seen_sessions: set[str] = set()
    boxes_by_session: dict[str, int] = {session_id: 0 for session_id in sessions}
    digest_to_location: dict[str, Path] = {}
    perceptual_hashes: list[tuple[int, Path]] = []

    for split in ("train", "val", "test"):
        configured = config.get(split)
        if not isinstance(configured, str):
            raise DatasetValidationError(f"Dataset config requires a directory string for {split}")
        image_root = Path(configured)
        image_root = image_root if image_root.is_absolute() else root / image_root
        if not image_root.is_dir():
            raise DatasetValidationError(f"Image split directory not found: {image_root}")
        images = sorted(path for path in image_root.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)
        if not images:
            raise DatasetValidationError(f"Image split is empty: {split}")

        for image in images:
            relative = image.relative_to(image_root)
            if len(relative.parts) < 2:
                raise DatasetValidationError(
                    f"Image is not under a session directory: {image}. Expected images/{split}/<session_id>/..."
                )
            session_id = relative.parts[0]
            if session_id not in sessions:
                raise DatasetValidationError(f"Image session is absent from manifest: {session_id}")
            session = sessions[session_id]
            if session.split != split:
                raise DatasetValidationError(
                    f"Session leakage: {session_id} is manifest={session.split}, filesystem={split}"
                )
            seen_sessions.add(session_id)

            digest = file_sha256(image)
            previous = digest_to_location.get(digest)
            if previous:
                raise DatasetValidationError(f"Exact image duplicate: {previous} and {image}")
            digest_to_location[digest] = image

            perceptual_hash = image_difference_hash(image)
            near_duplicate = next(
                (
                    previous_path
                    for previous_hash, previous_path in perceptual_hashes
                    if (perceptual_hash ^ previous_hash).bit_count() <= 6
                ),
                None,
            )
            if near_duplicate:
                raise DatasetValidationError(
                    f"Perceptual near duplicate (dHash distance <= 6): {near_duplicate} and {image}"
                )
            perceptual_hashes.append((perceptual_hash, image))

            label = _label_path(root, split, relative)
            box_count = validate_label_file(label)
            if session.content_type == "hard_negative" and box_count:
                raise DatasetValidationError(f"Hard-negative session has a positive box: {label}")
            summary.images += 1
            summary.boxes += box_count
            boxes_by_session[session_id] += box_count
            if box_count:
                summary.positive_images += 1
            else:
                summary.negative_images += 1
                if session.content_type == "hard_negative":
                    summary.hard_negative_images += 1

    missing_sessions = set(sessions) - seen_sessions
    if missing_sessions:
        raise DatasetValidationError(f"Manifest sessions have no dataset images: {sorted(missing_sessions)}")
    empty_positive_sessions = sorted(
        session_id
        for session_id, session in sessions.items()
        if session.content_type == "positive" and boxes_by_session[session_id] == 0
    )
    if empty_positive_sessions:
        raise DatasetValidationError(
            f"Positive sessions contain no golf_ball boxes: {empty_positive_sessions}"
        )
    summary.sessions = len(seen_sessions)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="training/dataset.yaml")
    parser.add_argument("--manifest", default="training/dataset_manifest.csv")
    parser.add_argument("--json-output")
    args = parser.parse_args()

    summary = validate_dataset_config(Path(args.data).resolve(), Path(args.manifest).resolve())
    payload = asdict(summary)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.json_output:
        Path(args.json_output).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

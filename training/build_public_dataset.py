#!/usr/bin/env python3
"""Build a licensed, deduplicated one-class YOLO dataset from public sources.

The initial implementation deliberately supports the official Open Images V7
source catalog only. It downloads source annotations/metadata into an ignored
cache, verifies every selected image's per-image license, keeps real Golf ball
boxes, creates empty labels for hard negatives, removes exact/perceptual
duplicates, and performs a deterministic image/scene-level split.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import sys
import time
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import yaml
from PIL import Image, ImageOps

try:
    from .validate_dataset import validate_dataset_config
except ImportError:
    from validate_dataset import validate_dataset_config


OFFICIAL_SPLITS = ("train", "validation", "test")
OUTPUT_SPLITS = ("train", "val", "test")
DOWNLOAD_CHUNK_BYTES = 4 * 1024 * 1024
PROGRESS_BYTES = 128 * 1024 * 1024


@dataclass
class SourceImage:
    key: str
    official_split: str
    image_id: str
    content_type: str
    boxes: list[tuple[float, float, float, float]] = field(default_factory=list)
    negative_classes: set[str] = field(default_factory=set)
    metadata: dict[str, str] = field(default_factory=dict)
    cached_path: Path | None = None
    sha256: str | None = None
    difference_hash: int | None = None
    width: int | None = None
    height: int | None = None


def stable_rank(value: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def difference_hash(image: Image.Image, hash_size: int = 16) -> int:
    gray = ImageOps.grayscale(image).resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    pixels = gray.tobytes()
    value = 0
    for row in range(hash_size):
        offset = row * (hash_size + 1)
        for column in range(hash_size):
            value = (value << 1) | int(pixels[offset + column] > pixels[offset + column + 1])
    return value


def hamming_distance(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def download_file(url: str, destination: Path) -> None:
    """Download a large public file with resumable temporary storage."""
    if destination.is_file() and destination.stat().st_size > 0:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    existing = partial.stat().st_size if partial.exists() else 0
    headers = {"User-Agent": "GolfBallFinder-public-dataset/1.0"}
    if existing:
        headers["Range"] = f"bytes={existing}-"
    request = Request(url, headers=headers)
    with urlopen(request, timeout=120) as response:
        append = existing > 0 and getattr(response, "status", None) == 206
        if existing and not append:
            existing = 0
        mode = "ab" if append else "wb"
        downloaded = existing
        next_progress = ((downloaded // PROGRESS_BYTES) + 1) * PROGRESS_BYTES
        with partial.open(mode) as file:
            while True:
                chunk = response.read(DOWNLOAD_CHUNK_BYTES)
                if not chunk:
                    break
                file.write(chunk)
                downloaded += len(chunk)
                if downloaded >= next_progress:
                    print(f"downloaded {downloaded / (1024 ** 2):.0f} MiB: {destination.name}", flush=True)
                    next_progress += PROGRESS_BYTES
    partial.replace(destination)


def parse_box(row: dict[str, str]) -> tuple[float, float, float, float] | None:
    try:
        xmin = float(row["XMin"])
        xmax = float(row["XMax"])
        ymin = float(row["YMin"])
        ymax = float(row["YMax"])
    except (KeyError, ValueError):
        return None
    values = (xmin, xmax, ymin, ymax)
    if not all(math.isfinite(value) for value in values):
        return None
    if not (0 <= xmin < xmax <= 1 and 0 <= ymin < ymax <= 1):
        return None
    width = xmax - xmin
    height = ymax - ymin
    aspect = max(width / height, height / width)
    if aspect > 4:
        return None
    return xmin, ymin, xmax, ymax


def scan_annotations(
    paths: dict[str, Path],
    positive_mid: str,
    negative_mid_to_name: dict[str, str],
) -> tuple[dict[str, SourceImage], dict[str, set[str]], dict[str, Any]]:
    positives: dict[str, SourceImage] = {}
    negatives: dict[str, set[str]] = defaultdict(set)
    report: dict[str, Any] = {"annotation_rows": {}, "rejected_positive_boxes": 0}
    for split, path in paths.items():
        row_count = 0
        with path.open(encoding="utf-8", newline="") as file:
            for row in csv.DictReader(file):
                row_count += 1
                mid = row.get("LabelName", "")
                if mid != positive_mid and mid not in negative_mid_to_name:
                    continue
                image_id = row.get("ImageID", "")
                if not image_id:
                    continue
                key = f"{split}/{image_id}"
                if mid == positive_mid:
                    if row.get("IsDepiction") == "1" or row.get("IsGroupOf") == "1" or row.get("IsInside") == "1":
                        continue
                    box = parse_box(row)
                    if box is None:
                        report["rejected_positive_boxes"] += 1
                        continue
                    record = positives.setdefault(
                        key,
                        SourceImage(key, split, image_id, "positive"),
                    )
                    record.boxes.append(box)
                else:
                    negatives[key].add(negative_mid_to_name[mid])
        report["annotation_rows"][split] = row_count
        print(
            f"scanned {path.name}: rows={row_count:,}, positives={sum(1 for key in positives if key.startswith(split + '/')):,}",
            flush=True,
        )
    for key in positives:
        negatives.pop(key, None)
    return positives, negatives, report


def scan_scene_negative_labels(
    paths: dict[str, Path],
    positive_mid: str,
    scene_mid_to_name: dict[str, str],
    bbox_positive_keys: set[str],
) -> tuple[dict[str, set[str]], dict[str, int]]:
    """Find human-verified grass/course scenes that have no known golf ball.

    Open Images image-level annotations explicitly encode presence with
    Confidence=1.  A candidate is rejected if Golf ball is positively verified
    or if the dense bounding-box scan found a Golf ball in the same image.
    Empty/absent Golf ball image-level labels are not treated as proof by
    themselves; the bbox exclusion and subsequent contact-sheet review remain
    required quality controls.
    """
    candidates: dict[str, set[str]] = defaultdict(set)
    verified_positive: set[str] = set()
    row_counts: dict[str, int] = {}
    for split, path in paths.items():
        row_count = 0
        with path.open(encoding="utf-8", newline="") as file:
            for row in csv.DictReader(file):
                row_count += 1
                try:
                    present = float(row.get("Confidence", "0")) == 1.0
                except ValueError:
                    continue
                if not present:
                    continue
                image_id = row.get("ImageID", "")
                if not image_id:
                    continue
                key = f"{split}/{image_id}"
                mid = row.get("LabelName", "")
                if mid == positive_mid:
                    verified_positive.add(key)
                elif mid in scene_mid_to_name:
                    candidates[key].add(scene_mid_to_name[mid])
        row_counts[split] = row_count
        print(f"scanned {path.name}: rows={row_count:,}, scene candidates={len(candidates):,}", flush=True)
    for key in verified_positive | bbox_positive_keys:
        candidates.pop(key, None)
    return candidates, row_counts


def select_balanced_negatives(
    candidates: dict[str, set[str]],
    maximum: int,
    seed: int,
) -> list[SourceImage]:
    by_class: dict[str, list[str]] = defaultdict(list)
    for key, classes in candidates.items():
        for class_name in classes:
            by_class[class_name].append(key)
    queues = {
        name: deque(sorted(set(keys), key=lambda key: stable_rank(f"{name}:{key}", seed)))
        for name, keys in by_class.items()
    }
    selected: list[str] = []
    selected_set: set[str] = set()
    class_names = sorted(queues)
    while len(selected) < maximum and any(queues.values()):
        made_progress = False
        for class_name in class_names:
            queue = queues[class_name]
            while queue and queue[0] in selected_set:
                queue.popleft()
            if not queue:
                continue
            key = queue.popleft()
            selected.append(key)
            selected_set.add(key)
            made_progress = True
            if len(selected) == maximum:
                break
        if not made_progress:
            break
    return [
        SourceImage(
            key=key,
            official_split=key.split("/", 1)[0],
            image_id=key.split("/", 1)[1],
            content_type="hard_negative",
            negative_classes=set(candidates[key]),
        )
        for key in selected
    ]


def load_selected_metadata(paths: dict[str, Path], selected: dict[str, SourceImage]) -> None:
    remaining = set(selected)
    for split, path in paths.items():
        split_ids = {key.split("/", 1)[1] for key in remaining if key.startswith(split + "/")}
        if not split_ids:
            continue
        with path.open(encoding="utf-8", newline="") as file:
            for row in csv.DictReader(file):
                image_id = row.get("ImageID", "")
                if image_id in split_ids:
                    key = f"{split}/{image_id}"
                    selected[key].metadata = {name: value or "" for name, value in row.items()}
                    remaining.discard(key)
    if remaining:
        raise RuntimeError(f"Selected images missing Open Images metadata: {sorted(remaining)[:10]}")


def download_image(record: SourceImage, cache: Path, url_template: str, retries: int = 3) -> tuple[str, str | None]:
    target = cache / "images" / record.official_split / f"{record.image_id}.jpg"
    record.cached_path = target
    if target.is_file() and target.stat().st_size > 0:
        return record.key, None
    target.parent.mkdir(parents=True, exist_ok=True)
    url = url_template.format(official_split=record.official_split, image_id=record.image_id)
    for attempt in range(1, retries + 1):
        temporary = target.with_suffix(".jpg.part")
        try:
            request = Request(url, headers={"User-Agent": "GolfBallFinder-public-dataset/1.0"})
            with urlopen(request, timeout=90) as response, temporary.open("wb") as file:
                shutil.copyfileobj(response, file, length=1024 * 1024)
            temporary.replace(target)
            return record.key, None
        except (HTTPError, URLError, TimeoutError, OSError) as error:
            temporary.unlink(missing_ok=True)
            if attempt == retries:
                return record.key, str(error)
            time.sleep(attempt)
    return record.key, "unreachable"


def inspect_image(record: SourceImage) -> str | None:
    assert record.cached_path is not None
    try:
        with Image.open(record.cached_path) as image:
            image.verify()
        with Image.open(record.cached_path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            record.width, record.height = image.size
            if record.width < 64 or record.height < 64:
                return "image_too_small"
            record.difference_hash = difference_hash(image)
        record.sha256 = file_sha256(record.cached_path)
    except (OSError, ValueError) as error:
        return f"corrupt_image:{error}"
    return None


def deduplicate(
    records: Iterable[SourceImage],
    max_hamming_distance: int,
) -> tuple[list[SourceImage], list[dict[str, Any]]]:
    ordered = sorted(
        records,
        key=lambda record: (
            0 if record.content_type == "positive" else 1,
            stable_rank(record.key, 0),
        ),
    )
    kept: list[SourceImage] = []
    sha_to_record: dict[str, SourceImage] = {}
    removed: list[dict[str, Any]] = []
    for record in ordered:
        assert record.sha256 is not None and record.difference_hash is not None
        exact = sha_to_record.get(record.sha256)
        if exact is not None:
            removed.append({"removed": record.key, "kept": exact.key, "reason": "exact"})
            continue
        near = next(
            (
                existing
                for existing in kept
                if existing.difference_hash is not None
                and hamming_distance(record.difference_hash, existing.difference_hash) <= max_hamming_distance
            ),
            None,
        )
        if near is not None:
            removed.append(
                {
                    "removed": record.key,
                    "kept": near.key,
                    "reason": "perceptual",
                    "hamming_distance": hamming_distance(record.difference_hash, near.difference_hash),
                }
            )
            continue
        kept.append(record)
        sha_to_record[record.sha256] = record
    return kept, removed


def output_split(key: str, seed: int, train_ratio: float, val_ratio: float) -> str:
    bucket = int(stable_rank(key, seed)[:8], 16) / 0xFFFFFFFF
    if bucket < train_ratio:
        return "train"
    if bucket < train_ratio + val_ratio:
        return "val"
    return "test"


def yolo_label(boxes: Iterable[tuple[float, float, float, float]]) -> str:
    lines = []
    for xmin, ymin, xmax, ymax in boxes:
        width = xmax - xmin
        height = ymax - ymin
        center_x = xmin + width / 2
        center_y = ymin + height / 2
        lines.append(f"0 {center_x:.9f} {center_y:.9f} {width:.9f} {height:.9f}")
    return "\n".join(lines) + ("\n" if lines else "")


def write_dataset(
    records: list[SourceImage],
    output: Path,
    source: dict[str, Any],
    seed: int,
    train_ratio: float,
    val_ratio: float,
    selection_report: dict[str, Any],
    dedupe_report: list[dict[str, Any]],
) -> dict[str, Any]:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output must be absent or empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, str]] = []
    attribution_rows: list[dict[str, str]] = []
    split_counts: dict[str, dict[str, int]] = {
        split: {"positive": 0, "hard_negative": 0, "images": 0} for split in OUTPUT_SPLITS
    }
    for record in records:
        assert record.cached_path is not None
        split = output_split(record.key, seed, train_ratio, val_ratio)
        session_id = f"oi_{record.official_split}_{record.image_id}"
        image_target = output / "images" / split / session_id / f"{record.image_id}.jpg"
        label_target = output / "labels" / split / session_id / f"{record.image_id}.txt"
        image_target.parent.mkdir(parents=True, exist_ok=True)
        label_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(record.cached_path, image_target)
        label_target.write_text(yolo_label(record.boxes), encoding="utf-8")
        manifest_rows.append(
            {
                "session_id": session_id,
                "split": split,
                "source_dir": str(record.cached_path.parent.resolve()),
                "content_type": record.content_type,
            }
        )
        metadata = record.metadata
        attribution_rows.append(
            {
                "session_id": session_id,
                "split": split,
                "content_type": record.content_type,
                "open_images_id": record.image_id,
                "official_split": record.official_split,
                "source_dataset": source["dataset_name"],
                "source_url": source["source_url"],
                "image_url": source["image_url_template"].format(
                    official_split=record.official_split,
                    image_id=record.image_id,
                ),
                "original_url": metadata.get("OriginalURL", ""),
                "original_landing_url": metadata.get("OriginalLandingURL", ""),
                "license": metadata.get("License", ""),
                "author": metadata.get("Author", ""),
                "author_profile_url": metadata.get("AuthorProfileURL", ""),
                "title": metadata.get("Title", ""),
                "negative_classes": "|".join(sorted(record.negative_classes)),
                "sha256": record.sha256 or "",
            }
        )
        split_counts[split][record.content_type] += 1
        split_counts[split]["images"] += 1

    for split in OUTPUT_SPLITS:
        (output / "images" / split).mkdir(parents=True, exist_ok=True)
        (output / "labels" / split).mkdir(parents=True, exist_ok=True)

    dataset_yaml = output / "dataset.yaml"
    dataset_yaml.write_text(
        yaml.safe_dump(
            {
                "path": str(output.resolve()),
                "train": "images/train",
                "val": "images/val",
                "test": "images/test",
                "names": {0: "golf_ball"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    manifest_path = output / "dataset_manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["session_id", "split", "source_dir", "content_type"])
        writer.writeheader()
        writer.writerows(manifest_rows)
    attribution_path = output / "attribution.csv"
    with attribution_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(attribution_rows[0]))
        writer.writeheader()
        writer.writerows(attribution_rows)

    positives = sum(record.content_type == "positive" for record in records)
    negatives = sum(record.content_type == "hard_negative" for record in records)
    source_manifest = {
        "schema_version": 1,
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "sources": [
            {
                "source_id": "open_images_v7",
                "source_url": source["source_url"],
                "dataset_name": source["dataset_name"],
                "license": {
                    "annotations": source["annotation_license"],
                    "images": source["image_license"],
                },
                "downloaded_image_count": selection_report["downloaded_image_count"],
                "used_image_count": len(records),
                "positive_image_count": positives,
                "negative_image_count": negatives,
                "synthetic_image_count": 0,
                "attribution_required": bool(source["attribution_required"]),
                "attribution_file": "attribution.csv",
                "notes": source.get("notes", ""),
            }
        ],
        "split_counts": split_counts,
        "selection": selection_report,
        "deduplication": {
            "removed_count": len(dedupe_report),
            "exact_removed": sum(item["reason"] == "exact" for item in dedupe_report),
            "perceptual_removed": sum(item["reason"] == "perceptual" for item in dedupe_report),
            "details_file": "deduplication_report.json",
        },
    }
    (output / "source_manifest.json").write_text(
        json.dumps(source_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output / "deduplication_report.json").write_text(
        json.dumps(dedupe_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    summary = validate_dataset_config(dataset_yaml, manifest_path)
    source_manifest["validated_dataset_summary"] = asdict(summary)
    (output / "source_manifest.json").write_text(
        json.dumps(source_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return source_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", type=Path, default=Path("training/public_dataset_sources.yaml"))
    parser.add_argument("--cache", type=Path, default=Path(".cache/public_datasets/open_images_v7"))
    parser.add_argument("--output", type=Path, default=Path("training/datasets/public_mvp_v1"))
    parser.add_argument("--max-positives", type=int, default=800)
    parser.add_argument("--max-negatives", type=int, default=500)
    parser.add_argument(
        "--scene-negatives",
        type=int,
        default=150,
        help="Reserve this many hard negatives for human-verified grass/course scenes",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.75)
    parser.add_argument("--val-ratio", type=float, default=0.125)
    parser.add_argument("--near-duplicate-hamming", type=int, default=6)
    parser.add_argument("--download-workers", type=int, default=12)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_positives <= 0 or args.max_negatives <= 0:
        raise SystemExit("max-positive and max-negative counts must be positive")
    if not 0 <= args.scene_negatives <= args.max_negatives:
        raise SystemExit("scene-negatives must be between zero and max-negatives")
    if not (0 < args.train_ratio < 1 and 0 < args.val_ratio < 1 - args.train_ratio):
        raise SystemExit("train/val ratios must leave a non-empty test ratio")
    catalog = yaml.safe_load(args.sources.read_text(encoding="utf-8"))
    source = catalog["sources"]["open_images_v7"]
    args.cache.mkdir(parents=True, exist_ok=True)

    annotation_paths: dict[str, Path] = {}
    metadata_paths: dict[str, Path] = {}
    image_label_paths: dict[str, Path] = {}
    for split in OFFICIAL_SPLITS:
        annotation_paths[split] = args.cache / "metadata" / f"{split}-bbox.csv"
        metadata_paths[split] = args.cache / "metadata" / f"{split}-images.csv"
        download_file(source["annotation_urls"][split], annotation_paths[split])
        download_file(source["image_metadata_urls"][split], metadata_paths[split])
    for split, url in source.get("human_image_label_urls", {}).items():
        image_label_paths[split] = args.cache / "metadata" / f"{split}-human-imagelabels.csv"
        download_file(url, image_label_paths[split])

    negative_mid_to_name = {mid: name for name, mid in source["hard_negative_classes"].items()}
    positives, negative_candidates, scan_report = scan_annotations(
        annotation_paths,
        source["positive_class"]["mid"],
        negative_mid_to_name,
    )
    selected_positives = sorted(
        positives.values(), key=lambda record: stable_rank(record.key, args.seed)
    )[: args.max_positives]
    scene_mid_to_name = {mid: name for name, mid in source.get("scene_negative_classes", {}).items()}
    scene_candidates, image_label_rows = scan_scene_negative_labels(
        image_label_paths,
        source["positive_class"]["mid"],
        scene_mid_to_name,
        set(positives),
    )
    selected_scene_negatives = select_balanced_negatives(
        scene_candidates, args.scene_negatives, args.seed
    )
    scene_keys = {record.key for record in selected_scene_negatives}
    object_candidates = {
        key: classes for key, classes in negative_candidates.items() if key not in scene_keys
    }
    selected_object_negatives = select_balanced_negatives(
        object_candidates, args.max_negatives - len(selected_scene_negatives), args.seed
    )
    selected_negatives = [*selected_scene_negatives, *selected_object_negatives]
    selected = {record.key: record for record in [*selected_positives, *selected_negatives]}
    load_selected_metadata(metadata_paths, selected)

    allowed_licenses = set(source["allowed_image_license_urls"])
    rejected_license = [
        key for key, record in selected.items() if record.metadata.get("License") not in allowed_licenses
    ]
    for key in rejected_license:
        selected.pop(key)

    failures: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=args.download_workers) as executor:
        futures = {
            executor.submit(
                download_image,
                record,
                args.cache,
                source["image_url_template"],
            ): record.key
            for record in selected.values()
        }
        completed = 0
        for future in as_completed(futures):
            key, error = future.result()
            completed += 1
            if error:
                failures[key] = error
            if completed % 100 == 0 or completed == len(futures):
                print(f"images downloaded/available: {completed}/{len(futures)}", flush=True)

    valid: list[SourceImage] = []
    corrupt: dict[str, str] = {}
    for key, record in selected.items():
        if key in failures:
            continue
        error = inspect_image(record)
        if error:
            corrupt[key] = error
            continue
        valid.append(record)
    deduplicated, duplicate_report = deduplicate(valid, args.near_duplicate_hamming)
    selection_report = {
        "positive_candidates": len(positives),
        "hard_negative_candidates": len(negative_candidates),
        "scene_negative_candidates": len(scene_candidates),
        "requested_scene_negative_images": args.scene_negatives,
        "selected_scene_negative_images": len(selected_scene_negatives),
        "selected_object_negative_images": len(selected_object_negatives),
        "requested_positive_images": args.max_positives,
        "requested_negative_images": args.max_negatives,
        "selected_before_license_filter": len(selected_positives) + len(selected_negatives),
        "license_rejected": len(rejected_license),
        "download_failed": len(failures),
        "corrupt_or_too_small": len(corrupt),
        "downloaded_image_count": len(valid),
        "annotation_scan": scan_report,
        "human_image_label_rows": image_label_rows,
        "download_failures": failures,
        "image_validation_failures": corrupt,
    }
    report = write_dataset(
        deduplicated,
        args.output,
        source,
        args.seed,
        args.train_ratio,
        args.val_ratio,
        selection_report,
        duplicate_report,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit(130)

#!/usr/bin/env python3
"""Build a split-safe Build 4 recall dataset from the licensed public MVP data.

Real images remain the majority. Synthetic images are derived only from a ball
crop and a hard-negative background already assigned to the same split. The
manifest records target scale, measured visible fraction, and both parents so
that no source scene can leak across train/val/test through augmentation.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shutil
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image, ImageEnhance, ImageFilter

try:
    from .validate_dataset import image_difference_hash, validate_dataset_config
except ImportError:
    from validate_dataset import image_difference_hash, validate_dataset_config


SPLITS = ("train", "val", "test")
SCENE_NEGATIVE_TOKENS = {"Grass", "Grassland", "Lawn", "Meadow", "Pasture"}
SIZE_SEQUENCE = ("<12",) * 9 + ("12-20",) * 7 + ("20-40",) * 5 + (">40",) * 3
VISIBILITY_SEQUENCE = (0.85, 0.62, 0.40, 0.25)
MIN_BACKGROUND_GRASS_SCORE = 0.45
MIN_PLACEMENT_GRASS_SCORE = 0.40


class UnsuitableBackground(RuntimeError):
    """Raised when a source crop is not sufficiently grass-like for compositing."""


@dataclass(frozen=True)
class BallCrop:
    session_id: str
    split: str
    image_path: Path
    box_xywhn: tuple[float, float, float, float]
    crop_box: tuple[int, int, int, int]


@dataclass(frozen=True)
class Background:
    session_id: str
    split: str
    image_path: Path
    scene_negative: bool


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def session_image(root: Path, split: str, session_id: str) -> Path:
    images = sorted(path for path in (root / "images" / split / session_id).glob("*") if path.is_file())
    if len(images) != 1:
        raise ValueError(f"Expected exactly one image for {session_id}, found {len(images)}")
    return images[0]


def label_path(root: Path, split: str, session_id: str, image: Path) -> Path:
    return root / "labels" / split / session_id / image.with_suffix(".txt").name


def load_boxes(path: Path) -> list[tuple[float, float, float, float]]:
    boxes: list[tuple[float, float, float, float]] = []
    if not path.is_file():
        return boxes
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            _, x, y, width, height = line.split()
            boxes.append((float(x), float(y), float(width), float(height)))
    return boxes


def neutral_ball_score(crop: Image.Image) -> tuple[float, float]:
    pixels = np.asarray(crop.convert("RGB"), dtype=np.float32)
    height, width = pixels.shape[:2]
    yy, xx = np.ogrid[:height, :width]
    ellipse = ((xx - (width - 1) / 2) / max(width / 2, 1)) ** 2 + (
        (yy - (height - 1) / 2) / max(height / 2, 1)
    ) ** 2 <= 1
    selected = pixels[ellipse]
    if not selected.size:
        return 0.0, 0.0
    neutral = (selected.min(axis=1) >= 105) & ((selected.max(axis=1) - selected.min(axis=1)) <= 85)
    return float(neutral.mean()), float(selected.mean())


def collect_ball_crops(root: Path, manifest: list[dict[str, str]]) -> dict[str, list[BallCrop]]:
    collected: dict[str, list[BallCrop]] = {split: [] for split in SPLITS}
    for row in manifest:
        if row["content_type"] != "positive":
            continue
        split = row["split"]
        image_path = session_image(root, split, row["session_id"])
        with Image.open(image_path) as opened:
            image = opened.convert("RGB")
            image_width, image_height = image.size
            for box in load_boxes(label_path(root, split, row["session_id"], image_path)):
                center_x, center_y, width, height = box
                pixel_width = width * image_width
                pixel_height = height * image_height
                aspect = pixel_width / pixel_height
                if min(pixel_width, pixel_height) < 28 or not 0.68 <= aspect <= 1.47:
                    continue
                left = max(0, math.floor((center_x - width / 2) * image_width))
                top = max(0, math.floor((center_y - height / 2) * image_height))
                right = min(image_width, math.ceil((center_x + width / 2) * image_width))
                bottom = min(image_height, math.ceil((center_y + height / 2) * image_height))
                crop = image.crop((left, top, right, bottom))
                neutral_fraction, brightness = neutral_ball_score(crop)
                if neutral_fraction < 0.55 or brightness < 125:
                    continue
                collected[split].append(
                    BallCrop(
                        row["session_id"],
                        split,
                        image_path,
                        box,
                        (left, top, right, bottom),
                    )
                )
    return collected


def collect_backgrounds(
    root: Path,
    manifest: list[dict[str, str]],
    attribution: dict[str, dict[str, str]],
) -> dict[str, list[Background]]:
    collected: dict[str, list[Background]] = {split: [] for split in SPLITS}
    for row in manifest:
        if row["content_type"] != "hard_negative":
            continue
        classes = set(filter(None, attribution.get(row["session_id"], {}).get("negative_classes", "").split("|")))
        collected[row["split"]].append(
            Background(
                row["session_id"],
                row["split"],
                session_image(root, row["split"], row["session_id"]),
                bool(classes & SCENE_NEGATIVE_TOKENS),
            )
        )
    for split in SPLITS:
        collected[split].sort(key=lambda item: (not item.scene_negative, item.session_id))
    return collected


def target_diameter(size_bin: str, rng: random.Random) -> int:
    ranges = {
        "<12": (7, 11),
        "12-20": (12, 20),
        "20-40": (21, 40),
        ">40": (41, 64),
    }
    return rng.randint(*ranges[size_bin])


def grass_statistics(image: Image.Image) -> tuple[float, float, float]:
    pixels = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    red, green, blue = pixels[..., 0], pixels[..., 1], pixels[..., 2]
    chroma = pixels.max(axis=2) - pixels.min(axis=2)
    green_grass = (
        (green > 0.12)
        & (green < 0.92)
        & (chroma > 0.045)
        & (green >= red * 1.02)
        & (green >= blue * 1.08)
    )
    dry_grass = (
        (red > 0.16)
        & (red < 0.88)
        & (green > 0.12)
        & (chroma > 0.035)
        & (green >= blue * 1.08)
        & (red >= green * 0.72)
        & (red <= green * 1.65)
    )
    sky_or_water = (blue > red * 1.12) & (blue > green * 1.05) & (blue > 0.35)
    # Brown roads, wood, skin, and clothing can satisfy a broad "dry grass"
    # color rule. Use genuinely green vegetation for source-crop admission;
    # dry-grass color is added later as a controlled augmentation and remains
    # available here only for reporting.
    score = float(green_grass.mean() - 0.85 * sky_or_water.mean())
    return score, float(green_grass.mean()), float(dry_grass.mean())


def prepare_background(
    background: Background,
    rng: random.Random,
    size: int,
) -> tuple[Image.Image, dict[str, float]]:
    with Image.open(background.image_path) as opened:
        image = opened.convert("RGB")
    width, height = image.size
    short = min(width, height)
    best_crop: Image.Image | None = None
    best_stats = (-1.0, 0.0, 0.0)
    for _ in range(64):
        crop_side = max(64, int(short * rng.uniform(0.18, 0.52)))
        left = rng.randint(0, max(0, width - crop_side))
        top = rng.randint(int(max(0, height - crop_side) * 0.20), max(0, height - crop_side))
        candidate = image.crop((left, top, left + crop_side, top + crop_side))
        stats = grass_statistics(candidate.resize((96, 96), Image.Resampling.BILINEAR))
        if stats[0] > best_stats[0]:
            best_crop = candidate
            best_stats = stats
    assert best_crop is not None
    image = best_crop.resize(
        (size, size), Image.Resampling.LANCZOS
    )
    image = ImageEnhance.Brightness(image).enhance(rng.uniform(0.82, 1.18))
    image = ImageEnhance.Contrast(image).enhance(rng.uniform(0.88, 1.18))
    image = ImageEnhance.Color(image).enhance(rng.uniform(0.78, 1.12))
    return image, {
        "background_grass_score": best_stats[0],
        "background_green_fraction": best_stats[1],
        "background_dry_fraction": best_stats[2],
    }


def choose_ball_position(
    canvas: Image.Image,
    ball_width: int,
    ball_height: int,
    rng: random.Random,
) -> tuple[int, int, float]:
    best = (1, max(1, int(canvas.height * 0.55)), -1.0)
    patch_side = max(36, min(128, max(ball_width, ball_height) * 4))
    for _ in range(48):
        x = rng.randint(1, canvas.width - ball_width - 1)
        y = rng.randint(
            max(1, int(canvas.height * 0.42)),
            canvas.height - ball_height - 1,
        )
        center_x = x + ball_width // 2
        center_y = y + ball_height // 2
        half = patch_side // 2
        patch = canvas.crop(
            (
                max(0, center_x - half),
                max(0, center_y - half),
                min(canvas.width, center_x + half),
                min(canvas.height, center_y + half),
            )
        )
        score, _, _ = grass_statistics(patch)
        if score > best[2]:
            best = (x, y, score)
    return best


def prepare_ball(crop: BallCrop, diameter: int, rng: random.Random) -> tuple[Image.Image, np.ndarray[Any, Any]]:
    with Image.open(crop.image_path) as opened:
        source = opened.convert("RGB").crop(crop.crop_box)
    height = max(4, int(round(diameter * rng.uniform(0.88, 1.06))))
    source = source.resize((diameter, height), Image.Resampling.LANCZOS)
    pixels = np.asarray(source, dtype=np.float32)
    luminance = pixels[..., 0] * 0.299 + pixels[..., 1] * 0.587 + pixels[..., 2] * 0.114
    yy_full, xx_full = np.mgrid[:height, :diameter]
    radial = np.sqrt(
        ((xx_full - diameter * 0.42) / max(diameter / 2, 1)) ** 2
        + ((yy_full - height * 0.35) / max(height / 2, 1)) ** 2
    )
    sphere = 218 + (luminance - luminance.mean()) * 0.32 - np.clip(radial, 0, 1.3) * 32
    sphere *= rng.uniform(0.90, 1.08)
    neutral = np.clip(sphere, 105, 252).astype(np.uint8)
    source = Image.fromarray(np.repeat(neutral[..., None], 3, axis=2), mode="RGB")
    yy, xx = np.ogrid[:height, :diameter]
    base = ((xx - (diameter - 1) / 2) / max(diameter / 2, 1)) ** 2 + (
        (yy - (height - 1) / 2) / max(height / 2, 1)
    ) ** 2 <= 1
    return source, base


def visible_mask(base: np.ndarray[Any, Any], target: float, rng: random.Random) -> np.ndarray[Any, Any]:
    if target >= 0.99:
        return base.copy()
    height, width = base.shape
    yy, xx = np.mgrid[:height, :width]
    phase = rng.uniform(0, math.tau)
    slope = rng.uniform(-0.18, 0.18)
    jagged = 0.10 * height * np.sin(xx / max(width, 1) * math.tau * rng.uniform(1.0, 2.5) + phase)
    score = (yy + slope * (xx - width / 2) + jagged).astype(np.float32)
    # High-score, narrow wedges emulate foreground blades of grass. Because
    # the final cutoff is still a quantile, the requested visible fraction is
    # preserved while the occlusion boundary is no longer a smooth half-moon.
    for _ in range(max(2, width // 12)):
        center = rng.randrange(width)
        tip = rng.randint(max(0, height // 12), max(1, int(height * 0.72)))
        lean = rng.uniform(-0.20, 0.20)
        base_half_width = rng.randint(1, max(1, width // 18))
        for row in range(tip, height):
            progress = (row - tip) / max(1, height - tip)
            row_center = int(round(center + lean * (row - tip)))
            half_width = max(1, int(round(base_half_width * (0.25 + 0.75 * progress))))
            left = max(0, row_center - half_width)
            right = min(width, row_center + half_width + 1)
            score[row, left:right] += height * 2.0
    cutoff = float(np.quantile(score[base], target))
    visible = base & (score <= cutoff)
    return visible


def paste_shadow(image: Image.Image, x: int, y: int, width: int, height: int, rng: random.Random) -> Image.Image:
    shadow = Image.new("L", image.size, 0)
    shadow_pixels = np.zeros((image.height, image.width), dtype=np.uint8)
    yy, xx = np.ogrid[: image.height, : image.width]
    center_x = x + width * rng.uniform(0.45, 0.75)
    center_y = y + height * rng.uniform(0.82, 1.05)
    ellipse = ((xx - center_x) / max(width * 0.65, 1)) ** 2 + (
        (yy - center_y) / max(height * 0.28, 1)
    ) ** 2 <= 1
    shadow_pixels[ellipse] = rng.randint(35, 80)
    shadow = Image.fromarray(shadow_pixels, mode="L").filter(ImageFilter.GaussianBlur(max(1, width / 8)))
    dark = Image.new("RGB", image.size, (12, 15, 8))
    return Image.composite(dark, image, shadow)


def render_synthetic(
    crop: BallCrop,
    background: Background,
    size_bin: str,
    requested_visibility: float,
    rng: random.Random,
    canvas_size: int = 640,
) -> tuple[Image.Image, tuple[float, float, float, float], dict[str, Any]]:
    canvas, background_metadata = prepare_background(background, rng, canvas_size)
    if background_metadata["background_grass_score"] < MIN_BACKGROUND_GRASS_SCORE:
        raise UnsuitableBackground(
            f"background grass score {background_metadata['background_grass_score']:.3f} is below "
            f"{MIN_BACKGROUND_GRASS_SCORE:.3f}"
        )
    diameter = target_diameter(size_bin, rng)
    ball, base_mask = prepare_ball(crop, diameter, rng)
    shown_mask = visible_mask(base_mask, requested_visibility, rng)
    ball_width, ball_height = ball.size

    x, y, placement_grass_score = choose_ball_position(canvas, ball_width, ball_height, rng)
    if placement_grass_score < MIN_PLACEMENT_GRASS_SCORE:
        raise UnsuitableBackground(
            f"placement grass score {placement_grass_score:.3f} is below "
            f"{MIN_PLACEMENT_GRASS_SCORE:.3f}"
        )
    dry_tint = rng.random() < 0.20
    if dry_tint:
        pixels = np.asarray(canvas, dtype=np.float32)
        pixels[..., 0] *= rng.uniform(1.04, 1.16)
        pixels[..., 1] *= rng.uniform(0.82, 0.94)
        pixels[..., 2] *= rng.uniform(0.62, 0.82)
        canvas = Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8), mode="RGB")
    edge_case = rng.random() < 0.10
    if edge_case:
        edge = rng.choice(("left", "right", "bottom"))
        if edge == "left":
            x = -rng.randint(0, max(1, ball_width // 4))
        elif edge == "right":
            x = canvas_size - ball_width + rng.randint(0, max(1, ball_width // 4))
        else:
            y = canvas_size - ball_height + rng.randint(0, max(1, ball_height // 4))

    shadow = rng.random() < 0.45
    if shadow:
        canvas = paste_shadow(canvas, x, y, ball_width, ball_height, rng)

    alpha = (shown_mask.astype(np.uint8) * 255)
    alpha_image = Image.fromarray(alpha, mode="L").filter(ImageFilter.GaussianBlur(0.35))
    canvas.paste(ball, (x, y), alpha_image)

    full_mask = np.zeros((canvas_size, canvas_size), dtype=bool)
    source_left = max(0, -x)
    source_top = max(0, -y)
    destination_left = max(0, x)
    destination_top = max(0, y)
    copy_width = min(ball_width - source_left, canvas_size - destination_left)
    copy_height = min(ball_height - source_top, canvas_size - destination_top)
    if copy_width <= 0 or copy_height <= 0:
        raise ValueError("Synthetic ball was placed fully outside the canvas")
    full_mask[
        destination_top : destination_top + copy_height,
        destination_left : destination_left + copy_width,
    ] = shown_mask[source_top : source_top + copy_height, source_left : source_left + copy_width]
    visible_y, visible_x = np.where(full_mask)
    left = int(visible_x.min())
    right = int(visible_x.max()) + 1
    top = int(visible_y.min())
    bottom = int(visible_y.max()) + 1

    blur_radius = rng.choice((0.0, 0.0, 0.35, 0.55, 0.85))
    if blur_radius:
        canvas = canvas.filter(ImageFilter.GaussianBlur(blur_radius))
    if rng.random() < 0.35:
        pixels = np.asarray(canvas, dtype=np.int16)
        noise = np.random.default_rng(rng.randrange(2**32)).normal(0, rng.uniform(1.5, 5.0), pixels.shape)
        canvas = Image.fromarray(np.clip(pixels + noise, 0, 255).astype(np.uint8), mode="RGB")

    width = right - left
    height = bottom - top
    box = (
        (left + right) / 2 / canvas_size,
        (top + bottom) / 2 / canvas_size,
        width / canvas_size,
        height / canvas_size,
    )
    actual_visibility = float(full_mask.sum() / max(1, base_mask.sum()))
    label_scale = math.sqrt(width * height)
    if requested_visibility < 0.30:
        category = "D_deep_rough"
    elif requested_visibility < 0.75:
        category = "C_partial_occlusion"
    elif label_scale > 40:
        category = "A_near_visible"
    elif dry_tint:
        category = "F_brown_dry_grass"
    elif edge_case:
        category = "G_edge_corner"
    elif shadow:
        category = "E_shadow"
    elif label_scale < 20:
        category = "B_small_distant"
    else:
        category = "synthetic_other"
    metadata = {
        "requested_size_bin": size_bin,
        "requested_ball_diameter_px": diameter,
        "label_bbox_scale_px_640": label_scale,
        "requested_visibility_pct": requested_visibility * 100,
        "ball_visibility_pct": actual_visibility * 100,
        "challenge_category": category,
        "edge_case": edge_case,
        "shadow": shadow,
        "dry_grass_tint": dry_tint,
        "blur_radius": blur_radius,
        "placement_grass_score": placement_grass_score,
        **background_metadata,
    }
    return canvas, box, metadata


def write_label(path: Path, box: tuple[float, float, float, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("0 " + " ".join(f"{value:.9f}" for value in box) + "\n", encoding="utf-8")


def build_dataset(source: Path, output: Path, counts: dict[str, int], seed: int) -> dict[str, Any]:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output must be absent or empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    for folder in ("images", "labels"):
        shutil.copytree(source / folder, output / folder, dirs_exist_ok=True)

    source_manifest_rows = load_csv(source / "dataset_manifest.csv")
    attribution_rows = load_csv(source / "attribution.csv")
    attribution = {row["session_id"]: row for row in attribution_rows}
    crops = collect_ball_crops(source, source_manifest_rows)
    backgrounds = collect_backgrounds(source, source_manifest_rows, attribution)
    if any(not crops[split] for split in SPLITS):
        raise ValueError("Every split needs at least one eligible white/neutral real ball crop")
    if any(not backgrounds[split] for split in SPLITS):
        raise ValueError("Every split needs at least one hard-negative background")

    manifest_fields = [
        "session_id",
        "split",
        "source_dir",
        "content_type",
        "source_type",
        "ball_visibility_pct",
        "label_bbox_scale_px_640",
        "challenge_category",
        "parent_ball_session_id",
        "parent_background_session_id",
        "background_grass_score",
        "placement_grass_score",
    ]
    combined_manifest: list[dict[str, Any]] = []
    for row in source_manifest_rows:
        combined_manifest.append(
            {
                **row,
                "source_type": "real",
                "ball_visibility_pct": "",
                "label_bbox_scale_px_640": "",
                "challenge_category": "H_hard_negative" if row["content_type"] == "hard_negative" else "real_unlabeled",
                "parent_ball_session_id": "",
                "parent_background_session_id": "",
                "background_grass_score": "",
                "placement_grass_score": "",
            }
        )

    attribution_fields = list(attribution_rows[0]) + [
        "derived_ball_session_id",
        "derived_background_session_id",
        "synthetic_recipe",
        "ball_visibility_pct",
    ]
    for row in attribution_rows:
        for field_name in attribution_fields:
            row.setdefault(field_name, "")

    existing_hashes: list[int] = []
    for split in SPLITS:
        for image_path in (output / "images" / split).rglob("*"):
            if image_path.is_file():
                existing_hashes.append(image_difference_hash(image_path))

    generated: list[dict[str, Any]] = []
    for split_index, split in enumerate(SPLITS):
        rng = random.Random(seed + split_index * 100_003)
        split_crops = list(crops[split])
        split_backgrounds = list(backgrounds[split])
        rng.shuffle(split_crops)
        scene_backgrounds = [item for item in split_backgrounds if item.scene_negative]
        other_backgrounds = [item for item in split_backgrounds if not item.scene_negative]
        ordered_backgrounds = scene_backgrounds or other_backgrounds
        rng.shuffle(ordered_backgrounds)
        for index in range(counts[split]):
            crop = split_crops[index % len(split_crops)]
            size_bin = SIZE_SEQUENCE[index % len(SIZE_SEQUENCE)]
            requested_visibility = VISIBILITY_SEQUENCE[(index // len(SIZE_SEQUENCE) + index) % len(VISIBILITY_SEQUENCE)]
            rejected_backgrounds = 0
            for attempt in range(max(60, len(ordered_backgrounds) * 2)):
                background = ordered_backgrounds[(index + attempt) % len(ordered_backgrounds)]
                try:
                    image, box, metadata = render_synthetic(
                        crop,
                        background,
                        size_bin,
                        requested_visibility,
                        rng,
                    )
                except UnsuitableBackground:
                    rejected_backgrounds += 1
                    continue
                temp = output / f".synthetic-check-{split}-{index}.jpg"
                image.save(temp, quality=92, optimize=True)
                candidate_hash = image_difference_hash(temp)
                if all((candidate_hash ^ previous).bit_count() > 6 for previous in existing_hashes):
                    temp.unlink()
                    break
                temp.unlink()
            else:
                raise RuntimeError(
                    f"Could not generate a suitable unique synthetic image for {split}/{index}; "
                    f"rejected {rejected_backgrounds} background attempts"
                )

            session_id = f"syn_{split}_{index:04d}"
            image_target = output / "images" / split / session_id / f"{session_id}.jpg"
            label_target = output / "labels" / split / session_id / f"{session_id}.txt"
            image_target.parent.mkdir(parents=True, exist_ok=True)
            image.save(image_target, quality=92, optimize=True)
            write_label(label_target, box)
            existing_hashes.append(candidate_hash)
            combined_manifest.append(
                {
                    "session_id": session_id,
                    "split": split,
                    "source_dir": str(source.resolve()),
                    "content_type": "positive",
                    "source_type": "synthetic",
                    "ball_visibility_pct": f"{metadata['ball_visibility_pct']:.3f}",
                    "label_bbox_scale_px_640": f"{metadata['label_bbox_scale_px_640']:.3f}",
                    "challenge_category": metadata["challenge_category"],
                    "parent_ball_session_id": crop.session_id,
                    "parent_background_session_id": background.session_id,
                    "background_grass_score": f"{metadata['background_grass_score']:.6f}",
                    "placement_grass_score": f"{metadata['placement_grass_score']:.6f}",
                }
            )
            ball_source = attribution[crop.session_id]
            background_source = attribution[background.session_id]
            attribution_rows.append(
                {
                    "session_id": session_id,
                    "split": split,
                    "content_type": "positive",
                    "open_images_id": "",
                    "official_split": "derived",
                    "source_dataset": "GolfBallFinder Build 4 split-safe synthetic augmentation",
                    "source_url": ball_source.get("source_url", ""),
                    "image_url": "",
                    "original_url": "",
                    "original_landing_url": ball_source.get("original_landing_url", ""),
                    "license": ball_source.get("license", ""),
                    "author": ball_source.get("author", ""),
                    "author_profile_url": ball_source.get("author_profile_url", ""),
                    "title": f"Derived ball from {crop.session_id}; background from {background.session_id}",
                    "negative_classes": background_source.get("negative_classes", ""),
                    "sha256": "",
                    "derived_ball_session_id": crop.session_id,
                    "derived_background_session_id": background.session_id,
                    "synthetic_recipe": "scale+grass_occlusion+lighting+shadow+blur+noise+edge",
                    "ball_visibility_pct": f"{metadata['ball_visibility_pct']:.3f}",
                }
            )
            generated.append({"session_id": session_id, "split": split, **metadata})

    (output / "dataset.yaml").write_text(
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
    write_csv(output / "dataset_manifest.csv", combined_manifest, manifest_fields)
    write_csv(output / "attribution.csv", attribution_rows, attribution_fields)

    original_source_manifest = json.loads((source / "source_manifest.json").read_text(encoding="utf-8"))
    split_counts: dict[str, dict[str, int]] = {}
    for split in SPLITS:
        rows = [row for row in combined_manifest if row["split"] == split]
        split_counts[split] = {
            "real_positive": sum(row["source_type"] == "real" and row["content_type"] == "positive" for row in rows),
            "synthetic_positive": sum(row["source_type"] == "synthetic" for row in rows),
            "hard_negative": sum(row["content_type"] == "hard_negative" for row in rows),
            "images": len(rows),
        }
    source_manifest = {
        "schema_version": 2,
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "base_dataset": str(source.resolve()),
        "sources": [
            *original_source_manifest["sources"],
            {
                "source_id": "build4_split_safe_synthetic",
                "source_url": "internal deterministic builder",
                "dataset_name": "GolfBallFinder Build 4 recall augmentation",
                "license": "Derived images retain both Open Images source image licenses",
                "downloaded_image_count": 0,
                "used_image_count": len(generated),
                "positive_image_count": len(generated),
                "negative_image_count": 0,
                "synthetic_image_count": len(generated),
                "attribution_required": True,
                "attribution_file": "attribution.csv",
                "notes": "Each synthetic row records same-split ball/background parents and measured visibility.",
            },
        ],
        "split_counts": split_counts,
        "eligible_ball_crop_count": {split: len(crops[split]) for split in SPLITS},
        "eligible_background_count": {split: len(backgrounds[split]) for split in SPLITS},
        "deduplication": {
            "strategy": "reject exact/perceptual collisions against real and generated images at dHash distance <= 6",
            "removed_count": 0,
        },
    }
    summary = validate_dataset_config(output / "dataset.yaml", output / "dataset_manifest.csv")
    source_manifest["validated_dataset_summary"] = asdict(summary)
    (output / "source_manifest.json").write_text(
        json.dumps(source_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    report = {
        "schema_version": 1,
        "synthetic_image_count": len(generated),
        "split_counts": Counter(item["split"] for item in generated),
        "requested_size_bins": Counter(item["requested_size_bin"] for item in generated),
        "visibility_bins": Counter(
            "75-100" if item["ball_visibility_pct"] >= 75 else
            "50-75" if item["ball_visibility_pct"] >= 50 else
            "30-50" if item["ball_visibility_pct"] >= 30 else "<30"
            for item in generated
        ),
        "challenge_categories": Counter(item["challenge_category"] for item in generated),
        "split_safety": "ball crop and background parents always use the generated image split",
    }
    (output / "generation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("training/datasets/public_mvp_v3"))
    parser.add_argument("--output", type=Path, default=Path("training/datasets/build4_recall_v1"))
    parser.add_argument("--train-synthetic", type=int, default=240)
    parser.add_argument("--val-synthetic", type=int, default=36)
    parser.add_argument("--test-synthetic", type=int, default=48)
    parser.add_argument("--seed", type=int, default=404)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    counts = {
        "train": args.train_synthetic,
        "val": args.val_synthetic,
        "test": args.test_synthetic,
    }
    if any(value < 0 for value in counts.values()):
        raise SystemExit("Synthetic counts must be non-negative")
    report = build_dataset(args.source.resolve(), args.output.resolve(), counts, args.seed)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

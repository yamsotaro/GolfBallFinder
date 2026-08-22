#!/usr/bin/env python3
"""Reference Color Assist transforms and offline OFF/ON tile-order evaluation.

This mirrors the iOS Core Image formulas. It does not run or replace YOLO and its
tile-rank output is only a scheduler hypothesis, not a detection-accuracy claim.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


MODES = ("raw", "golf_contrast", "excess_green", "white_ball_saliency")
SEARCH_TILES: tuple[tuple[float, float, float, float], ...] = (
    (0.00, 0.00, 0.62, 0.62),
    (0.38, 0.00, 0.62, 0.62),
    (0.00, 0.38, 0.62, 0.62),
    (0.38, 0.38, 0.62, 0.62),
    (0.19, 0.19, 0.62, 0.62),
)
MEAN_WEIGHT = 0.35
PEAK_WEIGHT = 0.65


class ColorAssistError(ValueError):
    pass


@dataclass(frozen=True)
class ColorAssistAnalysis:
    mode: str
    processing_latency_ms: float
    tile_scores: tuple[float, ...]
    selected_tile_order: tuple[int, ...]
    maps: dict[str, np.ndarray]


def smoothstep(edge0: float, edge1: float, value: np.ndarray) -> np.ndarray:
    normalized = np.clip((value - edge0) / (edge1 - edge0), 0.0, 1.0)
    return normalized * normalized * (3.0 - 2.0 * normalized)


def downscale_rgb(image_rgb: np.ndarray, long_side: int = 320, short_side: int = 180) -> np.ndarray:
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ColorAssistError(f"Expected HxWx3 RGB image, got {image_rgb.shape}")
    height, width = image_rgb.shape[:2]
    if height <= 0 or width <= 0:
        raise ColorAssistError("Image dimensions must be positive")
    scale = min(1.0, long_side / max(height, width), short_side / min(height, width))
    resized_width = max(1, round(width * scale))
    resized_height = max(1, round(height * scale))
    if (resized_width, resized_height) == (width, height):
        return image_rgb.astype(np.float32, copy=False) / 255.0 if image_rgb.dtype == np.uint8 else np.clip(
            image_rgb.astype(np.float32, copy=False), 0.0, 1.0
        )
    resized = cv2.resize(image_rgb, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
    return resized.astype(np.float32) / 255.0 if resized.dtype == np.uint8 else np.clip(
        resized.astype(np.float32), 0.0, 1.0
    )


def color_assist_maps(image_rgb: np.ndarray) -> dict[str, np.ndarray]:
    raw = downscale_rgb(image_rgb)
    red, green, blue = (raw[..., index] for index in range(3))
    luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue

    golf = np.empty_like(raw)
    golf[..., 0] = 1.12 * red + 0.04 * luminance
    golf[..., 1] = 0.72 * green
    golf[..., 2] = 1.08 * blue + 0.03 * luminance
    golf = np.clip(golf, 0.0, 1.0)
    golf_luminance = (
        0.2126 * golf[..., 0] + 0.7152 * golf[..., 1] + 0.0722 * golf[..., 2]
    )

    excess_green = np.clip(2.0 * green - red - blue, 0.0, 1.0)
    maximum = raw.max(axis=2)
    minimum = raw.min(axis=2)
    chroma = maximum - minimum
    bright = smoothstep(0.55, 0.90, luminance)
    low_chroma = 1.0 - smoothstep(0.04, 0.24, chroma)
    saliency = np.clip(bright * low_chroma * (1.0 - 0.85 * excess_green), 0.0, 1.0)

    return {
        "raw_rgb": raw,
        "raw": luminance,
        "golf_contrast_rgb": golf,
        "golf_contrast": golf_luminance,
        "excess_green_map": excess_green,
        # Scheduler priority rejects vegetation; the displayed ExG map itself remains green-high.
        "excess_green": 1.0 - excess_green,
        "white_ball_saliency": saliency,
    }


def tile_score(score_map: np.ndarray, tile: tuple[float, float, float, float]) -> float:
    height, width = score_map.shape
    x, y, tile_width, tile_height = tile
    x0 = min(max(math.floor(x * width), 0), width - 1)
    y0 = min(max(math.floor(y * height), 0), height - 1)
    x1 = min(max(math.ceil((x + tile_width) * width), x0 + 1), width)
    y1 = min(max(math.ceil((y + tile_height) * height), y0 + 1), height)
    crop = score_map[y0:y1, x0:x1]
    return float(np.clip(MEAN_WEIGHT * crop.mean() + PEAK_WEIGHT * crop.max(), 0.0, 1.0))


def analyze_rgb(image_rgb: np.ndarray, mode: str = "white_ball_saliency") -> ColorAssistAnalysis:
    if mode not in MODES:
        raise ColorAssistError(f"Unknown mode {mode!r}; expected one of {MODES}")
    started = time.perf_counter()
    maps = color_assist_maps(image_rgb)
    scores = tuple(tile_score(maps[mode], tile) for tile in SEARCH_TILES)
    order = tuple(sorted(range(len(scores)), key=lambda index: (-scores[index], index)))
    return ColorAssistAnalysis(
        mode=mode,
        processing_latency_ms=max(0.0, (time.perf_counter() - started) * 1_000),
        tile_scores=scores,
        selected_tile_order=order,
        maps=maps,
    )


def containing_tiles(row: dict[str, str]) -> tuple[int, ...]:
    annotated = row.get("ball_tile_index", "").strip()
    if annotated:
        try:
            index = int(annotated)
        except ValueError as error:
            raise ColorAssistError(f"Invalid ball_tile_index: {annotated!r}") from error
        if index not in range(len(SEARCH_TILES)):
            raise ColorAssistError(f"ball_tile_index must be 0..{len(SEARCH_TILES) - 1}")
        return (index,)

    required = ("ball_x", "ball_y", "ball_width", "ball_height")
    if not all(row.get(field, "").strip() for field in required):
        return ()
    try:
        x, y, width, height = (float(row[field]) for field in required)
    except ValueError as error:
        raise ColorAssistError("Ball bbox values must be numeric") from error
    center_x = x + width / 2
    center_y = y + height / 2
    if not 0 <= center_x <= 1 or not 0 <= center_y <= 1:
        raise ColorAssistError("Ball bbox center must be inside normalized image coordinates")
    return tuple(
        index
        for index, (tile_x, tile_y, tile_width, tile_height) in enumerate(SEARCH_TILES)
        if tile_x <= center_x <= tile_x + tile_width
        and tile_y <= center_y <= tile_y + tile_height
    )


def rank_for_tiles(order: tuple[int, ...], tiles: tuple[int, ...]) -> int | None:
    return min((order.index(index) + 1 for index in tiles), default=None)


def percentile(values: list[float], percentile_value: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile_value))


def _rank_metrics(ranks: list[int]) -> dict[str, float | int | None]:
    return {
        "annotated_images": len(ranks),
        "mean_rank": float(np.mean(ranks)) if ranks else None,
        "mean_reciprocal_rank": float(np.mean([1.0 / rank for rank in ranks])) if ranks else None,
        "top1_rate": sum(rank <= 1 for rank in ranks) / len(ranks) if ranks else None,
        "top3_rate": sum(rank <= 3 for rank in ranks) / len(ranks) if ranks else None,
    }


def evaluate_manifest(manifest_path: Path, mode: str) -> dict[str, Any]:
    if not manifest_path.is_file():
        raise ColorAssistError(f"Manifest not found: {manifest_path}")
    with manifest_path.open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows or "image_path" not in rows[0]:
        raise ColorAssistError("Manifest requires at least one row and an image_path column")

    baseline = tuple(range(len(SEARCH_TILES)))
    off_ranks: list[int] = []
    on_ranks: list[int] = []
    latencies: list[float] = []
    negative_peak_scores: list[float] = []
    details: list[dict[str, Any]] = []
    for row_number, row in enumerate(rows, start=2):
        image_path = (manifest_path.parent / row["image_path"]).resolve()
        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ColorAssistError(f"Row {row_number}: cannot read image {image_path}")
        analysis = analyze_rgb(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), mode)
        latencies.append(analysis.processing_latency_ms)
        is_positive = row.get("ball_present", "").strip().lower() in {"1", "true", "yes"}
        tiles = containing_tiles(row) if is_positive else ()
        off_rank = rank_for_tiles(baseline, tiles)
        on_rank = rank_for_tiles(analysis.selected_tile_order, tiles)
        if off_rank is not None and on_rank is not None:
            off_ranks.append(off_rank)
            on_ranks.append(on_rank)
        if not is_positive:
            negative_peak_scores.append(max(analysis.tile_scores))
        details.append(
            {
                "image_path": row["image_path"],
                "scene_id": row.get("scene_id") or None,
                "ball_present": is_positive,
                "containing_tiles": list(tiles),
                "off_rank": off_rank,
                "on_rank": on_rank,
                "tile_scores": list(analysis.tile_scores),
                "selected_tile_order": list(analysis.selected_tile_order),
                "processing_latency_ms": analysis.processing_latency_ms,
            }
        )

    return {
        "schema_version": 1,
        "purpose": "tile_order_hypothesis_only_raw_yolo_unchanged",
        "mode": mode,
        "manifest": str(manifest_path),
        "sample_count": len(rows),
        "off": _rank_metrics(off_ranks),
        "on": _rank_metrics(on_ranks),
        "reference_processing_latency_ms_p50": percentile(latencies, 50),
        "reference_processing_latency_ms_p95": percentile(latencies, 95),
        "negative_max_tile_score_p95": percentile(negative_peak_scores, 95),
        "images": details,
    }


def _write_previews(image_path: Path, output_directory: Path) -> None:
    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ColorAssistError(f"Cannot read image: {image_path}")
    maps = color_assist_maps(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    output_directory.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_directory / "raw.png"), cv2.cvtColor(
        np.round(maps["raw_rgb"] * 255).astype(np.uint8), cv2.COLOR_RGB2BGR
    ))
    cv2.imwrite(str(output_directory / "golf_contrast.png"), cv2.cvtColor(
        np.round(maps["golf_contrast_rgb"] * 255).astype(np.uint8), cv2.COLOR_RGB2BGR
    ))
    for name in ("excess_green_map", "white_ball_saliency"):
        cv2.imwrite(str(output_directory / f"{name}.png"), np.round(maps[name] * 255).astype(np.uint8))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, help="CSV field-image manifest for OFF/ON tile-rank evaluation")
    parser.add_argument("--mode", choices=MODES, default="white_ball_saliency")
    parser.add_argument("--output", type=Path, default=Path("color_assist_comparison.json"))
    parser.add_argument("--preview-image", type=Path)
    parser.add_argument("--preview-dir", type=Path)
    args = parser.parse_args()

    if args.preview_image or args.preview_dir:
        if not args.preview_image or not args.preview_dir:
            parser.error("--preview-image and --preview-dir must be supplied together")
        _write_previews(args.preview_image.resolve(), args.preview_dir.resolve())
    if args.manifest:
        payload = evaluate_manifest(args.manifest.resolve(), args.mode)
        rendered = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
        print(rendered, end="")
    elif not args.preview_image:
        parser.error("supply --manifest or the preview arguments")


if __name__ == "__main__":
    main()

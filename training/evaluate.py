#!/usr/bin/env python3
"""Evaluate detector quality, including hard-negative false positives.

Ultralytics mAP is retained for comparability. A deterministic second pass
adds explicit TP/FP/FN counts, confidence sweeps, high-confidence FP details,
and validation-only operating-threshold recommendations.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import yaml

try:
    from .validate_dataset import IMAGE_SUFFIXES, file_sha256, validate_dataset_config
except ImportError:
    from validate_dataset import IMAGE_SUFFIXES, file_sha256, validate_dataset_config


BBox = tuple[float, float, float, float]


@dataclass(frozen=True)
class Detection:
    confidence: float
    bbox_xyxy: BBox


@dataclass
class ImageEvaluation:
    true_positives: int
    false_positives: list[Detection]
    false_negatives: int
    matched_ground_truth: list[tuple[int, Detection]] = field(default_factory=list)
    false_negative_indices: list[int] = field(default_factory=list)


SIZE_BIN_ORDER = ("<12", "12-20", "20-40", ">40")
VISIBILITY_BIN_ORDER = ("75-100", "50-75", "30-50", "<30")

FIVE_TILE_REGIONS: tuple[BBox, ...] = (
    (0.00, 0.00, 0.62, 0.62),
    (0.38, 0.00, 1.00, 0.62),
    (0.00, 0.38, 0.62, 1.00),
    (0.38, 0.38, 1.00, 1.00),
    (0.19, 0.19, 0.81, 0.81),
)


def grid3_regions(tile_side: float = 0.42) -> tuple[BBox, ...]:
    """Return a deterministic overlapping 3x3 cover of the full normalized frame."""
    last = 1.0 - tile_side
    starts = (0.0, last / 2.0, last)
    return tuple((x, y, x + tile_side, y + tile_side) for y in starts for x in starts)


def scan_regions(layout: str) -> tuple[BBox, ...]:
    full: BBox = (0.0, 0.0, 1.0, 1.0)
    if layout == "full":
        return (full,)
    if layout == "full+five":
        return (full, *FIVE_TILE_REGIONS)
    if layout == "full+grid3":
        return (full, *grid3_regions())
    raise ValueError(f"Unsupported scan layout: {layout}")


def scan_invocations_per_cycle(layout: str) -> int:
    """Return app invocations when full-frame and local scans alternate."""
    if layout == "full":
        return 1
    local_count = len(scan_regions(layout)) - 1
    return local_count * 2


def bbox_iou(left: BBox, right: BBox) -> float:
    x1 = max(left[0], right[0])
    y1 = max(left[1], right[1])
    x2 = min(left[2], right[2])
    y2 = min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0


def match_detections(
    ground_truth: list[BBox],
    predictions: Iterable[Detection],
    confidence_threshold: float,
    iou_threshold: float,
) -> ImageEvaluation:
    unmatched = set(range(len(ground_truth)))
    true_positives = 0
    false_positives: list[Detection] = []
    matched_ground_truth: list[tuple[int, Detection]] = []
    for prediction in sorted(predictions, key=lambda item: item.confidence, reverse=True):
        if prediction.confidence < confidence_threshold:
            continue
        matches = sorted(
            ((bbox_iou(prediction.bbox_xyxy, ground_truth[index]), index) for index in unmatched),
            reverse=True,
        )
        if matches and matches[0][0] >= iou_threshold:
            matched_index = matches[0][1]
            unmatched.remove(matched_index)
            true_positives += 1
            matched_ground_truth.append((matched_index, prediction))
        else:
            false_positives.append(prediction)
    return ImageEvaluation(
        true_positives,
        false_positives,
        len(unmatched),
        matched_ground_truth,
        sorted(unmatched),
    )


def aggregate_at_threshold(
    ground_truth: dict[Path, list[BBox]],
    predictions: dict[Path, list[Detection]],
    threshold: float,
    iou_threshold: float,
) -> dict[str, Any]:
    tp = fp = fn = 0
    negative_images_with_fp = 0
    positive_images_with_fn = 0
    for path, boxes in ground_truth.items():
        result = match_detections(boxes, predictions.get(path, []), threshold, iou_threshold)
        tp += result.true_positives
        fp += len(result.false_positives)
        fn += result.false_negatives
        if not boxes and result.false_positives:
            negative_images_with_fp += 1
        if boxes and result.false_negatives:
            positive_images_with_fn += 1
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "threshold": round(threshold, 4),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "negative_images_with_false_positive": negative_images_with_fp,
        "positive_images_with_false_negative": positive_images_with_fn,
    }


def bbox_scale_pixels(box: BBox, imgsz: int) -> float:
    """Equivalent square side at model input, sqrt(width * height) * imgsz."""
    width = max(0.0, box[2] - box[0])
    height = max(0.0, box[3] - box[1])
    return math.sqrt(width * height) * imgsz


def size_bin_for_box(box: BBox, imgsz: int) -> str:
    pixels = bbox_scale_pixels(box, imgsz)
    if pixels < 12:
        return "<12"
    if pixels <= 20:
        return "12-20"
    if pixels <= 40:
        return "20-40"
    return ">40"


def visibility_bin(value: float) -> str:
    ratio = value / 100.0 if value > 1 else value
    if not math.isfinite(ratio) or not 0 <= ratio <= 1:
        raise ValueError(f"Visibility must be finite and within [0, 1] or [0, 100]: {value}")
    if ratio >= 0.75:
        return "75-100"
    if ratio >= 0.50:
        return "50-75"
    if ratio >= 0.30:
        return "30-50"
    return "<30"


def _empty_bin() -> dict[str, Any]:
    return {
        "ground_truth": 0,
        "true_positives": 0,
        "false_positives": 0,
        "false_negatives": 0,
        "precision": 1.0,
        "recall": 1.0,
    }


def _finish_bins(bins: dict[str, dict[str, Any]], order: Iterable[str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key in order:
        values = bins.get(key, _empty_bin())
        tp = values["true_positives"]
        fp = values["false_positives"]
        fn = values["false_negatives"]
        values["precision"] = tp / (tp + fp) if tp + fp else 1.0
        values["recall"] = tp / (tp + fn) if tp + fn else 1.0
        result[key] = values
    return result


def aggregate_by_size(
    ground_truth: dict[Path, list[BBox]],
    predictions: dict[Path, list[Detection]],
    threshold: float,
    iou_threshold: float,
    imgsz: int,
) -> dict[str, Any]:
    bins = {key: _empty_bin() for key in SIZE_BIN_ORDER}
    for path, boxes in ground_truth.items():
        result = match_detections(boxes, predictions.get(path, []), threshold, iou_threshold)
        for index, _ in result.matched_ground_truth:
            key = size_bin_for_box(boxes[index], imgsz)
            bins[key]["ground_truth"] += 1
            bins[key]["true_positives"] += 1
        for index in result.false_negative_indices:
            key = size_bin_for_box(boxes[index], imgsz)
            bins[key]["ground_truth"] += 1
            bins[key]["false_negatives"] += 1
        for detection in result.false_positives:
            key = size_bin_for_box(detection.bbox_xyxy, imgsz)
            bins[key]["false_positives"] += 1
    return {
        "scale_definition": "sqrt(normalized_width * normalized_height) * model_input_size",
        "boundary_definition": "<12, [12,20], (20,40], >40 pixels",
        "bins": _finish_bins(bins, SIZE_BIN_ORDER),
    }


def session_id_for_image(path: Path, root: Path) -> str:
    relative = path.relative_to(root)
    return relative.parts[2] if len(relative.parts) >= 4 else ""


def aggregate_by_manifest_group(
    ground_truth: dict[Path, list[BBox]],
    predictions: dict[Path, list[Detection]],
    root: Path,
    manifest_details: dict[str, dict[str, str]],
    field_name: str,
    group_order: Iterable[str],
    threshold: float,
    iou_threshold: float,
    transform: Any | None = None,
) -> dict[str, Any]:
    bins: dict[str, dict[str, Any]] = {}
    excluded_images = 0
    for path, boxes in ground_truth.items():
        raw = manifest_details.get(session_id_for_image(path, root), {}).get(field_name, "").strip()
        if not raw:
            excluded_images += 1
            continue
        try:
            key = transform(float(raw)) if transform is not None else raw
        except (TypeError, ValueError):
            excluded_images += 1
            continue
        values = bins.setdefault(key, _empty_bin())
        result = match_detections(boxes, predictions.get(path, []), threshold, iou_threshold)
        values["ground_truth"] += len(boxes)
        values["true_positives"] += result.true_positives
        values["false_positives"] += len(result.false_positives)
        values["false_negatives"] += result.false_negatives
    dynamic_order = [*group_order, *sorted(set(bins) - set(group_order))]
    return {
        "field": field_name,
        "excluded_images_without_valid_group": excluded_images,
        "bins": _finish_bins(bins, dynamic_order),
    }


def f_beta(point: dict[str, Any], beta: float) -> float:
    precision = point["precision"]
    recall = point["recall"]
    denominator = beta * beta * precision + recall
    return (1 + beta * beta) * precision * recall / denominator if denominator else 0.0


def recommend_thresholds(points: list[dict[str, Any]]) -> dict[str, Any]:
    candidate_pool = [point for point in points if point["recall"] >= 0.90]
    candidate = max(
        candidate_pool or points,
        key=lambda point: (
            point["precision"] if candidate_pool else f_beta(point, 2.0),
            point["recall"],
            -point["threshold"],
        ),
    )
    beta_gate = [
        point for point in points if point["precision"] >= 0.90 and point["recall"] >= 0.85
    ]
    confirmed = max(
        beta_gate or points,
        key=lambda point: (
            point["recall"] if beta_gate else f_beta(point, 0.5),
            point["precision"],
            -point["false_positives"],
        ),
    )
    candidate_threshold = min(candidate["threshold"], confirmed["threshold"])
    return {
        "method": (
            "precision>=0.90_and_recall>=0.85_gate"
            if beta_gate
            else "fallback_max_f0.5_no_point_met_precision_recall_gate"
        ),
        "model_confidence_threshold": round(max(0.01, candidate_threshold - 0.02), 2),
        "candidate_min_confidence": round(candidate_threshold, 2),
        "confirmed_average_confidence": round(confirmed["threshold"], 2),
        "candidate_point": candidate,
        "confirmed_point": confirmed,
        "warning": "Offline frame metrics do not measure temporal 3-of-5 confirmation; verify on iPhone scenes.",
    }


def dataset_root(data: Path, config: dict[str, Any]) -> Path:
    root = Path(str(config["path"]))
    return root if root.is_absolute() else (data.parent / root).resolve()


def split_images(data: Path, split: str) -> tuple[Path, list[Path]]:
    config = yaml.safe_load(data.read_text(encoding="utf-8"))
    root = dataset_root(data, config)
    configured = Path(str(config[split]))
    image_root = configured if configured.is_absolute() else root / configured
    images = sorted(path.resolve() for path in image_root.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)
    return root, images


def load_ground_truth(root: Path, split: str, images: list[Path]) -> dict[Path, list[BBox]]:
    config = yaml.safe_load((root / "dataset.yaml").read_text(encoding="utf-8"))
    configured = Path(str(config[split]))
    image_root = configured if configured.is_absolute() else root / configured
    truth: dict[Path, list[BBox]] = {}
    for image in images:
        relative = image.relative_to(image_root.resolve())
        label = root / "labels" / split / relative.with_suffix(".txt")
        boxes: list[BBox] = []
        if label.is_file():
            for line in label.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                _, x_text, y_text, width_text, height_text = line.split()
                x = float(x_text)
                y = float(y_text)
                width = float(width_text)
                height = float(height_text)
                boxes.append((x - width / 2, y - height / 2, x + width / 2, y + height / 2))
        truth[image] = boxes
    return truth


def collect_predictions(
    model: Any,
    images: list[Path],
    imgsz: int,
    confidence_floor: float,
    device: str | None,
    scan_layout: str = "full",
) -> dict[Path, list[Detection]]:
    regions = scan_regions(scan_layout)
    if scan_layout != "full":
        return collect_tiled_predictions(
            model,
            images,
            imgsz,
            confidence_floor,
            device,
            regions,
        )
    kwargs: dict[str, Any] = {
        "source": [str(path) for path in images],
        "imgsz": imgsz,
        "conf": confidence_floor,
        "iou": 0.70,
        "max_det": 300,
        "stream": True,
        "verbose": False,
    }
    if device is not None:
        kwargs["device"] = device
    collected: dict[Path, list[Detection]] = {}
    for result in model.predict(**kwargs):
        path = Path(result.path).resolve()
        detections: list[Detection] = []
        if result.boxes is not None:
            xyxyn = result.boxes.xyxyn.cpu().tolist()
            confidence = result.boxes.conf.cpu().tolist()
            detections = [
                Detection(float(score), tuple(float(value) for value in box))
                for box, score in zip(xyxyn, confidence, strict=True)
            ]
        collected[path] = detections
    return collected


def suppress_overlaps(detections: Iterable[Detection], iou_threshold: float) -> list[Detection]:
    kept: list[Detection] = []
    for candidate in sorted(detections, key=lambda item: item.confidence, reverse=True):
        if all(bbox_iou(candidate.bbox_xyxy, existing.bbox_xyxy) < iou_threshold for existing in kept):
            kept.append(candidate)
    return kept


def map_local_box_to_frame(local_box: BBox, region: BBox) -> BBox:
    left, top, right, bottom = region
    width = right - left
    height = bottom - top
    return (
        left + local_box[0] * width,
        top + local_box[1] * height,
        left + local_box[2] * width,
        top + local_box[3] * height,
    )


def collect_tiled_predictions(
    model: Any,
    images: list[Path],
    imgsz: int,
    confidence_floor: float,
    device: str | None,
    regions: tuple[BBox, ...],
) -> dict[Path, list[Detection]]:
    """Run one full scan cycle and merge detector-local boxes in frame coordinates."""
    import numpy as np
    from PIL import Image

    prediction_kwargs: dict[str, Any] = {
        "imgsz": imgsz,
        "conf": confidence_floor,
        "iou": 0.70,
        "max_det": 300,
        "stream": False,
        "verbose": False,
    }
    if device is not None:
        prediction_kwargs["device"] = device

    collected: dict[Path, list[Detection]] = {}
    for path in images:
        with Image.open(path) as opened:
            image = opened.convert("RGB")
            width, height = image.size
            crops: list[np.ndarray[Any, Any]] = []
            for left, top, right, bottom in regions:
                pixel_box = (
                    math.floor(left * width),
                    math.floor(top * height),
                    math.ceil(right * width),
                    math.ceil(bottom * height),
                )
                crops.append(np.asarray(image.crop(pixel_box)))
        results = model.predict(source=crops, **prediction_kwargs)
        detections: list[Detection] = []
        for region, result in zip(regions, results, strict=True):
            if result.boxes is None:
                continue
            for local_box, score in zip(
                result.boxes.xyxyn.cpu().tolist(),
                result.boxes.conf.cpu().tolist(),
                strict=True,
            ):
                mapped = map_local_box_to_frame(
                    tuple(float(value) for value in local_box),
                    region,
                )
                detections.append(Detection(float(score), mapped))
        collected[path.resolve()] = suppress_overlaps(detections, 0.70)
    return collected


def detailed_errors(
    ground_truth: dict[Path, list[BBox]],
    predictions: dict[Path, list[Detection]],
    root: Path,
    threshold: float,
    iou_threshold: float,
    maximum: int = 500,
    attribution: dict[str, dict[str, str]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    false_positives: list[dict[str, Any]] = []
    false_negatives: list[dict[str, Any]] = []
    for path, boxes in ground_truth.items():
        result = match_detections(boxes, predictions.get(path, []), threshold, iou_threshold)
        for detection in result.false_positives:
            relative = path.relative_to(root)
            session_id = relative.parts[2] if len(relative.parts) >= 4 else ""
            source = (attribution or {}).get(session_id, {})
            false_positives.append(
                {
                    "image": str(relative),
                    "content_type": source.get("content_type"),
                    "negative_classes": list(filter(None, source.get("negative_classes", "").split("|"))),
                    "original_landing_url": source.get("original_landing_url"),
                    "confidence": detection.confidence,
                    "bbox_xyxy_normalized": detection.bbox_xyxy,
                    "ground_truth_box_count": len(boxes),
                    "best_iou": max(
                        (bbox_iou(detection.bbox_xyxy, box) for box in boxes),
                        default=0.0,
                    ),
                }
            )
        if result.false_negatives:
            false_negatives.append(
                {
                    "image": str(path.relative_to(root)),
                    "missed_box_count": result.false_negatives,
                    "ground_truth_boxes": boxes,
                }
            )
    false_positives.sort(key=lambda item: item["confidence"], reverse=True)
    return false_positives[:maximum], false_negatives[:maximum]


def load_attribution(path: Path | None) -> dict[str, dict[str, str]]:
    if path is None or not path.is_file():
        return {}
    with path.open(encoding="utf-8", newline="") as file:
        return {row["session_id"]: row for row in csv.DictReader(file)}


def load_manifest_details(path: Path) -> dict[str, dict[str, str]]:
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8", newline="") as file:
        return {row["session_id"]: row for row in csv.DictReader(file)}


def finite_metric(value: Any) -> float | None:
    result = float(value)
    return result if math.isfinite(result) else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data", default="training/dataset.yaml")
    parser.add_argument("--manifest", default="training/dataset_manifest.csv")
    parser.add_argument("--attribution", default=None)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--output", required=True, help="Write reproducible offline metrics JSON")
    parser.add_argument("--device", default=None)
    parser.add_argument("--confidence-floor", type=float, default=0.01)
    parser.add_argument("--operating-threshold", type=float, default=0.20)
    parser.add_argument("--match-iou", type=float, default=0.50)
    parser.add_argument(
        "--scan-layout",
        choices=["full", "full+five", "full+grid3"],
        default="full",
        help="Aggregate a complete broad-search scan cycle before matching",
    )
    parser.add_argument("--recommend-thresholds", action="store_true")
    return parser.parse_args()


def main() -> None:
    config_dir = (Path.cwd() / ".cache" / "ultralytics").resolve()
    config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(config_dir))
    matplotlib_dir = (Path.cwd() / ".cache" / "matplotlib").resolve()
    matplotlib_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_dir))
    from ultralytics import YOLO

    args = parse_args()
    weights = Path(args.weights).resolve()
    data = Path(args.data).resolve()
    manifest = Path(args.manifest).resolve()
    output = Path(args.output).resolve()
    dataset_summary = validate_dataset_config(data, manifest)
    root, images = split_images(data, args.split)
    attribution_path = Path(args.attribution).resolve() if args.attribution else data.parent / "attribution.csv"
    attribution = load_attribution(attribution_path)
    manifest_details = load_manifest_details(manifest)
    ground_truth = load_ground_truth(root, args.split, images)
    model = YOLO(str(weights))
    validation_kwargs: dict[str, Any] = {
        "data": str(data),
        "imgsz": args.imgsz,
        "split": args.split,
        "plots": True,
        "project": str(output.parent),
        "name": f"{output.stem}_ultralytics",
        "exist_ok": True,
    }
    if args.device is not None:
        validation_kwargs["device"] = args.device
    metrics = model.val(**validation_kwargs)
    predictions = collect_predictions(
        model,
        images,
        args.imgsz,
        args.confidence_floor,
        args.device,
        args.scan_layout,
    )
    thresholds = sorted(
        {
            round(args.confidence_floor, 4),
            round(args.operating_threshold, 4),
            *[round(value / 100, 2) for value in range(5, 96)],
        }
    )
    points = [
        aggregate_at_threshold(ground_truth, predictions, threshold, args.match_iou)
        for threshold in thresholds
    ]
    operating = aggregate_at_threshold(ground_truth, predictions, args.operating_threshold, args.match_iou)
    high_confidence = aggregate_at_threshold(ground_truth, predictions, 0.80, args.match_iou)
    false_positives, false_negatives = detailed_errors(
        ground_truth, predictions, root, args.operating_threshold, args.match_iou, attribution=attribution
    )
    high_confidence_false_positives, _ = detailed_errors(
        ground_truth, predictions, root, 0.80, args.match_iou, attribution=attribution
    )
    summary = {
        "schema_version": 2,
        "weights": str(weights),
        "weights_sha256": file_sha256(weights),
        "dataset_yaml_sha256": file_sha256(data),
        "dataset_manifest_sha256": file_sha256(manifest),
        "attribution_sha256": file_sha256(attribution_path) if attribution_path.is_file() else None,
        "split": args.split,
        "imgsz": args.imgsz,
        "scan_layout": args.scan_layout,
        "scan_invocations_per_cycle": scan_invocations_per_cycle(args.scan_layout),
        "offline_unique_scan_regions": len(scan_regions(args.scan_layout)),
        "image_count": len(images),
        "positive_image_count": sum(bool(boxes) for boxes in ground_truth.values()),
        "negative_image_count": sum(not boxes for boxes in ground_truth.values()),
        "ground_truth_box_count": sum(len(boxes) for boxes in ground_truth.values()),
        "dataset_summary": dataset_summary.__dict__,
        "offline_metrics": {
            "precision": finite_metric(metrics.box.mp),
            "recall": finite_metric(metrics.box.mr),
            "map50": finite_metric(metrics.box.map50),
            "map50_95": finite_metric(metrics.box.map),
        },
        "operating_point": operating,
        "high_confidence_0_8": high_confidence,
        "stratified_metrics": {
            "operating_threshold": args.operating_threshold,
            "bbox_size_pixels": aggregate_by_size(
                ground_truth,
                predictions,
                args.operating_threshold,
                args.match_iou,
                args.imgsz,
            ),
            "visibility": aggregate_by_manifest_group(
                ground_truth,
                predictions,
                root,
                manifest_details,
                "ball_visibility_pct",
                VISIBILITY_BIN_ORDER,
                args.operating_threshold,
                args.match_iou,
                visibility_bin,
            ),
            "challenge_category": aggregate_by_manifest_group(
                ground_truth,
                predictions,
                root,
                manifest_details,
                "challenge_category",
                (),
                args.operating_threshold,
                args.match_iou,
            ),
        },
        "threshold_recommendation": recommend_thresholds(points) if args.recommend_thresholds else None,
        "confidence_sweep": points,
        "error_summary": {
            "false_positive_count": operating["false_positives"],
            "false_negative_count": operating["false_negatives"],
            "high_confidence_false_positive_count": high_confidence["false_positives"],
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "high_confidence_false_positives": high_confidence_false_positives,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    output.write_text(rendered, encoding="utf-8")
    console_summary = {
        "output": str(output),
        "weights_sha256": summary["weights_sha256"],
        "split": args.split,
        "offline_metrics": summary["offline_metrics"],
        "operating_point": summary["operating_point"],
        "high_confidence_0_8": summary["high_confidence_0_8"],
        "threshold_recommendation": summary["threshold_recommendation"],
    }
    print(json.dumps(console_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

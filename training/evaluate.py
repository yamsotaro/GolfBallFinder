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
from dataclasses import dataclass
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
    for prediction in sorted(predictions, key=lambda item: item.confidence, reverse=True):
        if prediction.confidence < confidence_threshold:
            continue
        matches = sorted(
            ((bbox_iou(prediction.bbox_xyxy, ground_truth[index]), index) for index in unmatched),
            reverse=True,
        )
        if matches and matches[0][0] >= iou_threshold:
            unmatched.remove(matches[0][1])
            true_positives += 1
        else:
            false_positives.append(prediction)
    return ImageEvaluation(true_positives, false_positives, len(unmatched))


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
        point for point in points if point["precision"] >= 0.90 and point["recall"] >= 0.80
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
            "precision>=0.90_and_recall>=0.80_gate"
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
) -> dict[Path, list[Detection]]:
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
    predictions = collect_predictions(model, images, args.imgsz, args.confidence_floor, args.device)
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

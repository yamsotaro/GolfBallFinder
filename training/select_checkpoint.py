#!/usr/bin/env python3
"""Select a training epoch by validation precision/recall and FP behavior."""
from __future__ import annotations

import argparse
import gc
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any

try:
    from .evaluate import (
        VISIBILITY_BIN_ORDER,
        aggregate_at_threshold,
        aggregate_by_manifest_group,
        aggregate_by_size,
        collect_predictions,
        f_beta,
        load_ground_truth,
        load_manifest_details,
        recommend_thresholds,
        split_images,
        visibility_bin,
    )
    from .validate_dataset import file_sha256, validate_dataset_config
except ImportError:
    from evaluate import (
        VISIBILITY_BIN_ORDER,
        aggregate_at_threshold,
        aggregate_by_manifest_group,
        aggregate_by_size,
        collect_predictions,
        f_beta,
        load_ground_truth,
        load_manifest_details,
        recommend_thresholds,
        split_images,
        visibility_bin,
    )
    from validate_dataset import file_sha256, validate_dataset_config


EPOCH_PATTERN = re.compile(r"epoch(\d+)\.pt$")


def selection_rank(candidate: dict[str, Any]) -> tuple[Any, ...]:
    point = candidate["threshold_recommendation"]["confirmed_point"]
    high_fp = candidate["high_confidence_0_8"]["false_positives"]
    fp = point["false_positives"]
    build4 = candidate.get("build4_gate", {})
    small_recall = build4.get("small_recall", 0.0)
    partial_recall = build4.get("partial_recall", 0.0)
    if (
        point["precision"] >= 0.90
        and point["recall"] >= 0.85
        and small_recall >= 0.80
        and partial_recall >= 0.80
    ):
        return 0, high_fp, fp, -point["recall"], -point["precision"]
    if point["precision"] >= 0.80 and point["recall"] >= 0.60:
        return 1, high_fp, fp, -small_recall, -partial_recall, -point["recall"]
    return 2, -f_beta(point, 0.5), high_fp, fp, -small_recall, -partial_recall, -point["recall"]


def epoch_number(path: Path) -> int:
    match = EPOCH_PATTERN.search(path.name)
    if match is None:
        raise ValueError(f"Not an epoch checkpoint: {path}")
    return int(match.group(1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights-dir", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default=None)
    parser.add_argument("--confidence-floor", type=float, default=0.01)
    parser.add_argument("--match-iou", type=float, default=0.50)
    args = parser.parse_args()

    config_dir = (Path.cwd() / ".cache" / "ultralytics").resolve()
    config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(config_dir))
    from ultralytics import YOLO

    validate_dataset_config(args.data.resolve(), args.manifest.resolve())
    root, images = split_images(args.data.resolve(), "val")
    ground_truth = load_ground_truth(root, "val", images)
    manifest_details = load_manifest_details(args.manifest.resolve())
    checkpoints = sorted(args.weights_dir.glob("epoch*.pt"), key=epoch_number)
    if not checkpoints:
        raise SystemExit(f"No epoch checkpoints found in {args.weights_dir}")
    thresholds = sorted(
        {round(args.confidence_floor, 4), *[round(value / 100, 2) for value in range(5, 96)]}
    )
    candidates: list[dict[str, Any]] = []
    for index, checkpoint in enumerate(checkpoints, start=1):
        print(f"Evaluating validation checkpoint {index}/{len(checkpoints)}: {checkpoint.name}", flush=True)
        model = YOLO(str(checkpoint.resolve()))
        predictions = collect_predictions(model, images, args.imgsz, args.confidence_floor, args.device)
        points = [
            aggregate_at_threshold(ground_truth, predictions, threshold, args.match_iou)
            for threshold in thresholds
        ]
        recommendation = recommend_thresholds(points)
        operating_threshold = recommendation["confirmed_average_confidence"]
        size_metrics = aggregate_by_size(
            ground_truth,
            predictions,
            operating_threshold,
            args.match_iou,
            args.imgsz,
        )
        visibility_metrics = aggregate_by_manifest_group(
            ground_truth,
            predictions,
            root,
            manifest_details,
            "ball_visibility_pct",
            VISIBILITY_BIN_ORDER,
            operating_threshold,
            args.match_iou,
            visibility_bin,
        )
        small_recall = size_metrics["bins"].get("<12", {}).get("recall", 0.0)
        partial_bins = [
            visibility_metrics["bins"].get(name, {})
            for name in ("50-75", "30-50")
        ]
        partial_true_positives = sum(item.get("true_positives", 0) for item in partial_bins)
        partial_false_negatives = sum(item.get("false_negatives", 0) for item in partial_bins)
        partial_total = partial_true_positives + partial_false_negatives
        partial_recall = partial_true_positives / partial_total if partial_total else 0.0
        candidate = {
            "epoch_index": epoch_number(checkpoint),
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": file_sha256(checkpoint),
            "threshold_recommendation": recommendation,
            "high_confidence_0_8": aggregate_at_threshold(
                ground_truth, predictions, 0.80, args.match_iou
            ),
            "build4_gate": {
                "small_recall": small_recall,
                "partial_recall": partial_recall,
                "size_metrics": size_metrics,
                "visibility_metrics": visibility_metrics,
            },
        }
        candidates.append(candidate)
        del predictions, model
        gc.collect()

    selected = min(candidates, key=selection_rank)
    args.output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(selected["checkpoint"], args.output_checkpoint)
    selected_hash = file_sha256(args.output_checkpoint)
    if selected_hash != selected["checkpoint_sha256"]:
        raise SystemExit("Selected checkpoint copy failed SHA256 verification")
    report = {
        "schema_version": 1,
        "selection_policy": [
            "meet precision>=0.90, recall>=0.85, <12px recall>=0.80, and partial recall>=0.80",
            "otherwise meet precision>=0.80 and recall>=0.60",
            "otherwise maximize validation F0.5 while retaining small/partial recall as tie-breakers",
            "within a tier minimize confidence>=0.8 FP, then total FP",
        ],
        "dataset_yaml_sha256": file_sha256(args.data),
        "dataset_manifest_sha256": file_sha256(args.manifest),
        "validation_image_count": len(images),
        "selected": {
            **selected,
            "release_checkpoint": str(args.output_checkpoint.resolve()),
            "release_checkpoint_sha256": selected_hash,
        },
        "candidates": candidates,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    args.report.write_text(rendered, encoding="utf-8")
    print(json.dumps(report["selected"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

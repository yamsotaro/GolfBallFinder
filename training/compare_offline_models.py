#!/usr/bin/env python3
"""Compare seed and candidate detectors on the exact same held-out split."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


class OfflineComparisonError(RuntimeError):
    pass


def load_report(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 2:
        raise OfflineComparisonError(f"Unsupported evaluation report: {path}")
    return payload


def compare_reports(seed: dict[str, Any], new: dict[str, Any]) -> dict[str, Any]:
    identity_fields = (
        "dataset_yaml_sha256",
        "dataset_manifest_sha256",
        "split",
        "imgsz",
        "scan_layout",
    )
    mismatched = [field for field in identity_fields if seed.get(field) != new.get(field)]
    if mismatched:
        raise OfflineComparisonError(f"Reports are not comparable; mismatched: {mismatched}")
    seed_threshold = seed["operating_point"]["threshold"]
    new_threshold = new["operating_point"]["threshold"]
    if seed_threshold != new_threshold:
        raise OfflineComparisonError(
            f"Operating thresholds differ: seed={seed_threshold} new={new_threshold}"
        )

    def model_summary(report: dict[str, Any]) -> dict[str, Any]:
        operating = report["operating_point"]
        high = report["high_confidence_0_8"]
        return {
            "weights": report["weights"],
            "weights_sha256": report["weights_sha256"],
            "ultralytics_metrics": report["offline_metrics"],
            "operating_point": operating,
            "false_positive_count": operating["false_positives"],
            "false_negative_count": operating["false_negatives"],
            "high_confidence_false_positive_count": high["false_positives"],
        }

    seed_summary = model_summary(seed)
    new_summary = model_summary(new)
    seed_operating = seed_summary["operating_point"]
    new_operating = new_summary["operating_point"]
    seed_high_fp = seed_summary["high_confidence_false_positive_count"]
    new_high_fp = new_summary["high_confidence_false_positive_count"]
    high_fp_reduction = seed_high_fp - new_high_fp
    new_size_bins = (
        new.get("stratified_metrics", {})
        .get("bbox_size_pixels", {})
        .get("bins", {})
    )
    new_visibility_bins = (
        new.get("stratified_metrics", {})
        .get("visibility", {})
        .get("bins", {})
    )
    small_recall = new_size_bins.get("<12", {}).get("recall")
    partial_values = [
        new_visibility_bins.get(key, {})
        for key in ("50-75", "30-50")
        if new_visibility_bins.get(key, {}).get("ground_truth", 0)
    ]
    partial_tp = sum(item.get("true_positives", 0) for item in partial_values)
    partial_fn = sum(item.get("false_negatives", 0) for item in partial_values)
    partial_recall = partial_tp / (partial_tp + partial_fn) if partial_tp + partial_fn else None
    return {
        "schema_version": 1,
        "dataset_yaml_sha256": seed["dataset_yaml_sha256"],
        "dataset_manifest_sha256": seed["dataset_manifest_sha256"],
        "split": seed["split"],
        "imgsz": seed["imgsz"],
        "scan_layout": seed.get("scan_layout", "full"),
        "operating_threshold": seed_threshold,
        "seed": seed_summary,
        "new": new_summary,
        "delta_new_minus_seed": {
            "precision": new_operating["precision"] - seed_operating["precision"],
            "recall": new_operating["recall"] - seed_operating["recall"],
            "map50": new["offline_metrics"]["map50"] - seed["offline_metrics"]["map50"],
            "map50_95": new["offline_metrics"]["map50_95"] - seed["offline_metrics"]["map50_95"],
            "false_positive_count": new_operating["false_positives"] - seed_operating["false_positives"],
            "high_confidence_false_positive_count": new_high_fp - seed_high_fp,
        },
        "false_positive_reduction": {
            "count": seed_operating["false_positives"] - new_operating["false_positives"],
            "fraction": (
                (seed_operating["false_positives"] - new_operating["false_positives"])
                / seed_operating["false_positives"]
                if seed_operating["false_positives"]
                else 0.0
            ),
            "high_confidence_count": high_fp_reduction,
            "high_confidence_fraction": high_fp_reduction / seed_high_fp if seed_high_fp else 0.0,
        },
        "mvp_gate": {
            "target_precision": 0.90,
            "target_recall": 0.85,
            "target_small_recall": 0.80,
            "target_partial_recall": 0.80,
            "precision_pass": new_operating["precision"] >= 0.90,
            "recall_pass": new_operating["recall"] >= 0.85,
            "small_recall": small_recall,
            "small_recall_pass": small_recall is not None and small_recall >= 0.80,
            "partial_recall": partial_recall,
            "partial_recall_pass": partial_recall is not None and partial_recall >= 0.80,
            "numeric_gate_pass": (
                new_operating["precision"] >= 0.90
                and new_operating["recall"] >= 0.85
                and small_recall is not None
                and small_recall >= 0.80
                and partial_recall is not None
                and partial_recall >= 0.80
            ),
            "warning": "Dataset bias and temporal confirmation are not measured by this numeric gate.",
        },
        "remaining_high_confidence_false_positives": new["error_summary"][
            "high_confidence_false_positives"
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=Path, required=True)
    parser.add_argument("--new", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        comparison = compare_reports(load_report(args.seed), load_report(args.new))
    except OfflineComparisonError as error:
        raise SystemExit(str(error)) from error
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(comparison, ensure_ascii=False, indent=2) + "\n"
    args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()

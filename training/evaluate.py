#!/usr/bin/env python3
"""Run detector validation and print the metrics that matter for model iteration."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ultralytics import YOLO

try:
    from .validate_dataset import file_sha256, validate_dataset_config
except ImportError:
    from validate_dataset import file_sha256, validate_dataset_config


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--data", default="training/dataset.yaml")
    p.add_argument("--manifest", default="training/dataset_manifest.csv")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--split", choices=["val", "test"], default="test")
    p.add_argument("--output", help="Write reproducible offline metrics JSON")
    args = p.parse_args()

    weights = Path(args.weights).resolve()
    data = Path(args.data).resolve()
    manifest = Path(args.manifest).resolve()
    dataset_summary = validate_dataset_config(data, manifest)
    model = YOLO(str(weights))
    metrics = model.val(data=str(data), imgsz=args.imgsz, split=args.split, plots=True)
    summary = {
        "schema_version": 1,
        "weights": str(weights),
        "weights_sha256": file_sha256(weights),
        "dataset_yaml_sha256": file_sha256(data),
        "dataset_manifest_sha256": file_sha256(manifest),
        "split": args.split,
        "imgsz": args.imgsz,
        "dataset_summary": dataset_summary.__dict__,
        "offline_metrics": {
            "precision": float(metrics.box.mp),
            "recall": float(metrics.box.mr),
            "map50": float(metrics.box.map50),
            "map50_95": float(metrics.box.map),
        },
    }
    rendered = json.dumps(summary, indent=2) + "\n"
    print(rendered, end="")
    if args.output:
        Path(args.output).write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run detector validation and print the metrics that matter for model iteration."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--data", default="training/dataset.yaml")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--split", choices=["val", "test"], default="test")
    args = p.parse_args()

    model = YOLO(args.weights)
    metrics = model.val(data=args.data, imgsz=args.imgsz, split=args.split, plots=True)
    summary = {
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

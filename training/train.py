#!/usr/bin/env python3
"""Fine-tune a one-class golf-ball detector for rough/grass search."""
from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="training/dataset.yaml")
    p.add_argument("--base", default="yolo26n.pt", help="Base checkpoint; yolo11n.pt is a fallback.")
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", default=None, help="Examples: 0, mps, cpu. Omit for Ultralytics auto-selection.")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--name", default="golf_ball_yolo26n")
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data = Path(args.data)
    if not data.exists():
        raise SystemExit(f"Dataset config not found: {data}. Copy training/dataset.yaml.example first.")

    model = YOLO(args.base)
    kwargs = dict(
        data=str(data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        project="runs/golfball",
        name=args.name,
        pretrained=True,
        patience=30,
        close_mosaic=10,
        # Small-object / outdoor robustness. Revisit from evidence, not intuition.
        mosaic=0.8,
        scale=0.55,
        translate=0.10,
        fliplr=0.5,
        hsv_h=0.015,
        hsv_s=0.45,
        hsv_v=0.35,
        degrees=5.0,
        perspective=0.0002,
        cache=False,
        plots=True,
        save=True,
        val=True,
        resume=args.resume,
    )
    if args.device is not None:
        kwargs["device"] = args.device

    model.train(**kwargs)


if __name__ == "__main__":
    main()

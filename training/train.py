#!/usr/bin/env python3
"""Fine-tune a one-class golf-ball detector for rough/grass search."""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

try:
    from .validate_dataset import file_sha256, validate_dataset_config
except ImportError:
    from validate_dataset import file_sha256, validate_dataset_config


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="training/dataset.yaml")
    p.add_argument("--manifest", default="training/dataset_manifest.csv")
    p.add_argument("--base", default="yolo26n.pt", help="Base checkpoint; yolo11n.pt is a fallback.")
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", default=None, help="Examples: 0, mps, cpu. Omit for Ultralytics auto-selection.")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--project", default="runs/golfball")
    p.add_argument("--name", default="golf_ball_yolo26n")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--save-period", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--close-mosaic", type=int, default=10)
    p.add_argument("--mosaic", type=float, default=0.8)
    p.add_argument("--scale", type=float, default=0.55)
    p.add_argument("--translate", type=float, default=0.10)
    p.add_argument("--degrees", type=float, default=5.0)
    p.add_argument("--perspective", type=float, default=0.0002)
    p.add_argument("--box", type=float, default=7.5)
    p.add_argument("--cls", type=float, default=0.5)
    p.add_argument("--dfl", type=float, default=1.5)
    p.add_argument(
        "--multi-scale",
        type=float,
        default=0.0,
        help="Ultralytics multi-scale range; 0 disables it.",
    )
    return p.parse_args()


def main() -> None:
    config_dir = (Path.cwd() / ".cache" / "ultralytics").resolve()
    config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(config_dir))
    matplotlib_dir = (Path.cwd() / ".cache" / "matplotlib").resolve()
    matplotlib_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_dir))
    from ultralytics import YOLO

    args = parse_args()
    data = Path(args.data)
    if not data.exists():
        raise SystemExit(f"Dataset config not found: {data}. Copy training/dataset.yaml.example first.")
    manifest = Path(args.manifest)
    summary = validate_dataset_config(data.resolve(), manifest.resolve())
    print(f"Validated session-split dataset: {summary}")

    model = YOLO(args.base)
    kwargs = dict(
        data=str(data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        project=str(Path(args.project).resolve()),
        name=args.name,
        pretrained=True,
        patience=args.patience,
        close_mosaic=args.close_mosaic,
        # Small-object / outdoor robustness. Revisit from evidence, not intuition.
        mosaic=args.mosaic,
        scale=args.scale,
        translate=args.translate,
        fliplr=0.5,
        hsv_h=0.015,
        hsv_s=0.45,
        hsv_v=0.35,
        degrees=args.degrees,
        perspective=args.perspective,
        box=args.box,
        cls=args.cls,
        dfl=args.dfl,
        multi_scale=args.multi_scale,
        cache=False,
        plots=True,
        save=True,
        save_period=args.save_period,
        val=True,
        resume=args.resume,
        seed=args.seed,
        deterministic=True,
    )
    if args.device is not None:
        kwargs["device"] = args.device

    started_at_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    started = time.perf_counter()
    result = model.train(**kwargs)
    training_seconds = time.perf_counter() - started
    completed_at_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    save_dir = Path(getattr(result, "save_dir", Path("runs/golfball") / args.name))
    save_dir.mkdir(parents=True, exist_ok=True)
    base_path = Path(args.base)
    best_checkpoint = save_dir / "weights" / "best.pt"
    provenance = {
        "schema_version": 2,
        "base": args.base,
        "base_sha256": file_sha256(base_path) if base_path.is_file() else None,
        "dataset_yaml": str(data.resolve()),
        "dataset_yaml_sha256": file_sha256(data),
        "dataset_manifest": str(manifest.resolve()),
        "dataset_manifest_sha256": file_sha256(manifest),
        "seed": args.seed,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "project": str(Path(args.project).resolve()),
        "save_period": args.save_period,
        "patience": args.patience,
        "device": args.device,
        "started_at_utc": started_at_utc,
        "completed_at_utc": completed_at_utc,
        "training_seconds": training_seconds,
        "best_checkpoint": str(best_checkpoint.resolve()) if best_checkpoint.is_file() else None,
        "best_checkpoint_sha256": file_sha256(best_checkpoint) if best_checkpoint.is_file() else None,
        "optimizer": "auto",
        "augmentation": {
            "mosaic": args.mosaic,
            "close_mosaic": args.close_mosaic,
            "scale": args.scale,
            "translate": args.translate,
            "fliplr": 0.5,
            "hsv_h": 0.015,
            "hsv_s": 0.45,
            "hsv_v": 0.35,
            "degrees": args.degrees,
            "perspective": args.perspective,
            "box": args.box,
            "cls": args.cls,
            "dfl": args.dfl,
            "multi_scale": args.multi_scale,
        },
        "dataset_summary": summary.__dict__,
    }
    (save_dir / "training_provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()

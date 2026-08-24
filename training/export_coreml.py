#!/usr/bin/env python3
"""Export a trained Ultralytics detector to GolfBall.mlpackage and place it in the iOS app."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

from ultralytics import YOLO

try:
    from .coreml_spec import (
        inspect_coreml_package,
        raw_yolo_output_contract,
        validate_raw_yolo_detection_contract,
    )
except ImportError:  # Direct `python training/export_coreml.py` execution.
    from coreml_spec import (
        inspect_coreml_package,
        raw_yolo_output_contract,
        validate_raw_yolo_detection_contract,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True, help="Path to .pt weights")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--int8", action="store_true", help="Try Core ML int8/palettized export; benchmark against FP16.")
    p.add_argument("--output", default="GolfBallFinder/Resources/GolfBall.mlpackage")
    p.add_argument("--manifest", default="GolfBallFinder/Resources/ModelManifest.json")
    return p.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    weights = Path(args.weights).resolve()
    if not weights.exists():
        raise SystemExit(f"Weights not found: {weights}")

    model = YOLO(str(weights))
    export_kwargs = dict(format="coreml", imgsz=args.imgsz, nms=False)
    if args.int8:
        export_kwargs["int8"] = True
    else:
        export_kwargs["half"] = True

    exported = Path(model.export(**export_kwargs)).resolve()
    if exported.suffix != ".mlpackage":
        raise SystemExit(f"Expected .mlpackage, got: {exported}")

    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if exported != output:
        if output.is_dir():
            shutil.rmtree(output)
        elif output.exists():
            output.unlink()
        shutil.copytree(exported, output)

    coreml_spec = inspect_coreml_package(output)
    validate_raw_yolo_detection_contract(
        coreml_spec,
        expected_input_size=args.imgsz,
        expected_class_count=len(model.names),
    )

    manifest = {
        "schema_version": 2,
        "model_resource": output.name,
        "checkpoint_filename": weights.name,
        "checkpoint_sha256": sha256(weights),
        "task": "detect",
        "class_names": model.names,
        "input_size": [args.imgsz, args.imgsz],
        "export_precision": "INT8" if args.int8 else "FP16",
        "coreml_nms_embedded": False,
        "coreml_spec": coreml_spec,
        "output_contract": raw_yolo_output_contract(len(model.names)),
        "ultralytics_version": version("ultralytics"),
        "coremltools_version": version("coremltools"),
        "source_revision": os.environ.get("CM_COMMIT") or os.environ.get("GITHUB_SHA"),
        "exported_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path = Path(args.manifest).resolve()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Core ML package copied to: {output}")
    print(f"Model manifest written to: {manifest_path}")
    print("Next: run `xcodegen generate`, open GolfBallFinder.xcodeproj, select your iPhone, and Run.")


if __name__ == "__main__":
    main()

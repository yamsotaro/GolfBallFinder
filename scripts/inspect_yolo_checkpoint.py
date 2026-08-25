#!/usr/bin/env python3
"""Validate the one-class raw YOLO checkpoint contract before Core ML export."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class CheckpointContractError(RuntimeError):
    pass


def validate_contract(names: dict[int, str], shape: list[int], input_size: int) -> None:
    if names != {0: "golf_ball"}:
        raise CheckpointContractError(f"Expected exactly class 0 golf_ball, got: {names}")
    expected_anchors = sum((input_size // stride) ** 2 for stride in (8, 16, 32))
    expected = [1, 5, expected_anchors]
    if shape != expected:
        raise CheckpointContractError(f"Expected raw checkpoint output {expected}, got: {shape}")


def raw_output_tensor(output: Any) -> Any:
    value = output
    while isinstance(value, (tuple, list)):
        if not value:
            raise CheckpointContractError("Model returned an empty output container")
        value = value[0]
    if not hasattr(value, "shape"):
        raise CheckpointContractError(f"Model returned a non-tensor output: {type(value).__name__}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    config_dir = (Path.cwd() / ".cache" / "ultralytics").resolve()
    config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(config_dir))
    import torch
    from ultralytics import YOLO

    from scripts.fetch_release_model import sha256

    weights_label = args.weights.as_posix()
    weights = args.weights.resolve()
    model = YOLO(str(weights))
    model.model.eval()
    with torch.inference_mode():
        output = model.model(torch.zeros(1, 3, args.imgsz, args.imgsz))
    tensor = raw_output_tensor(output)
    shape = [int(value) for value in tensor.shape]
    names = {int(key): str(value) for key, value in model.names.items()}
    validate_contract(names, shape, args.imgsz)
    report = {
        "weights": weights_label,
        "weights_sha256": sha256(weights),
        "task": model.task,
        "class_names": names,
        "input": {"name": "image", "color_space": "RGB", "shape": [1, 3, args.imgsz, args.imgsz]},
        "raw_output_shape": shape,
        "raw_layout": "[batch, center_x/center_y/width/height + class_confidence, anchors]",
        "independent_objectness": False,
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    try:
        main()
    except CheckpointContractError as error:
        raise SystemExit(str(error)) from error

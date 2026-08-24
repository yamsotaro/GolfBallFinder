#!/usr/bin/env python3
"""Print and validate the actual GolfBall.mlpackage specification."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.coreml_spec import (  # noqa: E402
    CoreMLContractError,
    inspect_coreml_package,
    raw_yolo_output_contract,
    validate_raw_yolo_detection_contract,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("GolfBallFinder/Resources/GolfBall.mlpackage"),
    )
    parser.add_argument("--expected-input-size", type=int, default=640)
    parser.add_argument("--expected-class-count", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        spec = inspect_coreml_package(args.model.resolve())
        validate_raw_yolo_detection_contract(
            spec,
            expected_input_size=args.expected_input_size,
            expected_class_count=args.expected_class_count,
        )
    except CoreMLContractError as error:
        raise SystemExit(str(error)) from error

    report = {
        "coreml_spec": spec,
        "output_contract": raw_yolo_output_contract(args.expected_class_count),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

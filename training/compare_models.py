#!/usr/bin/env python3
"""Compare device field evaluations using discovery/false-alert/latency KPIs, not mAP alone."""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class ComparisonError(ValueError):
    pass


@dataclass(frozen=True)
class ModelResult:
    model_id: str
    protocol_id: str
    dataset_manifest_sha256: str
    device: str
    discovery_rate_10s: float
    false_alerts_per_min: float
    latency_p50_ms: float | None
    source: str


def load_result(path: Path) -> ModelResult:
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    metrics = payload.get("metrics")
    if not isinstance(metrics, dict):
        raise ComparisonError(f"{path}: metrics object is required")
    try:
        result = ModelResult(
            model_id=str(payload["model_id"]),
            protocol_id=str(payload["evaluation_protocol_id"]),
            dataset_manifest_sha256=str(payload["dataset_manifest_sha256"]),
            device=str(payload["device"]),
            discovery_rate_10s=float(metrics["scene_discovery_rate_10s"]),
            false_alerts_per_min=float(metrics["false_confirmed_alerts_per_min"]),
            latency_p50_ms=(
                None
                if metrics["detection_latency_ms_p50"] is None
                else float(metrics["detection_latency_ms_p50"])
            ),
            source=str(path),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ComparisonError(f"{path}: missing or invalid comparison field: {error}") from error
    if not 0 <= result.discovery_rate_10s <= 1:
        raise ComparisonError(f"{path}: scene_discovery_rate_10s must be within [0, 1]")
    finite_values = [result.discovery_rate_10s, result.false_alerts_per_min]
    if result.latency_p50_ms is not None:
        finite_values.append(result.latency_p50_ms)
    if not all(math.isfinite(value) for value in finite_values):
        raise ComparisonError(f"{path}: comparison metrics must be finite")
    if result.false_alerts_per_min < 0 or (
        result.latency_p50_ms is not None and result.latency_p50_ms < 0
    ):
        raise ComparisonError(f"{path}: false alerts and latency must be non-negative")
    if len(result.dataset_manifest_sha256) != 64 or any(
        character not in "0123456789abcdefABCDEF" for character in result.dataset_manifest_sha256
    ):
        raise ComparisonError(f"{path}: dataset_manifest_sha256 must be 64 hexadecimal characters")
    return result


def compare_results(results: list[ModelResult]) -> list[ModelResult]:
    if len(results) < 2:
        raise ComparisonError("At least two model results are required")
    protocols = {(item.protocol_id, item.dataset_manifest_sha256, item.device) for item in results}
    if len(protocols) != 1:
        raise ComparisonError(
            "Results are not comparable: evaluation_protocol_id, dataset manifest hash, and device must match"
        )
    return sorted(
        results,
        key=lambda item: (
            -item.discovery_rate_10s,
            item.false_alerts_per_min,
            item.latency_p50_ms if item.latency_p50_ms is not None else math.inf,
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", nargs="+")
    parser.add_argument("--output", default="model_comparison.json")
    args = parser.parse_args()
    ranked = compare_results([load_result(Path(item)) for item in args.results])
    payload = {
        "ranking_policy": [
            "scene_discovery_rate_10s_desc",
            "false_confirmed_alerts_per_min_asc",
            "detection_latency_ms_p50_asc",
        ],
        "results": [item.__dict__ for item in ranked],
    }
    rendered = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    Path(args.output).write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()

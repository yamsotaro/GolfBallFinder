#!/usr/bin/env python3
"""Convert app Field Diagnostics JSONL sessions into a comparable field-evaluation JSON."""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    from .validate_dataset import file_sha256
except ImportError:
    from validate_dataset import file_sha256


class FieldLogError(ValueError):
    pass


def parse_timestamp(value: str) -> datetime:
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (TypeError, ValueError) as error:
        raise FieldLogError(f"Invalid ISO-8601 timestamp: {value!r}") from error


def percentile(values: list[float], percentile_value: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(percentile_value / 100 * len(ordered)) - 1)
    return ordered[index]


def load_events(paths: list[Path]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for path in paths:
        if not path.is_file():
            raise FieldLogError(f"Field log not found: {path}")
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as error:
                raise FieldLogError(f"{path}:{line_number}: invalid JSON") from error
            required = {"eventID", "sessionID", "timestamp", "kind"}
            missing = required - set(event)
            if missing:
                raise FieldLogError(f"{path}:{line_number}: missing fields {sorted(missing)}")
            if event["eventID"] in seen_ids:
                raise FieldLogError(f"Duplicate eventID across logs: {event['eventID']}")
            seen_ids.add(event["eventID"])
            event["_parsed_timestamp"] = parse_timestamp(event["timestamp"])
            events.append(event)
    return sorted(events, key=lambda event: event["_parsed_timestamp"])


def summarize_events(
    events: list[dict[str, Any]],
    *,
    model_id: str,
    protocol_id: str,
    dataset_manifest_sha256: str,
    device: str,
) -> dict[str, Any]:
    scenes: dict[tuple[str, str], dict[str, Any]] = {}
    checkpoint_hashes = {
        value for event in events if (value := event.get("modelCheckpointSHA256"))
    }
    if len(checkpoint_hashes) > 1:
        raise FieldLogError(f"Logs contain multiple model checkpoint hashes: {sorted(checkpoint_hashes)}")

    for event in events:
        scene_id = event.get("sceneID")
        if not scene_id:
            continue
        key = (str(event["sessionID"]), str(scene_id))
        scene = scenes.setdefault(key, {"events": []})
        scene["events"].append(event)
        if event["kind"] == "scene_start":
            if "start" in scene:
                raise FieldLogError(f"Scene has multiple starts: {key}")
            scene["start"] = event["_parsed_timestamp"]
            scene["type"] = event.get("sceneType")
            scene["occlusion"] = event.get("occlusion") or "unknown"
        elif event["kind"] == "scene_end":
            scene["end"] = event["_parsed_timestamp"]

    incomplete = [key for key, scene in scenes.items() if "start" not in scene or "end" not in scene]
    if incomplete:
        raise FieldLogError(f"Scenes require exactly one start and end event: {incomplete}")

    positive_scenes = [scene for scene in scenes.values() if scene.get("type") == "positive"]
    negative_scenes = [scene for scene in scenes.values() if scene.get("type") == "negative"]
    if not positive_scenes:
        raise FieldLogError("No completed positive scenes")
    if not negative_scenes:
        raise FieldLogError("No completed negative scenes")

    successful_latencies: list[float] = []
    candidate_latencies: list[float] = []
    success_by_occlusion: dict[str, list[bool]] = defaultdict(list)
    successful_scenes = 0
    for scene in positive_scenes:
        true_events = [event for event in scene["events"] if event["kind"] == "true_positive"]
        latency_values = [
            float(event["sceneStartToConfirmedMs"])
            for event in true_events
            if event.get("sceneStartToConfirmedMs") is not None
        ]
        latency = min(latency_values) if latency_values else None
        success = latency is not None and latency <= 10_000
        success_by_occlusion[scene["occlusion"]].append(success)
        if success:
            successful_scenes += 1
            successful_latencies.append(latency)
            candidate_latencies.extend(
                float(event["candidateToConfirmedMs"])
                for event in true_events
                if event.get("candidateToConfirmedMs") is not None
                and event.get("sceneStartToConfirmedMs") is not None
                and float(event["sceneStartToConfirmedMs"]) == latency
            )

    negative_seconds = sum(
        (scene["end"] - scene["start"]).total_seconds() for scene in negative_scenes
    )
    if negative_seconds <= 0:
        raise FieldLogError("Negative scene duration must be positive")
    negative_events = [event for scene in negative_scenes for event in scene["events"]]
    rejected_confirmed = sum(
        event["kind"] == "false_positive" and event.get("detectionState") == "found"
        for event in negative_events
    )
    rejected_all = sum(event["kind"] == "false_positive" for event in negative_events)

    inference_samples = [event for event in events if event["kind"] == "inference_sample"]
    inference_latencies = [
        float(event["inferenceLatencyMs"])
        for event in inference_samples
        if event.get("inferenceLatencyMs") is not None
    ]
    effective_fps = [
        float(event["effectiveInferenceFPS"])
        for event in inference_samples
        if event.get("effectiveInferenceFPS") is not None
    ]
    thermal_seconds: dict[str, float] = {key: 0.0 for key in ("nominal", "fair", "serious", "critical", "unknown")}
    samples_by_session: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in inference_samples:
        samples_by_session[str(sample["sessionID"])].append(sample)
    for samples in samples_by_session.values():
        for index, sample in enumerate(samples):
            if index + 1 < len(samples):
                delta = (samples[index + 1]["_parsed_timestamp"] - sample["_parsed_timestamp"]).total_seconds()
                duration = min(max(delta, 0), 1.0)
            else:
                duration = 0.5
            state = sample.get("thermalState", "unknown")
            thermal_seconds[state if state in thermal_seconds else "unknown"] += duration

    metrics = {
        "scene_discovery_rate_10s": successful_scenes / len(positive_scenes),
        "false_confirmed_alerts_per_min": rejected_confirmed / (negative_seconds / 60),
        "detection_latency_ms_p50": percentile(successful_latencies, 50),
        "detection_latency_ms_p90": percentile(successful_latencies, 90),
        "candidate_to_confirmed_ms_p50": percentile(candidate_latencies, 50),
        "occlusion_success_rate": {
            bucket: sum(outcomes) / len(outcomes) for bucket, outcomes in sorted(success_by_occlusion.items())
        },
        "thermal": {
            "duration_minutes": sum(thermal_seconds.values()) / 60,
            **{f"{state}_seconds": round(seconds, 3) for state, seconds in thermal_seconds.items()},
        },
        "inference_latency_ms_p50": percentile(inference_latencies, 50),
        "inference_latency_ms_p95": percentile(inference_latencies, 95),
        "effective_inference_fps_p50": percentile(effective_fps, 50),
    }
    return {
        "schema_version": 1,
        "model_id": model_id,
        "evaluation_protocol_id": protocol_id,
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "device": device,
        "model_checkpoint_sha256": next(iter(checkpoint_hashes)) if len(checkpoint_hashes) == 1 else None,
        "sample_counts": {
            "positive_scenes": len(positive_scenes),
            "successful_positive_scenes": successful_scenes,
            "negative_scenes": len(negative_scenes),
            "negative_scan_minutes": negative_seconds / 60,
            "rejected_confirmed_alerts": rejected_confirmed,
            "rejected_candidate_or_confirmed_events": rejected_all,
            "inference_samples": len(inference_samples),
        },
        "metrics": metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("logs", nargs="+")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--protocol-id", required=True)
    parser.add_argument("--dataset-manifest", required=True)
    parser.add_argument("--device", default="iPhone 16 Pro")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    manifest = Path(args.dataset_manifest).resolve()
    payload = summarize_events(
        load_events([Path(item).resolve() for item in args.logs]),
        model_id=args.model_id,
        protocol_id=args.protocol_id,
        dataset_manifest_sha256=file_sha256(manifest),
        device=args.device,
    )
    rendered = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    Path(args.output).write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()

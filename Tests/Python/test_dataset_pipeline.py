from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import yaml
import cv2
import numpy as np

from training.color_assist import analyze_rgb, evaluate_manifest
from training.compare_models import ComparisonError, compare_results, load_result
from training.prepare_dataset import prepare_dataset
from training.summarize_field_logs import FieldLogError, load_events, summarize_events
from training.validate_dataset import DatasetValidationError, validate_dataset_config


class ColorAssistReferenceTests(unittest.TestCase):
    def test_white_ball_saliency_prioritizes_annotated_grass_tile(self) -> None:
        image = np.zeros((180, 320, 3), dtype=np.uint8)
        image[..., 1] = 150
        cv2.circle(image, (32, 18), 6, (255, 255, 255), thickness=-1)

        analysis = analyze_rgb(image, "white_ball_saliency")

        self.assertEqual(analysis.maps["raw_rgb"].shape, (180, 320, 3))
        self.assertEqual(analysis.selected_tile_order[0], 0)
        self.assertGreater(analysis.tile_scores[0], analysis.tile_scores[3])

    def test_manifest_compares_round_robin_and_saliency_rank(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = np.zeros((180, 320, 3), dtype=np.uint8)
            image[..., 1] = 150
            cv2.circle(image, (288, 18), 6, (255, 255, 255), thickness=-1)
            image_path = root / "field.png"
            cv2.imwrite(str(image_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            manifest = root / "manifest.csv"
            with manifest.open("w", encoding="utf-8", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=(
                    "image_path", "scene_id", "ball_present", "ball_x", "ball_y",
                    "ball_width", "ball_height", "ball_tile_index",
                ))
                writer.writeheader()
                writer.writerow({
                    "image_path": image_path.name,
                    "scene_id": "scene-1",
                    "ball_present": "true",
                    "ball_x": "0.88",
                    "ball_y": "0.07",
                    "ball_width": "0.04",
                    "ball_height": "0.04",
                    "ball_tile_index": "1",
                })

            payload = evaluate_manifest(manifest, "white_ball_saliency")

        self.assertEqual(payload["off"]["mean_rank"], 2)
        self.assertEqual(payload["on"]["mean_rank"], 1)
        self.assertEqual(payload["purpose"], "tile_order_hypothesis_only_raw_yolo_unchanged")


class DatasetFixture:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.dataset = root / "dataset"
        self.manifest = root / "manifest.csv"
        self.config = root / "dataset.yaml"

    def create(self) -> None:
        rows = []
        for index, (session, split, content_type) in enumerate(
            (
                ("train_session", "train", "mixed"),
                ("val_hard_negative", "val", "hard_negative"),
                ("test_session", "test", "positive"),
            )
        ):
            image = self.dataset / "images" / split / session / "frame.jpg"
            label = self.dataset / "labels" / split / session / "frame.txt"
            image.parent.mkdir(parents=True, exist_ok=True)
            label.parent.mkdir(parents=True, exist_ok=True)
            image.write_bytes(f"unique-image-{index}".encode())
            label.write_text("" if content_type == "hard_negative" else "0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
            rows.append(
                {
                    "session_id": session,
                    "split": split,
                    "source_dir": f"source/{session}",
                    "content_type": content_type,
                }
            )
        with self.manifest.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        self.config.write_text(
            yaml.safe_dump(
                {
                    "path": str(self.dataset),
                    "train": "images/train",
                    "val": "images/val",
                    "test": "images/test",
                    "names": {0: "golf_ball"},
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )


class DatasetValidationTests(unittest.TestCase):
    def test_validates_session_split_and_counts_hard_negatives(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = DatasetFixture(Path(directory))
            fixture.create()
            summary = validate_dataset_config(fixture.config, fixture.manifest)
            self.assertEqual(summary.sessions, 3)
            self.assertEqual(summary.images, 3)
            self.assertEqual(summary.positive_images, 2)
            self.assertEqual(summary.hard_negative_images, 1)

    def test_rejects_exact_image_duplicate_across_splits(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = DatasetFixture(Path(directory))
            fixture.create()
            train = fixture.dataset / "images" / "train" / "train_session" / "frame.jpg"
            test = fixture.dataset / "images" / "test" / "test_session" / "frame.jpg"
            test.write_bytes(train.read_bytes())
            with self.assertRaisesRegex(DatasetValidationError, "duplicate crosses splits"):
                validate_dataset_config(fixture.config, fixture.manifest)

    def test_rejects_hard_negative_with_box(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = DatasetFixture(Path(directory))
            fixture.create()
            label = fixture.dataset / "labels" / "val" / "val_hard_negative" / "frame.txt"
            label.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
            with self.assertRaisesRegex(DatasetValidationError, "Hard-negative"):
                validate_dataset_config(fixture.config, fixture.manifest)

    def test_rejects_box_crossing_image_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = DatasetFixture(Path(directory))
            fixture.create()
            label = fixture.dataset / "labels" / "test" / "test_session" / "frame.txt"
            label.write_text("0 0.95 0.5 0.2 0.2\n", encoding="utf-8")
            with self.assertRaisesRegex(DatasetValidationError, "horizontal image boundary"):
                validate_dataset_config(fixture.config, fixture.manifest)


class DatasetPreparationTests(unittest.TestCase):
    def test_prepares_manifest_sessions_without_random_frame_split(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = []
            for index, (session, split, content_type) in enumerate(
                (
                    ("source_train", "train", "mixed"),
                    ("source_val", "val", "hard_negative"),
                    ("source_test", "test", "positive"),
                )
            ):
                source = root / session
                source.mkdir()
                (source / "frame.jpg").write_bytes(f"source-{index}".encode())
                if content_type != "hard_negative":
                    (source / "frame.txt").write_text("0 0.5 0.5 0.1 0.1\n", encoding="utf-8")
                rows.append(
                    {
                        "session_id": session,
                        "split": split,
                        "source_dir": session,
                        "content_type": content_type,
                    }
                )
            manifest = root / "manifest.csv"
            with manifest.open("w", encoding="utf-8", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)

            output = root / "prepared"
            self.assertEqual(prepare_dataset(manifest, output), 3)
            self.assertTrue((output / "images" / "val" / "source_val" / "frame.jpg").is_file())
            self.assertEqual(
                (output / "labels" / "val" / "source_val" / "frame.txt").read_text(encoding="utf-8"),
                "",
            )


class ModelComparisonTests(unittest.TestCase):
    def _write_result(self, path: Path, model: str, discovery: float, false_alerts: float, latency: float) -> None:
        path.write_text(
            json.dumps(
                {
                    "model_id": model,
                    "evaluation_protocol_id": "field-v1",
                    "dataset_manifest_sha256": "a" * 64,
                    "device": "iPhone 16 Pro",
                    "metrics": {
                        "scene_discovery_rate_10s": discovery,
                        "false_confirmed_alerts_per_min": false_alerts,
                        "detection_latency_ms_p50": latency,
                    },
                }
            ),
            encoding="utf-8",
        )

    def test_ranks_field_discovery_before_offline_style_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first.json"
            second = root / "second.json"
            self._write_result(first, "model-a", 0.80, 0.1, 200)
            self._write_result(second, "model-b", 0.90, 0.5, 250)
            ranked = compare_results([load_result(first), load_result(second)])
            self.assertEqual(ranked[0].model_id, "model-b")

    def test_rejects_different_field_protocols(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first.json"
            second = root / "second.json"
            self._write_result(first, "model-a", 0.8, 0.1, 200)
            self._write_result(second, "model-b", 0.9, 0.2, 220)
            payload = json.loads(second.read_text(encoding="utf-8"))
            payload["evaluation_protocol_id"] = "field-v2"
            second.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ComparisonError, "not comparable"):
                compare_results([load_result(first), load_result(second)])


class FieldLogSummaryTests(unittest.TestCase):
    def _event(
        self,
        event_id: str,
        timestamp: str,
        kind: str,
        scene_id: str | None = None,
        scene_type: str | None = None,
        **extra: object,
    ) -> dict[str, object]:
        return {
            "eventID": event_id,
            "sessionID": "session-1",
            "sceneID": scene_id,
            "sceneType": scene_type,
            "timestamp": timestamp,
            "kind": kind,
            "thermalState": "nominal",
            **extra,
        }

    def test_summarizes_human_verified_field_kpis(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.jsonl"
            events = [
                self._event("1", "2026-08-22T00:00:00Z", "scene_start", "positive-1", "positive", occlusion="visible_50"),
                self._event(
                    "2",
                    "2026-08-22T00:00:08Z",
                    "true_positive",
                    "positive-1",
                    "positive",
                    sceneStartToConfirmedMs=8000,
                    sceneStartToFirstCandidateMs=7750,
                    candidateToConfirmedMs=250,
                    ballContainingTileRank=1,
                    detectionState="found",
                ),
                self._event("3", "2026-08-22T00:00:10Z", "scene_end", "positive-1", "positive"),
                self._event("4", "2026-08-22T00:01:00Z", "scene_start", "negative-1", "negative"),
                self._event(
                    "5",
                    "2026-08-22T00:01:20Z",
                    "false_positive",
                    "negative-1",
                    "negative",
                    detectionState="found",
                ),
                self._event("6", "2026-08-22T00:02:00Z", "scene_end", "negative-1", "negative"),
                self._event(
                    "7",
                    "2026-08-22T00:00:01Z",
                    "inference_sample",
                    "positive-1",
                    "positive",
                    inferenceLatencyMs=20,
                    effectiveInferenceFPS=18,
                    colorAssistRequested=True,
                    colorAssistEnabled=True,
                    colorAssistFilterMode="white_ball_saliency",
                    colorProcessingLatencyMs=2,
                ),
                self._event(
                    "8",
                    "2026-08-22T00:00:01.5Z",
                    "inference_sample",
                    "positive-1",
                    "positive",
                    inferenceLatencyMs=30,
                    effectiveInferenceFPS=17,
                    colorAssistRequested=True,
                    colorAssistEnabled=True,
                    colorAssistFilterMode="white_ball_saliency",
                    colorProcessingLatencyMs=4,
                ),
            ]
            path.write_text("\n".join(json.dumps(event) for event in events) + "\n", encoding="utf-8")
            payload = summarize_events(
                load_events([path]),
                model_id="model-a",
                protocol_id="field-v1",
                dataset_manifest_sha256="a" * 64,
                device="iPhone 16 Pro",
            )
            self.assertEqual(payload["metrics"]["scene_discovery_rate_10s"], 1)
            self.assertEqual(payload["metrics"]["false_confirmed_alerts_per_min"], 1)
            self.assertEqual(payload["metrics"]["detection_latency_ms_p50"], 8000)
            self.assertEqual(payload["metrics"]["time_to_first_candidate_ms_p50"], 7750)
            self.assertEqual(payload["metrics"]["candidate_to_confirmed_ms_p50"], 250)
            self.assertEqual(payload["metrics"]["ball_containing_tile_rank_p50"], 1)
            self.assertEqual(payload["metrics"]["color_processing_latency_ms_p50"], 2)
            self.assertEqual(payload["metrics"]["occlusion_success_rate"]["visible_50"], 1)
            self.assertEqual(
                payload["color_assist_configurations"],
                [{"requested": True, "filter_mode": "white_ball_saliency"}],
            )

    def test_rejects_mixed_color_assist_arms(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.jsonl"
            events = [
                self._event("1", "2026-08-22T00:00:00Z", "scene_start", "positive-1", "positive"),
                self._event(
                    "2", "2026-08-22T00:00:01Z", "inference_sample", "positive-1", "positive",
                    colorAssistRequested=False, colorAssistFilterMode="white_ball_saliency",
                ),
                self._event(
                    "3", "2026-08-22T00:00:02Z", "inference_sample", "positive-1", "positive",
                    colorAssistRequested=True, colorAssistFilterMode="white_ball_saliency",
                ),
                self._event("4", "2026-08-22T00:00:10Z", "scene_end", "positive-1", "positive"),
                self._event("5", "2026-08-22T00:01:00Z", "scene_start", "negative-1", "negative"),
                self._event("6", "2026-08-22T00:02:00Z", "scene_end", "negative-1", "negative"),
            ]
            path.write_text("\n".join(json.dumps(event) for event in events) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(FieldLogError, "summarize each A/B arm separately"):
                summarize_events(
                    load_events([path]),
                    model_id="model-a",
                    protocol_id="field-v1",
                    dataset_manifest_sha256="a" * 64,
                    device="iPhone 16 Pro",
                )

    def test_rejects_incomplete_scenes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.jsonl"
            path.write_text(
                json.dumps(self._event("1", "2026-08-22T00:00:00Z", "scene_start", "positive-1", "positive"))
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(FieldLogError, "start and end"):
                summarize_events(
                    load_events([path]),
                    model_id="model-a",
                    protocol_id="field-v1",
                    dataset_manifest_sha256="a" * 64,
                    device="iPhone 16 Pro",
                )


if __name__ == "__main__":
    unittest.main()

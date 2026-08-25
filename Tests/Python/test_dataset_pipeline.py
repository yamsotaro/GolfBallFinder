from __future__ import annotations

import csv
import json
import random
import tempfile
import unittest
from pathlib import Path

import yaml
import cv2
import numpy as np
from PIL import Image

from training.build_public_dataset import (
    SourceImage,
    deduplicate,
    output_split,
    scan_scene_negative_labels,
)
from training.build_recall_dataset import (
    Background,
    BallCrop,
    grass_statistics,
    render_synthetic,
    visible_mask,
)
from training.color_assist import analyze_rgb, evaluate_manifest
from training.compare_models import ComparisonError, compare_results, load_result
from training.compare_offline_models import OfflineComparisonError, compare_reports
from training.evaluate import (
    Detection,
    aggregate_at_threshold,
    aggregate_by_size,
    grid3_regions,
    map_local_box_to_frame,
    match_detections,
    recommend_thresholds,
    scan_invocations_per_cycle,
    scan_regions,
    suppress_overlaps,
    visibility_bin,
)
from training.select_checkpoint import selection_rank
from training.prepare_dataset import prepare_dataset
from training.summarize_field_logs import FieldLogError, load_events, summarize_events
from training.validate_dataset import DatasetValidationError, validate_dataset_config


def write_test_image(path: Path, seed: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pixels = np.random.default_rng(seed).integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
    Image.fromarray(pixels, mode="RGB").save(path, quality=95)


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
            label.parent.mkdir(parents=True, exist_ok=True)
            write_test_image(image, index)
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
            with self.assertRaisesRegex(DatasetValidationError, "Exact image duplicate"):
                validate_dataset_config(fixture.config, fixture.manifest)

    def test_rejects_perceptual_near_duplicate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = DatasetFixture(Path(directory))
            fixture.create()
            train = fixture.dataset / "images" / "train" / "train_session" / "frame.jpg"
            test = fixture.dataset / "images" / "test" / "test_session" / "frame.jpg"
            with Image.open(train) as image:
                pixels = np.array(image.convert("RGB"))
            pixels[-1, -1] = (pixels[-1, -1] + 1) % 255
            Image.fromarray(pixels, mode="RGB").save(test, quality=96)
            self.assertNotEqual(train.read_bytes(), test.read_bytes())
            with self.assertRaisesRegex(DatasetValidationError, "Perceptual near duplicate"):
                validate_dataset_config(fixture.config, fixture.manifest)

    def test_rejects_corrupt_image(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = DatasetFixture(Path(directory))
            fixture.create()
            image = fixture.dataset / "images" / "val" / "val_hard_negative" / "frame.jpg"
            image.write_bytes(b"not-a-decodable-image")
            with self.assertRaisesRegex(DatasetValidationError, "Corrupt or undecodable image"):
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
                write_test_image(source / "frame.jpg", 10 + index)
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


class PublicDatasetBuilderTests(unittest.TestCase):
    def test_scene_negatives_require_positive_human_label_and_exclude_known_golf_balls(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            labels = Path(directory) / "validation.csv"
            with labels.open("w", encoding="utf-8", newline="") as file:
                writer = csv.DictWriter(
                    file, fieldnames=("ImageID", "Source", "LabelName", "Confidence")
                )
                writer.writeheader()
                writer.writerows(
                    (
                        {"ImageID": "grass", "Source": "verification", "LabelName": "/grass", "Confidence": "1"},
                        {"ImageID": "absent", "Source": "verification", "LabelName": "/grass", "Confidence": "0"},
                        {"ImageID": "golf", "Source": "verification", "LabelName": "/grass", "Confidence": "1"},
                        {"ImageID": "golf", "Source": "verification", "LabelName": "/ball", "Confidence": "1"},
                        {"ImageID": "bbox", "Source": "verification", "LabelName": "/grass", "Confidence": "1"},
                    )
                )

            candidates, row_counts = scan_scene_negative_labels(
                {"validation": labels},
                positive_mid="/ball",
                scene_mid_to_name={"/grass": "Grass"},
                bbox_positive_keys={"validation/bbox"},
            )

        self.assertEqual(candidates, {"validation/grass": {"Grass"}})
        self.assertEqual(row_counts, {"validation": 5})

    def test_deduplicate_prefers_positive_and_removes_exact_and_near_duplicates(self) -> None:
        positive = SourceImage("train/positive", "train", "positive", "positive")
        exact_negative = SourceImage("train/exact", "train", "exact", "hard_negative")
        near_negative = SourceImage("test/near", "test", "near", "hard_negative")
        unique_negative = SourceImage("test/unique", "test", "unique", "hard_negative")
        positive.sha256 = exact_negative.sha256 = "a" * 64
        positive.difference_hash = exact_negative.difference_hash = 0b101010
        near_negative.sha256 = "b" * 64
        near_negative.difference_hash = 0b101011
        unique_negative.sha256 = "c" * 64
        unique_negative.difference_hash = (1 << 200) | 0b010101

        kept, removed = deduplicate(
            [exact_negative, near_negative, unique_negative, positive],
            max_hamming_distance=1,
        )

        self.assertEqual({record.key for record in kept}, {positive.key, unique_negative.key})
        self.assertEqual({item["reason"] for item in removed}, {"exact", "perceptual"})

    def test_output_split_is_deterministic_and_session_key_based(self) -> None:
        keys = [f"train/image-{index}" for index in range(100)]
        first = [output_split(key, 42, 0.75, 0.125) for key in keys]
        second = [output_split(key, 42, 0.75, 0.125) for key in keys]
        self.assertEqual(first, second)
        self.assertEqual(set(first), {"train", "val", "test"})


class RecallDatasetBuilderTests(unittest.TestCase):
    def test_grass_score_rejects_neutral_asphalt_like_surface(self) -> None:
        asphalt = np.full((96, 96, 3), 105, dtype=np.uint8)
        asphalt += np.random.default_rng(12).integers(0, 18, asphalt.shape, dtype=np.uint8)
        grass = np.zeros((96, 96, 3), dtype=np.uint8)
        grass[..., 0] = 65
        grass[..., 1] = 125
        grass[..., 2] = 45

        asphalt_score, _, _ = grass_statistics(Image.fromarray(asphalt))
        grass_score, _, _ = grass_statistics(Image.fromarray(grass))

        self.assertLess(asphalt_score, 0.45)
        self.assertGreater(grass_score, 0.90)

    def test_occlusion_mask_tracks_requested_visible_fraction(self) -> None:
        yy, xx = np.ogrid[:80, :80]
        base = (xx - 39.5) ** 2 + (yy - 39.5) ** 2 <= 38**2
        shown = visible_mask(base, 0.40, random.Random(7))
        self.assertAlmostEqual(shown.sum() / base.sum(), 0.40, delta=0.02)

    def test_synthetic_box_is_finite_bounded_and_measured(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            ball_path = root / "ball.jpg"
            background_path = root / "grass.jpg"
            ball = np.full((96, 96, 3), 225, dtype=np.uint8)
            cv2.circle(ball, (48, 48), 38, (250, 250, 250), thickness=-1)
            Image.fromarray(ball).save(ball_path)
            grass = np.random.default_rng(10).integers(0, 35, size=(300, 300, 3), dtype=np.uint8)
            grass[..., 1] += 95
            Image.fromarray(grass).save(background_path)
            crop = BallCrop("ball-parent", "train", ball_path, (0.5, 0.5, 1.0, 1.0), (0, 0, 96, 96))
            background = Background("grass-parent", "train", background_path, True)

            _, box, metadata = render_synthetic(
                crop,
                background,
                "<12",
                0.40,
                random.Random(11),
            )

        self.assertTrue(all(np.isfinite(value) for value in box))
        self.assertTrue(all(0 < value <= 1 for value in box))
        self.assertAlmostEqual(metadata["ball_visibility_pct"], 40, delta=4)
        self.assertEqual(metadata["requested_size_bin"], "<12")


class OfflineDetectionEvaluationTests(unittest.TestCase):
    def test_scan_cycle_counts_match_alternating_full_local_scheduler(self) -> None:
        self.assertEqual(scan_invocations_per_cycle("full"), 1)
        self.assertEqual(scan_invocations_per_cycle("full+five"), 10)
        self.assertEqual(scan_invocations_per_cycle("full+grid3"), 18)

    def test_matching_counts_duplicate_detection_as_false_positive(self) -> None:
        ground_truth = [(0.4, 0.4, 0.6, 0.6)]
        predictions = [
            Detection(0.9, (0.4, 0.4, 0.6, 0.6)),
            Detection(0.8, (0.41, 0.41, 0.59, 0.59)),
        ]
        result = match_detections(ground_truth, predictions, 0.2, 0.5)
        self.assertEqual(result.true_positives, 1)
        self.assertEqual(len(result.false_positives), 1)
        self.assertEqual(result.false_negatives, 0)

    def test_hard_negative_prediction_is_counted_as_false_positive(self) -> None:
        image = Path("negative.jpg")
        point = aggregate_at_threshold(
            {image: []},
            {image: [Detection(0.958, (0.1, 0.1, 0.2, 0.2))]},
            threshold=0.8,
            iou_threshold=0.5,
        )
        self.assertEqual(point["false_positives"], 1)
        self.assertEqual(point["negative_images_with_false_positive"], 1)

    def test_size_bins_report_small_object_recall_cliff(self) -> None:
        image = Path("positive.jpg")
        boxes = [
            (0.10, 0.10, 0.11, 0.11),
            (0.20, 0.20, 0.225, 0.225),
            (0.30, 0.30, 0.35, 0.35),
            (0.40, 0.40, 0.50, 0.50),
        ]
        predictions = {
            image: [
                Detection(0.9, boxes[1]),
                Detection(0.9, boxes[2]),
                Detection(0.9, boxes[3]),
            ]
        }

        result = aggregate_by_size(
            {image: boxes}, predictions, threshold=0.25, iou_threshold=0.5, imgsz=640
        )

        self.assertEqual(result["bins"]["<12"]["ground_truth"], 1)
        self.assertEqual(result["bins"]["<12"]["recall"], 0.0)
        self.assertEqual(result["bins"]["12-20"]["recall"], 1.0)
        self.assertEqual(result["bins"]["20-40"]["recall"], 1.0)
        self.assertEqual(result["bins"][">40"]["recall"], 1.0)

    def test_visibility_buckets_match_build4_contract(self) -> None:
        self.assertEqual(visibility_bin(80), "75-100")
        self.assertEqual(visibility_bin(0.60), "50-75")
        self.assertEqual(visibility_bin(40), "30-50")
        self.assertEqual(visibility_bin(0.20), "<30")
        with self.assertRaises(ValueError):
            visibility_bin(float("nan"))

    def test_three_by_three_tiles_cover_frame_and_overlap(self) -> None:
        tiles = grid3_regions()
        self.assertEqual(len(tiles), 9)
        self.assertEqual(tiles[0], (0.0, 0.0, 0.42, 0.42))
        for actual, expected in zip(tiles[-1], (0.58, 0.58, 1.0, 1.0), strict=True):
            self.assertAlmostEqual(actual, expected, places=12)
        self.assertLess(tiles[1][0], tiles[0][2])
        self.assertEqual(len(scan_regions("full+grid3")), 10)

    def test_tile_local_box_maps_to_full_frame(self) -> None:
        mapped = map_local_box_to_frame(
            (0.25, 0.25, 0.75, 0.75),
            (0.58, 0.58, 1.0, 1.0),
        )
        for actual, expected in zip(mapped, (0.685, 0.685, 0.895, 0.895), strict=True):
            self.assertAlmostEqual(actual, expected, places=12)

    def test_cross_tile_nms_removes_duplicate_without_losing_best_confidence(self) -> None:
        detections = [
            Detection(0.90, (0.4, 0.4, 0.5, 0.5)),
            Detection(0.80, (0.405, 0.405, 0.505, 0.505)),
            Detection(0.70, (0.7, 0.7, 0.8, 0.8)),
        ]
        kept = suppress_overlaps(detections, 0.70)
        self.assertEqual([item.confidence for item in kept], [0.90, 0.70])

    def test_threshold_recommendation_prefers_precision_recall_gate(self) -> None:
        points = [
            {"threshold": 0.1, "precision": 0.70, "recall": 0.95, "false_positives": 30},
            {"threshold": 0.2, "precision": 0.91, "recall": 0.86, "false_positives": 5},
            {"threshold": 0.3, "precision": 0.95, "recall": 0.79, "false_positives": 2},
        ]
        recommendation = recommend_thresholds(points)
        self.assertEqual(recommendation["confirmed_average_confidence"], 0.2)
        self.assertEqual(recommendation["method"], "precision>=0.90_and_recall>=0.85_gate")

    def test_compares_seed_and_new_fp_at_the_same_threshold(self) -> None:
        common = {
            "schema_version": 2,
            "dataset_yaml_sha256": "a" * 64,
            "dataset_manifest_sha256": "b" * 64,
            "split": "test",
            "imgsz": 640,
            "offline_metrics": {"precision": 0.9, "recall": 0.8, "map50": 0.8, "map50_95": 0.5},
            "error_summary": {"high_confidence_false_positives": []},
        }
        seed = {
            **common,
            "weights": "seed.pt",
            "weights_sha256": "c" * 64,
            "operating_point": {"threshold": 0.4, "precision": 0.5, "recall": 0.8, "false_positives": 20, "false_negatives": 5},
            "high_confidence_0_8": {"false_positives": 10},
        }
        new = {
            **common,
            "weights": "new.pt",
            "weights_sha256": "d" * 64,
            "operating_point": {"threshold": 0.4, "precision": 0.9, "recall": 0.8, "false_positives": 2, "false_negatives": 5},
            "high_confidence_0_8": {"false_positives": 1},
        }

        comparison = compare_reports(seed, new)

        self.assertEqual(comparison["false_positive_reduction"]["count"], 18)
        self.assertEqual(comparison["false_positive_reduction"]["high_confidence_count"], 9)
        self.assertFalse(comparison["mvp_gate"]["numeric_gate_pass"])

    def test_refuses_offline_comparison_at_different_thresholds(self) -> None:
        seed = {
            "schema_version": 2,
            "dataset_yaml_sha256": "a",
            "dataset_manifest_sha256": "b",
            "split": "test",
            "imgsz": 640,
            "operating_point": {"threshold": 0.2},
        }
        new = {**seed, "operating_point": {"threshold": 0.3}}
        with self.assertRaisesRegex(OfflineComparisonError, "thresholds differ"):
            compare_reports(seed, new)

    def test_checkpoint_selection_prefers_gate_then_high_confidence_fp(self) -> None:
        def candidate(precision: float, recall: float, fp: int, high_fp: int) -> dict[str, object]:
            return {
                "threshold_recommendation": {
                    "confirmed_point": {
                        "precision": precision,
                        "recall": recall,
                        "false_positives": fp,
                    }
                },
                "high_confidence_0_8": {"false_positives": high_fp},
                "build4_gate": {"small_recall": 0.81, "partial_recall": 0.82},
            }

        gate_more_fp = candidate(0.91, 0.86, 4, 2)
        gate_less_fp = candidate(0.90, 0.85, 3, 1)
        no_gate = candidate(0.89, 0.85, 1, 0)
        self.assertIs(min((gate_more_fp, gate_less_fp, no_gate), key=selection_rank), gate_less_fp)


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

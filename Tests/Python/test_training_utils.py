from __future__ import annotations

import argparse
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.fetch_seed_model import sha256  # noqa: E402
from scripts.fetch_release_model import install_checkpoint  # noqa: E402
from scripts.inspect_yolo_checkpoint import CheckpointContractError, validate_contract  # noqa: E402
from training.extract_frames import extract_video, positive_interval, prepare_session, session_name  # noqa: E402


class ExtractFramesTests(unittest.TestCase):
    def test_interval_must_be_positive_and_finite(self) -> None:
        self.assertEqual(positive_interval("0.75"), 0.75)
        for invalid in ("0", "-1", "nan", "inf"):
            with self.subTest(invalid=invalid), self.assertRaises(argparse.ArgumentTypeError):
                positive_interval(invalid)

    def test_equal_video_stems_from_different_paths_have_distinct_sessions(self) -> None:
        first = session_name(Path("course_a") / "rough.MOV")
        second = session_name(Path("course_b") / "rough.MOV")
        self.assertNotEqual(first, second)
        self.assertTrue(first.startswith("rough_"))

    def test_prepare_session_refuses_silent_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            session = Path(directory) / "session"
            session.mkdir()
            generated = session / "frame_0000000.jpg"
            generated.write_bytes(b"generated")
            retained = session / "reviewer_notes.txt"
            retained.write_text("keep", encoding="utf-8")

            with self.assertRaises(FileExistsError):
                prepare_session(session, overwrite=False)

            prepare_session(session, overwrite=True)
            self.assertFalse(generated.exists())
            self.assertEqual(retained.read_text(encoding="utf-8"), "keep")

    def test_open_video_with_no_decodable_frames_fails(self) -> None:
        capture = MagicMock()
        capture.isOpened.return_value = True
        capture.get.return_value = 30.0
        capture.read.return_value = (False, None)

        with tempfile.TemporaryDirectory() as directory, patch(
            "training.extract_frames.cv2.VideoCapture", return_value=capture
        ):
            source = Path(directory) / "empty.mov"
            with self.assertRaisesRegex(RuntimeError, "no decodable frames"):
                extract_video(source, Path(directory) / "output", interval=0.75)

        capture.release.assert_called_once()


class SeedHashTests(unittest.TestCase):
    def test_sha256_streaming_helper(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            sample = Path(directory) / "sample.bin"
            sample.write_bytes(b"golf-ball")
            self.assertEqual(
                sha256(sample),
                "86d19edee4bc11e9e97b41aaf054c5448ad9cea410621f10524424902d060132",
            )


class ReleaseCheckpointTests(unittest.TestCase):
    def test_requires_https_and_valid_sha256(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "model.pt"
            with self.assertRaisesRegex(ValueError, "HTTPS"):
                install_checkpoint("http://example.com/model.pt", "a" * 64, output)
            with self.assertRaisesRegex(ValueError, "64 lowercase"):
                install_checkpoint("https://example.com/model.pt", "invalid", output)

    def test_reuses_an_existing_sha256_verified_checkpoint_without_network(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "model.pt"
            output.write_bytes(b"verified model")
            expected = "6c736b3dfa943bf4e7c61df78d1dfcad9a3d8b56369f0559670497b19127e74d"
            self.assertEqual(
                install_checkpoint("https://invalid.example/model.pt", expected, output),
                expected,
            )


class CheckpointContractTests(unittest.TestCase):
    def test_accepts_pinned_one_class_640_raw_shape(self) -> None:
        validate_contract({0: "golf_ball"}, [1, 5, 8400], 640)

    def test_rejects_class_or_shape_drift(self) -> None:
        with self.assertRaises(CheckpointContractError):
            validate_contract({0: "ball"}, [1, 5, 8400], 640)
        with self.assertRaises(CheckpointContractError):
            validate_contract({0: "golf_ball"}, [1, 6, 8400], 640)


if __name__ == "__main__":
    unittest.main()

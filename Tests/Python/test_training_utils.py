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


if __name__ == "__main__":
    unittest.main()

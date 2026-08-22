#!/usr/bin/env python3
"""Extract spaced frames from iPhone videos while keeping each recording as a distinct session."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

import cv2


def positive_interval(value: str) -> float:
    interval = float(value)
    if not math.isfinite(interval) or interval <= 0:
        raise argparse.ArgumentTypeError("interval must be a finite number greater than zero")
    return interval


def session_name(video_path: Path) -> str:
    """Return a readable ID that cannot collide for equal stems in different folders."""
    safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", video_path.stem).strip("._-") or "video"
    source_key = str(video_path.resolve()).casefold().encode("utf-8")
    suffix = hashlib.sha256(source_key).hexdigest()[:8]
    return f"{safe_stem}_{suffix}"


def prepare_session(session: Path, overwrite: bool) -> None:
    existing_frames = list(session.glob("frame_*.jpg")) if session.exists() else []
    metadata = session / "session.json"
    if (existing_frames or metadata.exists()) and not overwrite:
        raise FileExistsError(
            f"Session already contains extracted data: {session}. "
            "Pass --overwrite to replace generated frames for this session."
        )
    session.mkdir(parents=True, exist_ok=True)
    if overwrite:
        for frame in existing_frames:
            frame.unlink()
        metadata.unlink(missing_ok=True)


def extract_video(video_path: Path, out_root: Path, interval: float, overwrite: bool = False) -> tuple[int, Path]:
    video_path = video_path.resolve()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if not math.isfinite(fps) or fps <= 0:
            fps = 30.0
        step = max(1, round(fps * interval))
        session = out_root / session_name(video_path)
        prepare_session(session, overwrite=overwrite)

        frame_index = 0
        saved = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_index % step == 0:
                dest = session / f"frame_{frame_index:07d}.jpg"
                written = cv2.imwrite(str(dest), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 94])
                if not written:
                    raise RuntimeError(f"Failed to write extracted frame: {dest}")
                saved += 1
            frame_index += 1
    finally:
        cap.release()

    if frame_index == 0:
        raise RuntimeError(f"Video contained no decodable frames: {video_path}")

    metadata = {
        "session_id": session.name,
        "source_video": str(video_path),
        "interval_seconds": interval,
        "source_fps": fps,
        "frame_step": step,
        "decoded_frames": frame_index,
        "saved_frames": saved,
    }
    (session / "session.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return saved, session


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("videos", nargs="+", help="Video files from iPhone")
    p.add_argument("--out", default="training/datasets/raw_sessions")
    p.add_argument("--interval", type=positive_interval, default=0.75, help="Seconds between frames")
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace only previously generated frame_*.jpg files for matching sessions.",
    )
    args = p.parse_args()

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    failures: list[str] = []
    for video_path in map(Path, args.videos):
        try:
            saved, session = extract_video(video_path, out_root, args.interval, overwrite=args.overwrite)
            print(f"{video_path.name}: saved {saved} frames -> {session}")
        except (OSError, RuntimeError) as error:
            failures.append(str(error))
            print(f"ERROR {error}")

    if failures:
        raise SystemExit(f"Frame extraction failed for {len(failures)} input(s)")


if __name__ == "__main__":
    main()

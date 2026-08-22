#!/usr/bin/env python3
"""Assemble YOLO split directories from a session-level manifest without random frame splitting."""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import yaml

try:
    from .validate_dataset import IMAGE_SUFFIXES, load_session_manifest, validate_dataset_config
except ImportError:  # Direct `python training/prepare_dataset.py` execution.
    from validate_dataset import IMAGE_SUFFIXES, load_session_manifest, validate_dataset_config


def prepare_dataset(manifest_path: Path, output: Path) -> int:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output must be absent or empty; refusing to merge datasets: {output}")
    output.mkdir(parents=True, exist_ok=True)
    sessions = load_session_manifest(manifest_path)
    copied = 0

    for session in sessions.values():
        source = Path(session.source_dir)
        if not source.is_absolute():
            source = (manifest_path.parent / source).resolve()
        if not source.is_dir():
            raise FileNotFoundError(f"Session source_dir not found for {session.session_id}: {source}")
        images = sorted(path for path in source.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)
        if not images:
            raise ValueError(f"Session contains no images: {session.session_id}")

        for image in images:
            relative = image.relative_to(source)
            image_target = output / "images" / session.split / session.session_id / relative
            label_target = output / "labels" / session.split / session.session_id / relative.with_suffix(".txt")
            image_target.parent.mkdir(parents=True, exist_ok=True)
            label_target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(image, image_target)

            adjacent_label = image.with_suffix(".txt")
            nested_label = source / "labels" / relative.with_suffix(".txt")
            label_source = adjacent_label if adjacent_label.is_file() else nested_label
            if label_source.is_file():
                shutil.copy2(label_source, label_target)
            else:
                label_target.write_text("", encoding="utf-8")
            copied += 1

    dataset_yaml = output / "dataset.yaml"
    dataset_yaml.write_text(
        yaml.safe_dump(
            {
                "path": str(output.resolve()),
                "train": "images/train",
                "val": "images/val",
                "test": "images/test",
                "names": {0: "golf_ball"},
            },
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    copied_manifest = output / "dataset_manifest.csv"
    shutil.copy2(manifest_path, copied_manifest)
    validate_dataset_config(dataset_yaml, copied_manifest)
    return copied


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="training/dataset_manifest.csv")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    copied = prepare_dataset(Path(args.manifest).resolve(), Path(args.output).resolve())
    print(f"Prepared and validated {copied} images without cross-session splitting")


if __name__ == "__main__":
    main()

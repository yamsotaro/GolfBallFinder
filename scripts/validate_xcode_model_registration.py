#!/usr/bin/env python3
"""Validate that XcodeGen registered a Core ML package for Xcode compilation."""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path


class ModelRegistrationError(ValueError):
    """Raised when the generated Xcode project violates the model contract."""


@dataclass(frozen=True)
class PBXObject:
    identifier: str
    comment: str
    body: str


def _section(text: str, name: str) -> str:
    match = re.search(
        rf"/\* Begin {re.escape(name)} section \*/(.*?)/\* End {re.escape(name)} section \*/",
        text,
        re.DOTALL,
    )
    if not match:
        raise ModelRegistrationError(f"{name} section is missing")
    return match.group(1)


def _objects(text: str, section_name: str) -> list[PBXObject]:
    section = _section(text, section_name)
    starts = list(
        re.finditer(
            r"(?m)^\s*([A-Za-z0-9]+) /\* (.*?) \*/ = \{",
            section,
        )
    )
    objects: list[PBXObject] = []
    for start in starts:
        depth = 0
        end_index: int | None = None
        for index in range(start.end() - 1, len(section)):
            character = section[index]
            if character == "{":
                depth += 1
            elif character == "}":
                depth -= 1
                if depth == 0:
                    end_index = index + 1
                    break
        if end_index is None:
            raise ModelRegistrationError(
                f"Unterminated {section_name} object {start.group(1)}"
            )
        objects.append(
            PBXObject(
                identifier=start.group(1),
                comment=start.group(2),
                body=section[start.start() : end_index],
            )
        )
    return objects


def _setting_matches(body: str, key: str, value: str) -> bool:
    return bool(
        re.search(
            rf"\b{re.escape(key)}\s*=\s*\"?{re.escape(value)}\"?\s*;",
            body,
        )
    )


def runtime_model_name(config_path: Path) -> str:
    text = config_path.read_text(encoding="utf-8")
    match = re.search(
        r"\bstatic\s+let\s+modelName(?:\s*:\s*[A-Za-z0-9_.<>]+)?\s*=\s*\"([^\"]+)\"",
        text,
    )
    if not match:
        raise ModelRegistrationError(
            f"Could not read AppConfig.modelName from {config_path}"
        )
    return match.group(1)


def validate_registration(
    pbxproj_path: Path,
    target_name: str,
    model_filename: str,
    runtime_config_path: Path,
) -> dict[str, str]:
    text = pbxproj_path.read_text(encoding="utf-8")

    runtime_name = runtime_model_name(runtime_config_path)
    compiled_name = Path(model_filename).stem
    if runtime_name != compiled_name:
        raise ModelRegistrationError(
            f"Runtime model name {runtime_name!r} does not match compiled model "
            f"name {compiled_name!r}"
        )

    file_references = [
        item
        for item in _objects(text, "PBXFileReference")
        if item.comment == model_filename
        or _setting_matches(item.body, "path", model_filename)
    ]
    if not file_references:
        raise ModelRegistrationError(
            f"{model_filename} has no PBXFileReference in {pbxproj_path}"
        )
    if len(file_references) != 1:
        raise ModelRegistrationError(
            f"Expected one PBXFileReference for {model_filename}, found "
            f"{len(file_references)}"
        )
    file_reference = file_references[0]

    build_files = [
        item
        for item in _objects(text, "PBXBuildFile")
        if re.search(
            rf"\bfileRef\s*=\s*{re.escape(file_reference.identifier)}\b",
            item.body,
        )
    ]
    if not build_files:
        raise ModelRegistrationError(
            f"{model_filename} has a PBXFileReference but no PBXBuildFile"
        )
    build_file_ids = {item.identifier for item in build_files}

    native_targets = [
        item
        for item in _objects(text, "PBXNativeTarget")
        if item.comment == target_name
        or _setting_matches(item.body, "name", target_name)
    ]
    if len(native_targets) != 1:
        raise ModelRegistrationError(
            f"Expected one PBXNativeTarget named {target_name}, found {len(native_targets)}"
        )
    target = native_targets[0]
    build_phases_match = re.search(
        r"\bbuildPhases\s*=\s*\((.*?)\);", target.body, re.DOTALL
    )
    if not build_phases_match:
        raise ModelRegistrationError(f"Target {target_name} has no buildPhases list")
    target_phase_ids = set(
        re.findall(r"(?m)^\s*([A-Za-z0-9]+)\s+/\*", build_phases_match.group(1))
    )

    source_phases = {
        item.identifier: item for item in _objects(text, "PBXSourcesBuildPhase")
    }
    target_source_phases = [
        source_phases[identifier]
        for identifier in target_phase_ids
        if identifier in source_phases
    ]
    matching_source_phase = next(
        (
            phase
            for phase in target_source_phases
            if any(
                re.search(rf"\b{re.escape(build_id)}\b", phase.body)
                for build_id in build_file_ids
            )
        ),
        None,
    )
    if matching_source_phase is None:
        raise ModelRegistrationError(
            f"{model_filename} is not in target {target_name}'s PBXSourcesBuildPhase"
        )

    resource_phases = {
        item.identifier: item for item in _objects(text, "PBXResourcesBuildPhase")
    }
    for identifier in target_phase_ids:
        phase = resource_phases.get(identifier)
        if phase and any(
            re.search(rf"\b{re.escape(build_id)}\b", phase.body)
            for build_id in build_file_ids
        ):
            raise ModelRegistrationError(
                f"{model_filename} is incorrectly registered in target "
                f"{target_name}'s PBXResourcesBuildPhase"
            )

    file_type_match = re.search(
        r"\b(?:lastKnownFileType|explicitFileType)\s*=\s*([^;]+);",
        file_reference.body,
    )
    return {
        "target_id": target.identifier,
        "file_reference_id": file_reference.identifier,
        "file_type": file_type_match.group(1) if file_type_match else "extension-inferred",
        "build_file_ids": ",".join(sorted(build_file_ids)),
        "sources_phase_id": matching_source_phase.identifier,
        "runtime_model_name": runtime_name,
        "compiled_bundle_name": f"{compiled_name}.mlmodelc",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pbxproj", type=Path, required=True)
    parser.add_argument("--target", default="GolfBallFinder")
    parser.add_argument("--model", default="GolfBall.mlpackage")
    parser.add_argument(
        "--runtime-config",
        type=Path,
        default=Path("GolfBallFinder/AppConfig.swift"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        result = validate_registration(
            args.pbxproj,
            args.target,
            args.model,
            args.runtime_config,
        )
    except (OSError, ModelRegistrationError) as error:
        print(f"Core ML project registration validation failed: {error}", file=sys.stderr)
        raise SystemExit(2) from error

    print("Validated Xcode Core ML model registration:")
    for key, value in result.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()

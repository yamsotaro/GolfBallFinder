from __future__ import annotations

import csv
import io
import plistlib
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import yaml
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts import configure_bundle_id, validate_xcode_model_registration  # noqa: E402


PROJECT_TEMPLATE = """\
targets:
  GolfBallFinder:
    settings:
      base:
        PRODUCT_BUNDLE_IDENTIFIER: dev.local.GolfBallFinder
        CURRENT_PROJECT_VERSION: 1
"""


class ConfigureBundleIDTests(unittest.TestCase):
    def run_configure(self, project: Path, *arguments: str) -> str:
        argv = [
            "configure_bundle_id.py",
            "--project",
            str(project),
            *arguments,
        ]
        output = io.StringIO()
        with patch.object(sys, "argv", argv), redirect_stdout(output):
            configure_bundle_id.main()
        return output.getvalue()

    def test_configures_bundle_id_and_positive_build_number(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory) / "project.yml"
            project.write_text(PROJECT_TEMPLATE, encoding="utf-8")

            output = self.run_configure(
                project,
                "--bundle-id",
                "com.example.golfballfinder",
                "--build-number",
                "42",
            )

            updated = project.read_text(encoding="utf-8")
            self.assertIn("PRODUCT_BUNDLE_IDENTIFIER: com.example.golfballfinder", updated)
            self.assertIn("CURRENT_PROJECT_VERSION: 42", updated)
            self.assertIn("Configured build number: 42", output)

    def test_rejects_invalid_bundle_identifier_without_writing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory) / "project.yml"
            project.write_text(PROJECT_TEMPLATE, encoding="utf-8")

            with self.assertRaises(SystemExit):
                self.run_configure(project, "--bundle-id", "not a bundle id")

            self.assertEqual(project.read_text(encoding="utf-8"), PROJECT_TEMPLATE)

    def test_rejects_nonpositive_build_number_without_writing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory) / "project.yml"
            project.write_text(PROJECT_TEMPLATE, encoding="utf-8")

            with self.assertRaises(SystemExit):
                self.run_configure(
                    project,
                    "--bundle-id",
                    "com.example.golfballfinder",
                    "--build-number",
                    "0",
                )

            self.assertEqual(project.read_text(encoding="utf-8"), PROJECT_TEMPLATE)


class RepositoryConfigurationTests(unittest.TestCase):
    def test_yaml_files_parse_and_required_workflows_exist(self) -> None:
        project = yaml.safe_load((ROOT / "project.yml").read_text(encoding="utf-8"))
        codemagic = yaml.safe_load((ROOT / "codemagic.yaml").read_text(encoding="utf-8"))
        dataset = yaml.safe_load(
            (ROOT / "training" / "dataset.yaml.example").read_text(encoding="utf-8")
        )

        self.assertIn("GolfBallFinder", project["targets"])
        self.assertIn("GolfBallFinderTests", project["targets"])
        self.assertFalse(project["targets"]["GolfBallFinder"]["info"]["properties"]["ITSAppUsesNonExemptEncryption"])
        self.assertEqual(dataset["names"], {0: "golf_ball"})
        self.assertIn("ios-compile-check", codemagic["workflows"])
        self.assertIn("ios-model-compile-check", codemagic["workflows"])
        self.assertIn("ios-testflight", codemagic["workflows"])

    def test_coreml_package_is_a_target_source_not_a_copied_directory(self) -> None:
        project = yaml.safe_load((ROOT / "project.yml").read_text(encoding="utf-8"))
        self.assertGreaterEqual(
            tuple(
                int(part)
                for part in project["options"]["minimumXcodeGenVersion"].split(".")
            ),
            (2, 38, 0),
        )
        target = project["targets"]["GolfBallFinder"]
        self.assertIn(
            {
                "path": "GolfBallFinder/Resources/GolfBall.mlpackage",
                "buildPhase": "sources",
            },
            target["sources"],
        )
        self.assertNotIn(
            {"path": "GolfBallFinder/Resources"},
            target["resources"],
        )
        self.assertIn(
            {"path": "GolfBallFinder/Resources/ModelManifest.json"},
            target["resources"],
        )

        compile_overlay = yaml.safe_load(
            (ROOT / "project.compile-check.yml").read_text(encoding="utf-8")
        )
        compile_target = compile_overlay["targets"]["GolfBallFinder"]
        self.assertNotIn(
            "GolfBallFinder/Resources/GolfBall.mlpackage",
            [entry["path"] for entry in compile_target["sources:REPLACE"]],
        )
        self.assertEqual(
            compile_target["resources:REPLACE"],
            [{"path": "GolfBallFinder/PrivacyInfo.xcprivacy"}],
        )

        codemagic = yaml.safe_load(
            (ROOT / "codemagic.yaml").read_text(encoding="utf-8")
        )
        compile_scripts = "\n".join(
            step["script"]
            for step in codemagic["workflows"]["ios-compile-check"]["scripts"]
        )
        self.assertIn("--spec project.compile-check.yml", compile_scripts)

    def test_model_compile_workflow_verifies_seed_and_bundled_model(self) -> None:
        workflow = yaml.safe_load((ROOT / "codemagic.yaml").read_text(encoding="utf-8"))["workflows"][
            "ios-model-compile-check"
        ]
        scripts = "\n".join(step["script"] for step in workflow["scripts"])
        self.assertIn("fetch_seed_model.py", scripts)
        self.assertIn("EXPECTED_SHA256", scripts)
        self.assertIn("export_coreml.py", scripts)
        self.assertIn("GolfBall.mlmodelc", scripts)
        self.assertIn("CODE_SIGNING_ALLOWED=NO", scripts)
        self.assertIn("validate_xcode_model_registration.py", scripts)
        self.assertIn("PBXFileReference", scripts)
        self.assertIn("coremlcompiler", scripts)
        self.assertIn("-name '*.mlmodelc'", scripts)
        self.assertIn("build/model-test-derived", scripts)
        self.assertIn("build/model-compile-derived", scripts)

    def test_release_workflow_sets_unique_build_number(self) -> None:
        text = (ROOT / "codemagic.yaml").read_text(encoding="utf-8")
        self.assertIn('--build-number "$BUILD_NUMBER"', text)

    def test_app_icon_is_configured(self) -> None:
        project = yaml.safe_load((ROOT / "project.yml").read_text(encoding="utf-8"))
        settings = project["targets"]["GolfBallFinder"]["settings"]["base"]
        self.assertEqual(settings["ASSETCATALOG_COMPILER_APPICON_NAME"], "AppIcon")
        icon_set = ROOT / "GolfBallFinder" / "Assets.xcassets" / "AppIcon.appiconset"
        self.assertTrue((icon_set / "Contents.json").is_file())
        self.assertTrue((icon_set / "AppIcon-1024.png").is_file())
        with Image.open(icon_set / "AppIcon-1024.png") as icon:
            self.assertEqual(icon.size, (1024, 1024))
            self.assertEqual(icon.mode, "RGB", "App Store icon must not contain an alpha channel")

    def test_field_log_rows_match_header_width(self) -> None:
        with (ROOT / "docs" / "FIELD_TEST_LOG_TEMPLATE.csv").open(
            encoding="utf-8", newline=""
        ) as file:
            rows = list(csv.reader(file))
        self.assertGreaterEqual(len(rows), 2)
        self.assertTrue(all(len(row) == len(rows[0]) for row in rows))

    def test_privacy_manifest_declares_no_collection_or_tracking(self) -> None:
        manifest_path = ROOT / "GolfBallFinder" / "PrivacyInfo.xcprivacy"
        manifest = plistlib.loads(manifest_path.read_bytes())
        self.assertFalse(manifest["NSPrivacyTracking"])
        self.assertEqual(manifest["NSPrivacyCollectedDataTypes"], [])
        self.assertEqual(manifest["NSPrivacyTrackingDomains"], [])
        self.assertIn(
            {"path": "GolfBallFinder/PrivacyInfo.xcprivacy"},
            yaml.safe_load((ROOT / "project.yml").read_text(encoding="utf-8"))["targets"][
                "GolfBallFinder"
            ]["resources"],
        )

    def test_field_diagnostics_are_local_and_files_recoverable(self) -> None:
        project = yaml.safe_load((ROOT / "project.yml").read_text(encoding="utf-8"))
        properties = project["targets"]["GolfBallFinder"]["info"]["properties"]
        self.assertTrue(properties["UIFileSharingEnabled"])
        self.assertTrue(properties["LSSupportsOpeningDocumentsInPlace"])

        source = (ROOT / "GolfBallFinder" / "Diagnostics" / "FieldDiagnostics.swift").read_text(
            encoding="utf-8"
        )
        for field in (
            "inferenceLatencyMs",
            "effectiveInferenceFPS",
            "thermalState",
            "detectionConfidence",
            "scanMode",
            "candidateToConfirmedMs",
            "sceneStartToConfirmedMs",
            "detectedBBox",
            "timestamp",
            "false_positive",
            "missed_golf_ball",
        ):
            self.assertIn(field, source)

    def test_inference_path_discards_late_frames_and_has_thermal_policy(self) -> None:
        camera = (ROOT / "GolfBallFinder" / "Camera" / "CameraController.swift").read_text(
            encoding="utf-8"
        )
        gate = (ROOT / "GolfBallFinder" / "Detection" / "InferenceFrameGate.swift").read_text(
            encoding="utf-8"
        )
        self.assertIn("alwaysDiscardsLateVideoFrames = true", camera)
        self.assertIn("guard !inferenceSuspended", camera)
        self.assertNotIn("pendingFrame", gate)
        self.assertIn("case .critical", gate)


PBXPROJ_WITH_COMPILED_MODEL_SOURCE = """\
/* Begin PBXBuildFile section */
  BUILD1 /* GolfBall.mlpackage in Sources */ = {isa = PBXBuildFile; fileRef = FILE1 /* GolfBall.mlpackage */; };
/* End PBXBuildFile section */
/* Begin PBXFileReference section */
  FILE1 /* GolfBall.mlpackage */ = {isa = PBXFileReference; lastKnownFileType = folder.mlpackage; path = GolfBall.mlpackage; sourceTree = \"<group>\"; };
/* End PBXFileReference section */
/* Begin PBXNativeTarget section */
  TARGET1 /* GolfBallFinder */ = {
    isa = PBXNativeTarget;
    buildPhases = (
      SOURCES1 /* Sources */,
      RESOURCES1 /* Resources */,
    );
    name = GolfBallFinder;
  };
/* End PBXNativeTarget section */
/* Begin PBXResourcesBuildPhase section */
  RESOURCES1 /* Resources */ = {
    isa = PBXResourcesBuildPhase;
    files = (
    );
  };
/* End PBXResourcesBuildPhase section */
/* Begin PBXSourcesBuildPhase section */
  SOURCES1 /* Sources */ = {
    isa = PBXSourcesBuildPhase;
    files = (
      BUILD1 /* GolfBall.mlpackage in Sources */,
    );
  };
/* End PBXSourcesBuildPhase section */
"""


class XcodeModelRegistrationValidatorTests(unittest.TestCase):
    def test_accepts_model_in_application_sources_phase(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            pbxproj = Path(directory) / "project.pbxproj"
            pbxproj.write_text(PBXPROJ_WITH_COMPILED_MODEL_SOURCE, encoding="utf-8")

            result = validate_xcode_model_registration.validate_registration(
                pbxproj,
                "GolfBallFinder",
                "GolfBall.mlpackage",
                ROOT / "GolfBallFinder" / "AppConfig.swift",
            )

        self.assertEqual(result["sources_phase_id"], "SOURCES1")
        self.assertEqual(result["runtime_model_name"], "GolfBall")
        self.assertEqual(result["compiled_bundle_name"], "GolfBall.mlmodelc")

    def test_rejects_model_in_copy_resources_phase(self) -> None:
        broken = PBXPROJ_WITH_COMPILED_MODEL_SOURCE.replace(
            "      BUILD1 /* GolfBall.mlpackage in Sources */,\n",
            "",
        ).replace(
            "    files = (\n    );\n  };\n/* End PBXResourcesBuildPhase section */",
            "    files = (\n      BUILD1 /* GolfBall.mlpackage in Resources */,\n    );\n  };\n/* End PBXResourcesBuildPhase section */",
        )
        with tempfile.TemporaryDirectory() as directory:
            pbxproj = Path(directory) / "project.pbxproj"
            pbxproj.write_text(broken, encoding="utf-8")

            with self.assertRaises(
                validate_xcode_model_registration.ModelRegistrationError
            ):
                validate_xcode_model_registration.validate_registration(
                    pbxproj,
                    "GolfBallFinder",
                    "GolfBall.mlpackage",
                    ROOT / "GolfBallFinder" / "AppConfig.swift",
                )


if __name__ == "__main__":
    unittest.main()

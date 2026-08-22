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

from scripts import configure_bundle_id  # noqa: E402


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
        self.assertIn("ios-testflight", codemagic["workflows"])

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


if __name__ == "__main__":
    unittest.main()

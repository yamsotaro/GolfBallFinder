from __future__ import annotations

import csv
import hashlib
import json
import re
import subprocess
import sys
import unittest
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts import audit_public_release  # noqa: E402


class PublicReleaseTests(unittest.TestCase):
    def test_history_rewrite_mapping_is_hash_only(self) -> None:
        mapping = (
            ROOT / "docs" / "GIT_HISTORY_REWRITE_MAP_2026-08-25.md"
        ).read_text(encoding="utf-8")
        pairs = re.findall(
            r"^\| `([0-9a-f]{40})` \| `([0-9a-f]{40})` \|$",
            mapping,
            flags=re.MULTILINE,
        )
        self.assertEqual(len(pairs), 8)
        self.assertEqual(len({old for old, _ in pairs}), 8)
        self.assertEqual(len({new for _, new in pairs}), 8)
        self.assertNotIn("@", mapping)

    def test_license_is_exact_official_agpl_v3_text(self) -> None:
        license_text = (ROOT / "LICENSE").read_text(encoding="utf-8")
        normalized = license_text.replace("\r\n", "\n")
        self.assertTrue(normalized.lstrip().startswith("GNU AFFERO GENERAL PUBLIC LICENSE\n"))
        self.assertIn("Version 3, 19 November 2007", normalized)
        self.assertIn("13. Remote Network Interaction; Use with the GNU General Public License.", normalized)
        self.assertIn("END OF TERMS AND CONDITIONS", normalized)
        self.assertEqual(
            hashlib.sha256(normalized.encode("utf-8")).hexdigest(),
            "0d96a4ff68ad6d4b6f1f30f713b18d5184912ba8dd389f86aa7710db079abcb0",
        )

    def test_readme_and_beta_notice_publish_source_and_license(self) -> None:
        repository_url = "https://github.com/yamsotaro/GolfBallFinder"
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        beta = (ROOT / "docs" / "TESTFLIGHT_BETA_DESCRIPTION.md").read_text(
            encoding="utf-8"
        )
        notices = (ROOT / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")
        for text in (readme, beta):
            self.assertIn(repository_url, text)
            self.assertIn("AGPL", text)
        self.assertIn("Ultralytics", notices)
        self.assertIn("Open Images Dataset V7", notices)

    def test_model_publication_manifest_is_pinned_but_not_claimed_published(self) -> None:
        release = json.loads(
            (ROOT / "training" / "public_mvp_release_v3.json").read_text(
                encoding="utf-8"
            )
        )
        base = release["training"]["base_checkpoint"]
        self.assertEqual(base["license"], "Apache-2.0")
        self.assertTrue(base["source_url"].startswith("https://"))
        self.assertEqual(base["sha256"], release["training"]["base_checkpoint_sha256"])
        requirements = (ROOT / "training" / "requirements-windows.txt").read_text(
            encoding="utf-8"
        )
        for package, version in release["training"]["toolchain"].items():
            requirement_name = package.replace("_", "-")
            self.assertIn(f"{requirement_name}=={version}".lower(), requirements.lower())
        checkpoint = release["checkpoint"]
        self.assertEqual(checkpoint["license"], "AGPL-3.0-only")
        self.assertEqual(checkpoint["release_asset_name"], "public_mvp_best.pt")
        self.assertEqual(
            checkpoint["sha256"],
            "5b18acff26464a447d00703b08875603e7e8cfa6e53827dc7092d03f2b643199",
        )
        publication = checkpoint["publication"]
        self.assertFalse(publication["published"])
        self.assertTrue(publication["planned_public_url"].startswith("https://"))
        self.assertEqual(publication["codemagic_url_variable"], "MODEL_CHECKPOINT_URL")

    def test_all_882_image_attributions_are_public_metadata(self) -> None:
        attribution_path = (
            ROOT / "training" / "datasets" / "public_mvp_v3" / "attribution.csv"
        )
        source_manifest_path = attribution_path.with_name("source_manifest.json")
        lock = json.loads(
            (ROOT / "training" / "public_mvp_v3.lock.json").read_text(encoding="utf-8")
        )
        source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
        with attribution_path.open(encoding="utf-8", newline="") as file:
            rows = list(csv.DictReader(file))

        self.assertEqual(len(rows), 882)
        self.assertEqual(Counter(row["content_type"] for row in rows), {"positive": 382, "hard_negative": 500})
        self.assertEqual(Counter(row["split"] for row in rows), {"train": 674, "val": 86, "test": 122})
        for row in rows:
            for field in ("open_images_id", "original_landing_url", "license", "author", "sha256"):
                self.assertTrue(row[field], f"missing {field} for {row['open_images_id']}")
            self.assertEqual(row["license"], "https://creativecommons.org/licenses/by/2.0/")
        self.assertEqual(
            hashlib.sha256(attribution_path.read_bytes()).hexdigest(),
            lock["sha256"]["attribution_csv"],
        )
        self.assertEqual(
            hashlib.sha256(source_manifest_path.read_bytes()).hexdigest(),
            lock["sha256"]["source_manifest_json"],
        )
        self.assertEqual(source_manifest["sources"][0]["used_image_count"], len(rows))

    def test_gitignore_exposes_only_public_dataset_metadata(self) -> None:
        ignored_dataset = "training/datasets/public_mvp_v3/dataset.yaml"
        public_metadata = (
            "training/datasets/public_mvp_v3/attribution.csv",
            "training/datasets/public_mvp_v3/source_manifest.json",
        )
        ignored = subprocess.run(
            ["git", "check-ignore", "--quiet", "--no-index", ignored_dataset],
            cwd=ROOT,
            check=False,
        )
        self.assertEqual(ignored.returncode, 0)
        for path in public_metadata:
            visible = subprocess.run(
                ["git", "check-ignore", "--quiet", "--no-index", path],
                cwd=ROOT,
                check=False,
            )
            self.assertEqual(visible.returncode, 1, f"public metadata is ignored: {path}")

    def test_corresponding_source_inventory_exists(self) -> None:
        required = (
            "GolfBallFinder",
            "Tests",
            "project.yml",
            "project.compile-check.yml",
            "codemagic.yaml",
            "scripts/fetch_release_model.py",
            "scripts/inspect_coreml_model.py",
            "training/build_public_dataset.py",
            "training/train.py",
            "training/evaluate.py",
            "training/export_coreml.py",
            "training/public_dataset_sources.yaml",
            "training/public_mvp_v3.lock.json",
            "training/public_mvp_release_v3.json",
            "docs/WINDOWS_CLOUD_BUILD.md",
            "LICENSE",
            "THIRD_PARTY_NOTICES.md",
        )
        missing = [path for path in required if not (ROOT / path).exists()]
        self.assertEqual(missing, [])

    def test_secret_scanner_reports_only_sanitized_metadata(self) -> None:
        sample = "-----BEGIN " + "PRIVATE KEY-----\n" + "not-a-real-secret-value"
        findings = audit_public_release.scan_text(
            sample, "synthetic.txt", "unit-test", None
        )
        self.assertEqual({finding.rule for finding in findings}, {"private_key_material"})
        rendered = "\n".join(finding.sanitized() for finding in findings)
        self.assertNotIn("not-a-real-secret-value", rendered)

        local_path = "C:" + "\\Users\\" + "private-account\\project.txt"
        findings = audit_public_release.scan_text(
            local_path, "synthetic.txt", "unit-test", None
        )
        self.assertEqual({finding.rule for finding in findings}, {"personal_windows_path"})

        blocked = audit_public_release.path_finding(
            "private/AuthKey.p8", "unit-test", "f" * 40
        )
        self.assertIsNotNone(blocked)
        self.assertEqual(blocked.rule, "apple_private_key_file")
        self.assertNotIn("private-account", blocked.sanitized())


if __name__ == "__main__":
    unittest.main()

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tds.evidence import EvidenceRow, build_evidence_bundle, write_evidence_bundle  # noqa: E402


class TestBuildEvidenceBundle(unittest.TestCase):
    def test_json_shape_and_pass_fail_strings(self):
        rows = [
            EvidenceRow("Tokenizer integrity", True, "Manifest record"),
            EvidenceRow("Throughput", False, "Performance report"),
        ]
        evidence_json, evidence_md = build_evidence_bundle(rows)

        self.assertEqual(len(evidence_json["requirements"]), 2)
        self.assertEqual(evidence_json["requirements"][0]["result"], "PASS")
        self.assertEqual(evidence_json["requirements"][1]["result"], "FAIL")
        self.assertEqual(evidence_json["requirements"][0]["evidence"], "Manifest record")

    def test_overall_is_pass_only_when_every_row_passes(self):
        all_pass = [EvidenceRow("a", True, "x"), EvidenceRow("b", True, "y")]
        one_fail = [EvidenceRow("a", True, "x"), EvidenceRow("b", False, "y")]
        self.assertEqual(build_evidence_bundle(all_pass)[0]["overall"], "PASS")
        self.assertEqual(build_evidence_bundle(one_fail)[0]["overall"], "FAIL")

    def test_markdown_contains_a_row_per_requirement(self):
        rows = [EvidenceRow("Tokenizer integrity", True, "Manifest record")]
        _, evidence_md = build_evidence_bundle(rows)
        self.assertIn("Tokenizer integrity", evidence_md)
        self.assertIn("PASS", evidence_md)
        self.assertIn("Manifest record", evidence_md)

    def test_empty_rows_is_vacuously_pass(self):
        evidence_json, _ = build_evidence_bundle([])
        self.assertEqual(evidence_json["overall"], "PASS")
        self.assertEqual(evidence_json["requirements"], [])


class TestWriteEvidenceBundle(unittest.TestCase):
    def test_writes_both_files_with_matching_content(self):
        with tempfile.TemporaryDirectory() as tmp:
            json_path = Path(tmp) / "evidence.json"
            md_path = Path(tmp) / "evidence.md"
            rows = [EvidenceRow("Crash recovery", True, "Expected and resumed batch ids")]

            written = write_evidence_bundle(rows, json_path, md_path)

            self.assertTrue(json_path.exists())
            self.assertTrue(md_path.exists())
            on_disk = json.loads(json_path.read_text())
            self.assertEqual(on_disk, written)
            self.assertIn("Crash recovery", md_path.read_text())

    def test_creates_parent_directories(self):
        with tempfile.TemporaryDirectory() as tmp:
            json_path = Path(tmp) / "nested" / "a" / "evidence.json"
            md_path = Path(tmp) / "nested" / "b" / "evidence.md"
            write_evidence_bundle([EvidenceRow("x", True, "y")], json_path, md_path)
            self.assertTrue(json_path.exists())
            self.assertTrue(md_path.exists())


if __name__ == "__main__":
    unittest.main()

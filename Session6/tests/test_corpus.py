import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tds.corpus import capability_lane_for, load_corpus  # noqa: E402

REAL_CORPUS = ROOT / "data" / "corpus" / "small_shard.parquet"


class TestCapabilityLaneMapping(unittest.TestCase):
    def test_indic_language_wins_over_domain(self):
        # A Hindi stackexchange thread is indic-lane, not code-lane.
        self.assertEqual(capability_lane_for("qa", "hi"), "indic")

    def test_code_domain(self):
        self.assertEqual(capability_lane_for("code", "en"), "code")

    def test_math_and_science_share_a_lane(self):
        self.assertEqual(capability_lane_for("math", "en"), "math_science")
        self.assertEqual(capability_lane_for("science", "en"), "math_science")

    def test_qa_domain_folds_into_code(self):
        # Every "qa"-domain row in the vendored corpus is source=stackexchange
        # and is itself a programming question, not general-knowledge Q&A --
        # see TestLoadCorpus.test_stackexchange_documents_land_in_code_lane.
        self.assertEqual(capability_lane_for("qa", "en"), "code")

    def test_instruction_domain(self):
        self.assertEqual(capability_lane_for("instruction", "en"), "instruction")

    def test_everything_else_falls_back_to_general_web(self):
        for domain in ("web", "news", "encyclopedia", "social"):
            self.assertEqual(capability_lane_for(domain, "en"), "general_web")


class TestLoadCorpus(unittest.TestCase):
    """Lightweight smoke test against the real vendored corpus."""

    @classmethod
    def setUpClass(cls):
        if not REAL_CORPUS.exists():
            raise unittest.SkipTest(f"vendored corpus not found at {REAL_CORPUS}")
        cls.documents = load_corpus(REAL_CORPUS)

    def test_loads_a_reasonable_number_of_documents(self):
        self.assertGreater(len(self.documents), 500)

    def test_no_empty_text(self):
        self.assertTrue(all(doc.text for doc in self.documents))

    def test_document_ids_are_unique(self):
        ids = [doc.document_id for doc in self.documents]
        self.assertEqual(len(ids), len(set(ids)))

    def test_multiple_lanes_present(self):
        lanes = {doc.capability_lane for doc in self.documents}
        self.assertIn("code", lanes)
        self.assertIn("indic", lanes)
        self.assertIn("general_web", lanes)

    def test_indic_lane_is_scarce(self):
        # Worth asserting explicitly: this is the lane later protected-floor
        # logic exists to guard, and it should actually be a minority here.
        lanes = [doc.capability_lane for doc in self.documents]
        indic_share = lanes.count("indic") / len(lanes)
        self.assertLess(indic_share, 0.15)

    def test_stackexchange_documents_land_in_code_lane(self):
        # The corpus's former "qa" domain was, on inspection, entirely
        # source=stackexchange programming questions -- verified directly
        # against the vendored file, not assumed. capability_lane_for folds
        # "qa" into "code" on that basis; confirm it holds for every such
        # document actually in this corpus, not just the domain string.
        stackexchange_docs = [d for d in self.documents if d.source_id == "stackexchange"]
        self.assertGreater(len(stackexchange_docs), 200)
        self.assertTrue(all(d.capability_lane == "code" for d in stackexchange_docs))


if __name__ == "__main__":
    unittest.main()

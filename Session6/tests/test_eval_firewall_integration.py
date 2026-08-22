"""End-to-end: registry -> firewall -> shard builder never sees blocked
content, and the tokenizer trained downstream never sees it either."""

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tds.corpus import Document  # noqa: E402
from tds.eval_firewall import filter_training_documents  # noqa: E402
from tds.eval_registry import EvalRegistry  # noqa: E402
from tds.shard_builder import ShardBuilderConfig, build_shards  # noqa: E402
from tds.tokenizer_utils import train_tokenizer  # noqa: E402

HELD_OUT_TEXT = "Q: What is the boiling point of water at sea level?\nA: 100 degrees Celsius."


def make_documents():
    return [
        Document("doc-000000", "src-0", "toy_qa", "en", "qa", HELD_OUT_TEXT),
        # exact-text duplicate under a totally different id/source/lane
        Document("doc-000001", "src-1", "toy_web_mirror", "en", "general_web", HELD_OUT_TEXT),
        Document("doc-000002", "src-2", "toy_code", "en", "code", "def add(a, b):\n    return a + b\n"),
        Document("doc-000003", "src-3", "toy_math", "en", "math_science", "2 + 2 = 4, a basic arithmetic fact."),
    ]


class TestEvalFirewallIntegration(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.registry = EvalRegistry(Path(self.tmpdir.name) / "eval_registry")
        self.registry.register_text(HELD_OUT_TEXT, benchmark_id="bench-a", version_tag="v1")

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_blocked_documents_never_appear_in_any_shard(self):
        documents = make_documents()
        admitted, blocked = filter_training_documents(documents, self.registry)

        self.assertEqual({e.document_id for e in blocked}, {"doc-000000", "doc-000001"})
        self.assertEqual({d.document_id for d in admitted}, {"doc-000002", "doc-000003"})

        tokenizer_dir = Path(self.tmpdir.name) / "tokenizer"
        tokenizer, tok_manifest = train_tokenizer(
            (d.text for d in admitted), tokenizer_dir, vocab_size=300, min_frequency=1
        )

        config = ShardBuilderConfig(
            shard_token_budget=500,
            shards_dir=str(Path(self.tmpdir.name) / "shards"),
            manifests_dir=str(Path(self.tmpdir.name) / "manifests"),
        )
        manifests = build_shards(admitted, tokenizer, tok_manifest["tokenizer_hash"], config)

        shard_document_ids = {
            span["document_id"] for m in manifests for span in m["document_spans"]
        }
        self.assertEqual(shard_document_ids, {"doc-000002", "doc-000003"})
        self.assertNotIn("doc-000000", shard_document_ids)
        self.assertNotIn("doc-000001", shard_document_ids)


if __name__ == "__main__":
    unittest.main()

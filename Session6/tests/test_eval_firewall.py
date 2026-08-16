import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tds.corpus import Document  # noqa: E402
from tds.eval_firewall import filter_training_documents  # noqa: E402
from tds.eval_registry import EvalRegistry  # noqa: E402


def make_doc(document_id, source_document_id, source_id, lane, text):
    return Document(
        document_id=document_id,
        source_document_id=source_document_id,
        source_id=source_id,
        language="en",
        capability_lane=lane,
        text=text,
    )


class TestEvalFirewall(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.registry = EvalRegistry(Path(self.tmpdir.name))

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_empty_registry_admits_everything(self):
        docs = [
            make_doc("doc-000000", "src-0", "web", "general_web", "some text"),
            make_doc("doc-000001", "src-1", "code", "code", "def f(): pass"),
        ]
        admitted, blocked = filter_training_documents(docs, self.registry)
        self.assertEqual(admitted, docs)
        self.assertEqual(blocked, [])

    def test_document_matching_registry_is_blocked_not_admitted(self):
        held_out_text = "Q: What is the boiling point of water?\nA: 100 degrees Celsius."
        self.registry.register_text(held_out_text, benchmark_id="bench-a", version_tag="v1")

        held_out_doc = make_doc("doc-000000", "src-0", "toy_qa", "qa", held_out_text)
        clean_doc = make_doc("doc-000001", "src-1", "toy_web", "general_web", "unrelated text")

        admitted, blocked = filter_training_documents([held_out_doc, clean_doc], self.registry)

        self.assertEqual(admitted, [clean_doc])
        self.assertEqual(len(blocked), 1)
        self.assertEqual(blocked[0].document_id, "doc-000000")
        self.assertEqual(blocked[0].benchmark_id, "bench-a")

    def test_matching_is_by_content_not_document_id_or_source(self):
        # Same text, completely different id/source/lane -- both must be
        # blocked, since the firewall is checking content, not identity.
        held_out_text = "Q: What is the boiling point of water?\nA: 100 degrees Celsius."
        self.registry.register_text(held_out_text, benchmark_id="bench-a", version_tag="v1")

        original = make_doc("doc-000000", "orig-src", "toy_qa", "qa", held_out_text)
        mirrored = make_doc("doc-000099", "mirror-src", "toy_web_mirror", "general_web", held_out_text)
        unrelated = make_doc("doc-000002", "other-src", "toy_code", "code", "def f(): pass")

        admitted, blocked = filter_training_documents([original, mirrored, unrelated], self.registry)

        self.assertEqual(admitted, [unrelated])
        blocked_ids = {event.document_id for event in blocked}
        self.assertEqual(blocked_ids, {"doc-000000", "doc-000099"})

    def test_admitted_order_is_preserved(self):
        held_out_text = "held out"
        self.registry.register_text(held_out_text, benchmark_id="bench-a", version_tag="v1")

        docs = [
            make_doc("doc-000000", "s0", "src", "general_web", "keep 1"),
            make_doc("doc-000001", "s1", "src", "general_web", held_out_text),
            make_doc("doc-000002", "s2", "src", "general_web", "keep 2"),
        ]
        admitted, _ = filter_training_documents(docs, self.registry)
        self.assertEqual([d.document_id for d in admitted], ["doc-000000", "doc-000002"])


if __name__ == "__main__":
    unittest.main()

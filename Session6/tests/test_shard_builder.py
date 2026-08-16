import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tds.corpus import Document  # noqa: E402
from tds.shard_builder import ShardBuilderConfig, build_shards  # noqa: E402
from tds.tokenizer_utils import sha256_bytes, train_tokenizer  # noqa: E402

WEB_SENTENCE = "the quick brown fox jumps over the lazy dog near the river bank. "
CODE_SNIPPET = "def add(a, b):\n    return a + b\n\ndef sub(a, b):\n    return a - b\n"


def make_synthetic_documents():
    docs = []
    for i in range(6):
        docs.append(
            Document(
                document_id=f"doc-{i:06d}",
                source_document_id=f"src-{i}",
                source_id="toy_web",
                language="en",
                capability_lane="general_web",
                text=WEB_SENTENCE * (5 + i),
            )
        )
    for i in range(6, 12):
        docs.append(
            Document(
                document_id=f"doc-{i:06d}",
                source_document_id=f"src-{i}",
                source_id="toy_code",
                language="en",
                capability_lane="code",
                text=CODE_SNIPPET * (3 + i),
            )
        )
    # one document deliberately much longer than any shard budget we'll use below
    docs.append(
        Document(
            document_id="doc-000012",
            source_document_id="src-12",
            source_id="toy_web",
            language="en",
            capability_lane="general_web",
            text=WEB_SENTENCE * 400,
        )
    )
    return docs


class TestShardBuilder(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.documents = make_synthetic_documents()
        cls.tmpdir = tempfile.TemporaryDirectory()
        tok_dir = Path(cls.tmpdir.name) / "tokenizer"
        cls.tokenizer, cls.tok_manifest = train_tokenizer(
            (d.text for d in cls.documents), tok_dir, vocab_size=1000, min_frequency=1
        )
        cls.tokenizer_hash = cls.tok_manifest["tokenizer_hash"]

    @classmethod
    def tearDownClass(cls):
        cls.tmpdir.cleanup()

    def _build(self, shard_token_budget=300):
        run_dir = tempfile.mkdtemp(dir=self.tmpdir.name)
        config = ShardBuilderConfig(
            shard_token_budget=shard_token_budget,
            shards_dir=str(Path(run_dir) / "shards"),
            manifests_dir=str(Path(run_dir) / "manifests"),
        )
        manifests = build_shards(self.documents, self.tokenizer, self.tokenizer_hash, config)
        return manifests, config

    def test_produces_multiple_shards(self):
        manifests, _ = self._build(shard_token_budget=300)
        self.assertGreater(len(manifests), 2)

    def test_every_shard_is_lane_homogeneous(self):
        manifests, _ = self._build()
        doc_lane = {d.document_id: d.capability_lane for d in self.documents}
        for m in manifests:
            spans_lanes = {doc_lane[s["document_id"]] for s in m["document_spans"]}
            self.assertEqual(spans_lanes, {m["capability_lane"]})

    def test_every_document_appears_in_exactly_one_shard(self):
        manifests, _ = self._build()
        seen = []
        for m in manifests:
            for s in m["document_spans"]:
                seen.append(s["document_id"])
        self.assertEqual(len(seen), len(set(seen)))
        self.assertEqual(set(seen), {d.document_id for d in self.documents})

    def test_document_spans_partition_the_shard_contiguously(self):
        manifests, _ = self._build()
        for m in manifests:
            spans = sorted(m["document_spans"], key=lambda s: s["start_token"])
            self.assertEqual(spans[0]["start_token"], 0)
            for a, b in zip(spans, spans[1:]):
                self.assertEqual(a["end_token"], b["start_token"])
            self.assertEqual(spans[-1]["end_token"], m["token_count"])

    def test_each_document_span_ends_with_eos(self):
        manifests, config = self._build()
        eos_id = self.tokenizer.token_to_id("<eos>")
        for m in manifests:
            arr = np.load(Path(config.shards_dir) / f"{m['shard_id']}.npy")
            for s in m["document_spans"]:
                self.assertEqual(arr[s["end_token"] - 1], eos_id)

    def test_shard_content_hash_matches_file_on_disk(self):
        manifests, config = self._build()
        for m in manifests:
            arr = np.load(Path(config.shards_dir) / f"{m['shard_id']}.npy")
            self.assertEqual(sha256_bytes(arr.tobytes()), m["content_hash"])

    def test_manifest_tokenizer_hash_matches_frozen_tokenizer(self):
        manifests, _ = self._build()
        for m in manifests:
            self.assertEqual(m["tokenizer_hash"], self.tokenizer_hash)

    def test_oversized_single_document_still_gets_its_own_shard(self):
        # doc-000012 alone (400x repeats) is far larger than this budget --
        # it must not be split across two shards.
        manifests, _ = self._build(shard_token_budget=300)
        owning_shards = [
            m for m in manifests if any(s["document_id"] == "doc-000012" for s in m["document_spans"])
        ]
        self.assertEqual(len(owning_shards), 1)
        self.assertEqual(owning_shards[0]["document_count"], 1)
        self.assertGreater(owning_shards[0]["token_count"], 300)

    def test_rebuild_with_same_inputs_is_deterministic(self):
        manifests_a, _ = self._build(shard_token_budget=300)
        manifests_b, _ = self._build(shard_token_budget=300)

        ids_a = [(m["shard_id"], m["content_hash"]) for m in manifests_a]
        ids_b = [(m["shard_id"], m["content_hash"]) for m in manifests_b]
        self.assertEqual(ids_a, ids_b)


if __name__ == "__main__":
    unittest.main()

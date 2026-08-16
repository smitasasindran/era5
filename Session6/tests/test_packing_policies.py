import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tds.corpus import Document  # noqa: E402
from tds.shard_builder import (  # noqa: E402
    ShardBuilderConfig,
    _pack_best_fit,
    _pack_greedy,
    build_shards,
)
from tds.tokenizer_utils import train_tokenizer  # noqa: E402


def fake_doc(doc_id: str) -> Document:
    return Document(
        document_id=doc_id,
        source_document_id=doc_id,
        source_id="synthetic",
        language="en",
        capability_lane="general_web",
        text="unused",
    )


def sized_items(sizes):
    """(doc, ids) pairs with ids whose length is exactly `size`, in order."""
    return [(fake_doc(f"doc-{i:03d}"), list(range(size))) for i, size in enumerate(sizes)]


class TestPackingAlgorithmsInIsolation(unittest.TestCase):
    """Pure bin-packing behavior, with sizes controlled directly (no
    tokenizer involved) so the combinatorial claims are exact and stable."""

    def test_every_item_is_placed_exactly_once(self):
        items = sized_items([51, 50, 51, 50, 51, 50, 5, 5, 5, 40])
        for pack_fn in (_pack_greedy, _pack_best_fit):
            bins = pack_fn(items, budget=100)
            placed = [doc.document_id for shard_bin in bins for doc, _ in shard_bin]
            self.assertEqual(sorted(placed), sorted(d.document_id for d, _ in items))

    def test_no_bin_exceeds_budget_unless_a_single_item_alone_does(self):
        items = sized_items([51, 50, 51, 50, 51, 50, 150, 5])
        for pack_fn in (_pack_greedy, _pack_best_fit):
            bins = pack_fn(items, budget=100)
            for shard_bin in bins:
                total = sum(len(ids) for _, ids in shard_bin)
                if len(shard_bin) > 1:
                    self.assertLessEqual(total, 100)

    def test_oversized_single_item_gets_its_own_bin_under_both_policies(self):
        items = sized_items([150, 10, 10])
        for pack_fn in (_pack_greedy, _pack_best_fit):
            bins = pack_fn(items, budget=100)
            owning = [b for b in bins if any(len(ids) == 150 for _, ids in b)]
            self.assertEqual(len(owning), 1)
            self.assertEqual(len(owning[0]), 1)

    def test_best_fit_beats_greedy_on_an_adversarial_arrival_order(self):
        # Alternating just-over-half-budget sizes: arrival-order greedy is
        # forced to open a fresh bin for every single item (each pairing
        # overflows the 100 budget), while best-fit-decreasing packs a 51
        # with a 40-or-smaller item once sizes are sorted, using fewer bins.
        sizes = [51, 50, 51, 50, 51, 50]
        items = sized_items(sizes)

        greedy_bins = _pack_greedy(items, budget=100)
        best_fit_bins = _pack_best_fit(items, budget=100)

        self.assertEqual(len(greedy_bins), 6)  # every item isolated
        self.assertLess(len(best_fit_bins), len(greedy_bins))

    def test_best_fit_is_deterministic(self):
        items = sized_items([51, 50, 51, 50, 51, 50, 5, 5, 5, 40])
        bins_a = _pack_best_fit(items, budget=100)
        bins_b = _pack_best_fit(items, budget=100)
        ids_a = [[doc.document_id for doc, _ in b] for b in bins_a]
        ids_b = [[doc.document_id for doc, _ in b] for b in bins_b]
        self.assertEqual(ids_a, ids_b)


WEB_SENTENCE = "the quick brown fox jumps over the lazy dog near the river bank. "


class TestBuildShardsWithPackingPolicy(unittest.TestCase):
    """Confirms the policy is correctly wired through build_shards() end to
    end, and that shared invariants hold regardless of which policy runs."""

    @classmethod
    def setUpClass(cls):
        cls.documents = [
            Document(
                document_id=f"doc-{i:06d}",
                source_document_id=f"src-{i}",
                source_id="toy_web",
                language="en",
                capability_lane="general_web",
                text=WEB_SENTENCE * (2 + (i * 7) % 40),  # varied, uneven lengths
            )
            for i in range(20)
        ]
        cls.tmpdir = tempfile.TemporaryDirectory()
        tok_dir = Path(cls.tmpdir.name) / "tokenizer"
        cls.tokenizer, cls.tok_manifest = train_tokenizer(
            (d.text for d in cls.documents), tok_dir, vocab_size=1000, min_frequency=1
        )

    @classmethod
    def tearDownClass(cls):
        cls.tmpdir.cleanup()

    def _build(self, packing_policy):
        run_dir = tempfile.mkdtemp(dir=self.tmpdir.name)
        config = ShardBuilderConfig(
            shard_token_budget=300,
            shards_dir=str(Path(run_dir) / "shards"),
            manifests_dir=str(Path(run_dir) / "manifests"),
            packing_policy=packing_policy,
        )
        manifests = build_shards(
            self.documents, self.tokenizer, self.tok_manifest["tokenizer_hash"], config
        )
        return manifests, config

    def test_rejects_unknown_policy(self):
        config = ShardBuilderConfig(packing_policy="worst_fit")
        with self.assertRaises(ValueError):
            build_shards(self.documents, self.tokenizer, self.tok_manifest["tokenizer_hash"], config)

    def test_manifest_records_the_policy_used(self):
        for policy in ("greedy", "best_fit"):
            manifests, _ = self._build(policy)
            self.assertTrue(all(m["packing_policy"] == policy for m in manifests))

    def test_best_fit_does_not_lose_or_duplicate_documents(self):
        manifests, _ = self._build("best_fit")
        seen = [s["document_id"] for m in manifests for s in m["document_spans"]]
        self.assertEqual(len(seen), len(set(seen)))
        self.assertEqual(set(seen), {d.document_id for d in self.documents})

    def test_best_fit_spans_still_partition_each_shard_contiguously(self):
        manifests, _ = self._build("best_fit")
        for m in manifests:
            spans = sorted(m["document_spans"], key=lambda s: s["start_token"])
            self.assertEqual(spans[0]["start_token"], 0)
            for a, b in zip(spans, spans[1:]):
                self.assertEqual(a["end_token"], b["start_token"])
            self.assertEqual(spans[-1]["end_token"], m["token_count"])

    def test_best_fit_shard_hash_matches_file_on_disk(self):
        from tds.tokenizer_utils import sha256_bytes

        manifests, config = self._build("best_fit")
        for m in manifests:
            arr = np.load(Path(config.shards_dir) / f"{m['shard_id']}.npy")
            self.assertEqual(sha256_bytes(arr.tobytes()), m["content_hash"])

    def test_best_fit_uses_no_more_shards_than_greedy(self):
        greedy_manifests, _ = self._build("greedy")
        best_fit_manifests, _ = self._build("best_fit")
        self.assertLessEqual(len(best_fit_manifests), len(greedy_manifests))


if __name__ == "__main__":
    unittest.main()

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tds.batch_assembler import BatchAssembler, Microbatch, assemble_batches  # noqa: E402
from tds.manifest_store import ManifestStore  # noqa: E402
from tds.packer import Packer  # noqa: E402

from test_packer import make_shard, one_lane_schedule  # noqa: E402


def make_sample(global_step, slot, token_ids):
    """A minimal PackedSample for exercising assemble_batches in isolation,
    without needing a real Packer/shard/manifest fixture."""
    from tds.packer import PackedSample

    n = len(token_ids)
    return PackedSample(
        global_step=global_step,
        slot=slot,
        lane="code",
        packing_policy="concatenate_and_chop",
        sequence_packing_policy="greedy",
        token_ids=tuple(token_ids),
        segment_id=tuple([0] * n),
        position_id=tuple(range(n)),
        loss_mask=tuple([1] * (n - 1) + [0]),
        segment_boundaries=((0, "shard-000000", "doc-0", 0, n, 0, n),),
        loss_mask_hash="sha256:fake",
    )


class TestAssembleBatches(unittest.TestCase):
    def test_slices_in_order_and_stacks_shapes(self):
        samples = [make_sample(0, slot, [slot * 10 + i for i in range(4)]) for slot in range(6)]
        microbatches = assemble_batches(samples, microbatch_size=2)

        self.assertEqual(len(microbatches), 3)
        for mb in microbatches:
            self.assertEqual(mb.token_ids.shape, (2, 4))
            self.assertEqual(mb.segment_id.shape, (2, 4))
            self.assertEqual(mb.position_id.shape, (2, 4))
            self.assertEqual(mb.loss_mask.shape, (2, 4))

        # row i of microbatch k must be samples[k*2 + i] -- order preserved exactly.
        np.testing.assert_array_equal(microbatches[0].token_ids, [[0, 1, 2, 3], [10, 11, 12, 13]])
        np.testing.assert_array_equal(microbatches[1].token_ids, [[20, 21, 22, 23], [30, 31, 32, 33]])
        np.testing.assert_array_equal(microbatches[2].token_ids, [[40, 41, 42, 43], [50, 51, 52, 53]])

    def test_microbatch_index_and_global_step_recorded(self):
        samples = [make_sample(7, slot, [1, 2, 3]) for slot in range(4)]
        microbatches = assemble_batches(samples, microbatch_size=2)
        self.assertEqual([mb.microbatch_index for mb in microbatches], [0, 1])
        self.assertTrue(all(mb.global_step == 7 for mb in microbatches))
        self.assertTrue(all(mb.rank == 0 for mb in microbatches))

    def test_single_microbatch_equals_whole_global_batch(self):
        samples = [make_sample(0, slot, [1, 2, 3]) for slot in range(4)]
        microbatches = assemble_batches(samples, microbatch_size=4)
        self.assertEqual(len(microbatches), 1)
        self.assertEqual(microbatches[0].token_ids.shape, (4, 3))

    def test_provenance_samples_preserved_in_order(self):
        samples = [make_sample(0, slot, [1, 2]) for slot in range(4)]
        microbatches = assemble_batches(samples, microbatch_size=2)
        self.assertEqual(microbatches[0].samples, (samples[0], samples[1]))
        self.assertEqual(microbatches[1].samples, (samples[2], samples[3]))

    def test_loss_mask_and_position_id_values_are_stacked_correctly(self):
        samples = [make_sample(0, 0, [1, 2, 3, 4]), make_sample(0, 1, [5, 6, 7, 8])]
        mb = assemble_batches(samples, microbatch_size=2)[0]
        np.testing.assert_array_equal(mb.loss_mask, [[1, 1, 1, 0], [1, 1, 1, 0]])
        np.testing.assert_array_equal(mb.position_id, [[0, 1, 2, 3], [0, 1, 2, 3]])

    def test_rejects_non_positive_microbatch_size(self):
        samples = [make_sample(0, 0, [1, 2])]
        with self.assertRaises(ValueError):
            assemble_batches(samples, microbatch_size=0)

    def test_rejects_uneven_division(self):
        samples = [make_sample(0, slot, [1, 2]) for slot in range(5)]
        with self.assertRaises(ValueError):
            assemble_batches(samples, microbatch_size=2)

    def test_rejects_mixed_global_steps(self):
        samples = [make_sample(0, 0, [1, 2]), make_sample(1, 1, [1, 2])]
        with self.assertRaises(ValueError):
            assemble_batches(samples, microbatch_size=2)

    def test_empty_input_returns_empty_list(self):
        self.assertEqual(assemble_batches([], microbatch_size=2), [])

    def test_microbatch_equality(self):
        samples = [make_sample(0, slot, [1, 2, 3]) for slot in range(2)]
        a = assemble_batches(samples, microbatch_size=2)[0]
        b = assemble_batches(samples, microbatch_size=2)[0]
        self.assertEqual(a, b)


class TestBatchAssemblerWithRealPacker(unittest.TestCase):
    """End-to-end: assemble_step's microbatches must reconstruct exactly
    what the underlying Packer.pack_step produced, in the same order."""

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        d = Path(self.tmpdir.name)
        self.shards_dir = d / "shards"
        self.store = ManifestStore(d / "manifests")

        docs_lengths = [12, 9, 15, 6, 20, 11, 8, 14]
        offsets = [sum(docs_lengths[:i]) for i in range(len(docs_lengths))]
        tokens = list(range(sum(docs_lengths)))
        spans = [
            {
                "document_id": f"d{i}",
                "start_token": offsets[i],
                "end_token": offsets[i] + docs_lengths[i],
                "response_start_token": None,
            }
            for i in range(len(docs_lengths))
        ]
        make_shard(self.shards_dir, self.store, "shard-000000", "code", tokens, spans)
        pool = {"code": [("shard-000000", f"d{i}", offsets[i]) for i in range(len(docs_lengths))]}
        self.schedule = one_lane_schedule(
            "code", sequence_length=10, available_tokens=sum(docs_lengths), num_steps=3, batch_size=4
        )
        self.packer = Packer("seed-1", self.schedule, pool, self.store, self.shards_dir)

    def test_reconstructs_pack_step_exactly_in_order(self):
        # Two independent, freshly-constructed Packer instances, one driven
        # directly and one through the assembler -- mirrors the replay
        # story: same seed advanced the same way must agree exactly.
        packer_a = Packer("seed-1", self.schedule, self.packer.lane_pools, self.store, self.shards_dir)
        packer_b = Packer("seed-1", self.schedule, self.packer.lane_pools, self.store, self.shards_dir)
        assembler_b = BatchAssembler(packer_b, microbatch_size=2)

        for step in range(3):
            expected = packer_a.pack_step(step)
            microbatches = assembler_b.assemble_step(step)
            reconstructed = [s for mb in microbatches for s in mb.samples]
            self.assertEqual(reconstructed, expected)

    def test_microbatch_shapes_match_stage_sequence_length(self):
        assembler = BatchAssembler(self.packer, microbatch_size=2)
        microbatches = assembler.assemble_step(0)
        self.assertEqual(len(microbatches), 2)  # global_batch_size=4 / microbatch_size=2
        for mb in microbatches:
            self.assertEqual(mb.token_ids.shape, (2, 10))  # sequence_length=10


if __name__ == "__main__":
    unittest.main()

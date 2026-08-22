import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tds.batch_assembler import BatchAssembler  # noqa: E402
from tds.manifest_store import ManifestStore  # noqa: E402
from tds.model import ToyTransformer, ToyTransformerConfig, attention_bias_for_microbatch, compute_batch_loss  # noqa: E402
from tds.packer import Packer, attention_bias_from_segments  # noqa: E402

from test_packer import make_shard, one_lane_schedule  # noqa: E402


class TestToyTransformerDeterminism(unittest.TestCase):
    def test_same_seed_gives_identical_initial_weights(self):
        config = ToyTransformerConfig(vocab_size=50, max_sequence_length=16)
        model_a = ToyTransformer(config, seed=42)
        model_b = ToyTransformer(config, seed=42)
        for pa, pb in zip(model_a.parameters(), model_b.parameters()):
            self.assertTrue(torch.equal(pa, pb))

    def test_different_seeds_give_different_weights(self):
        config = ToyTransformerConfig(vocab_size=50, max_sequence_length=16)
        model_a = ToyTransformer(config, seed=1)
        model_b = ToyTransformer(config, seed=2)
        differs = any(not torch.equal(pa, pb) for pa, pb in zip(model_a.parameters(), model_b.parameters()))
        self.assertTrue(differs)


class TestAttentionBiasForMicrobatch(unittest.TestCase):
    def test_matches_attention_bias_from_segments(self):
        from tds.batch_assembler import Microbatch

        segment_id = np.array([[0, 0, 1, 1], [0, 1, 1, 1]])
        mb = Microbatch(
            global_step=0,
            microbatch_index=0,
            rank=0,
            token_ids=np.zeros((2, 4), dtype=np.int64),
            segment_id=segment_id,
            position_id=np.zeros((2, 4), dtype=np.int64),
            loss_mask=np.ones((2, 4), dtype=np.float32),
            samples=(),
        )
        bias = attention_bias_for_microbatch(mb)
        self.assertEqual(tuple(bias.shape), (2, 1, 4, 4))

        for b in range(2):
            expected_bool = attention_bias_from_segments(segment_id[b])
            allowed = bias[b, 0].numpy() == 0.0
            np.testing.assert_array_equal(allowed, expected_bool)


class TestComputeBatchLossWithRealPacker(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        d = Path(self.tmpdir.name)
        self.shards_dir = d / "shards"
        self.store = ManifestStore(d / "manifests")

        make_shard(
            self.shards_dir,
            self.store,
            "shard-000000",
            "code",
            list(range(20)),
            [{"document_id": "d0", "start_token": 0, "end_token": 20, "response_start_token": None}],
        )
        pool = {"code": [("shard-000000", "d0", 0)]}
        schedule = one_lane_schedule("code", sequence_length=8, available_tokens=20, num_steps=1, batch_size=2)
        packer = Packer("seed-1", schedule, pool, self.store, self.shards_dir)
        self.assembler = BatchAssembler(packer, microbatch_size=2)
        self.config = ToyTransformerConfig(vocab_size=32, max_sequence_length=8, d_model=8, n_layers=1, n_heads=2, d_ff=16)

    def test_output_shapes_and_avg_matches_manual_computation(self):
        model = ToyTransformer(self.config, seed=0)
        mb = self.assembler.assemble_step(0)[0]
        per_token_loss, avg_loss = compute_batch_loss(model, mb)

        self.assertEqual(per_token_loss.shape, (2, 7))  # (microbatch_size, sequence_length - 1)
        mask = mb.loss_mask[:, :-1]
        expected_avg = per_token_loss.sum() / max(mask.sum(), 1)
        self.assertAlmostEqual(avg_loss, float(expected_avg), places=5)

    def test_avg_loss_is_a_plain_float_and_is_finite(self):
        model = ToyTransformer(self.config, seed=0)
        mb = self.assembler.assemble_step(0)[0]
        _, avg_loss = compute_batch_loss(model, mb)
        self.assertIsInstance(avg_loss, float)
        self.assertTrue(np.isfinite(avg_loss))

    def test_deterministic_across_two_fresh_models(self):
        model_a = ToyTransformer(self.config, seed=7)
        model_b = ToyTransformer(self.config, seed=7)
        mb = self.assembler.assemble_step(0)[0]
        loss_a, avg_a = compute_batch_loss(model_a, mb)
        loss_b, avg_b = compute_batch_loss(model_b, mb)
        np.testing.assert_array_almost_equal(loss_a, loss_b)
        self.assertAlmostEqual(avg_a, avg_b)


if __name__ == "__main__":
    unittest.main()

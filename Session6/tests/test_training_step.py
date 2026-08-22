import sys
import tempfile
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tds.batch_assembler import BatchAssembler  # noqa: E402
from tds.manifest_store import ManifestStore  # noqa: E402
from tds.model import ToyTransformer, ToyTransformerConfig  # noqa: E402
from tds.packer import Packer  # noqa: E402
from tds.training_step import run_training_step  # noqa: E402

from test_packer import make_shard, one_lane_schedule  # noqa: E402


class TrainingStepFixture(unittest.TestCase):
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
            list(range(40)),
            [{"document_id": "d0", "start_token": 0, "end_token": 40, "response_start_token": None}],
        )
        self.pool = {"code": [("shard-000000", "d0", 0)]}
        self.schedule = one_lane_schedule(
            "code", sequence_length=8, available_tokens=40, num_steps=2, batch_size=4
        )
        self.config = ToyTransformerConfig(
            vocab_size=48, max_sequence_length=8, d_model=8, n_layers=1, n_heads=2, d_ff=16
        )

    def make_assembler(self, microbatch_size=2):
        packer = Packer("seed-1", self.schedule, self.pool, self.store, self.shards_dir)
        return BatchAssembler(packer, microbatch_size=microbatch_size)


class TestRunTrainingStep(TrainingStepFixture):
    def test_rejects_empty_microbatches(self):
        model = ToyTransformer(self.config, seed=0)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        with self.assertRaises(ValueError):
            run_training_step(model, optimizer, [])

    def test_rejects_mixed_global_steps(self):
        assembler = self.make_assembler()
        mixed = assembler.assemble_step(0) + assembler.assemble_step(1)
        model = ToyTransformer(self.config, seed=0)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        with self.assertRaises(ValueError):
            run_training_step(model, optimizer, mixed)

    def test_optimizer_step_actually_changes_weights(self):
        assembler = self.make_assembler()
        microbatches = assembler.assemble_step(0)
        model = ToyTransformer(self.config, seed=0)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-1)
        before = [p.clone() for p in model.parameters()]

        run_training_step(model, optimizer, microbatches)

        after = list(model.parameters())
        changed = any(not torch.equal(b, a) for b, a in zip(before, after))
        self.assertTrue(changed)

    def test_result_fields_are_well_formed(self):
        assembler = self.make_assembler()
        microbatches = assembler.assemble_step(0)
        model = ToyTransformer(self.config, seed=0)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

        result = run_training_step(model, optimizer, microbatches)
        self.assertEqual(result.global_step, 0)
        self.assertEqual(len(result.microbatches), len(microbatches))
        self.assertEqual(len(result.per_token_loss_before), len(microbatches))
        self.assertEqual(len(result.per_token_loss_after), len(microbatches))
        self.assertIsInstance(result.avg_loss_before, float)
        self.assertIsInstance(result.avg_loss_after, float)

    def test_repeated_exposure_at_a_large_learning_rate_reduces_loss(self):
        # A large learning rate on the *same* data the model was just
        # shown should reliably reduce that data's own loss -- the
        # "before vs after exposure" signal the Learning Ledger exists to
        # capture, made unambiguous by exaggerating the step size.
        assembler = self.make_assembler()
        microbatches = assembler.assemble_step(0)
        model = ToyTransformer(self.config, seed=0)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.5)

        result = run_training_step(model, optimizer, microbatches)
        self.assertLess(result.avg_loss_after, result.avg_loss_before)


class TestDeterminism(TrainingStepFixture):
    def test_two_identical_runs_produce_identical_losses(self):
        assembler_a = self.make_assembler()
        assembler_b = self.make_assembler()
        microbatches_a = assembler_a.assemble_step(0)
        microbatches_b = assembler_b.assemble_step(0)

        model_a = ToyTransformer(self.config, seed=3)
        model_b = ToyTransformer(self.config, seed=3)
        optimizer_a = torch.optim.SGD(model_a.parameters(), lr=1e-2)
        optimizer_b = torch.optim.SGD(model_b.parameters(), lr=1e-2)

        result_a = run_training_step(model_a, optimizer_a, microbatches_a)
        result_b = run_training_step(model_b, optimizer_b, microbatches_b)

        self.assertAlmostEqual(result_a.avg_loss_before, result_b.avg_loss_before, places=6)
        self.assertAlmostEqual(result_a.avg_loss_after, result_b.avg_loss_after, places=6)
        for pa, pb in zip(model_a.parameters(), model_b.parameters()):
            self.assertTrue(torch.allclose(pa, pb, atol=1e-6))


if __name__ == "__main__":
    unittest.main()

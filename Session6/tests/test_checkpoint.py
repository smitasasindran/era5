import sys
import tempfile
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tds.checkpoint import CheckpointManager, next_step_after_checkpoint  # noqa: E402
from tds.model import ToyTransformer, ToyTransformerConfig  # noqa: E402


def make_model_and_optimizer(seed=0, lr=1e-2):
    config = ToyTransformerConfig(vocab_size=32, max_sequence_length=8, d_model=8, n_layers=1, n_heads=2, d_ff=16)
    model = ToyTransformer(config, seed=seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    return model, optimizer


class TestCheckpointManager(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        self.manager = CheckpointManager(self.tmpdir.name)

    def test_save_then_restore_roundtrips_weights_and_metadata(self):
        model_a, optimizer_a = make_model_and_optimizer(seed=1)
        self.manager.save(model_a, optimizer_a, "run-a", "main", global_step=5)

        model_b, optimizer_b = make_model_and_optimizer(seed=999)  # deliberately different
        metadata = self.manager.restore("run-a", "main", 5, model_b, optimizer_b)

        self.assertEqual(metadata.run_id, "run-a")
        self.assertEqual(metadata.branch_id, "main")
        self.assertEqual(metadata.global_step, 5)
        self.assertIsNone(metadata.parent_branch_id)
        self.assertIsNone(metadata.fork_step)

        for pa, pb in zip(model_a.parameters(), model_b.parameters()):
            self.assertTrue(torch.equal(pa, pb))

    def test_optimizer_state_is_restored_not_just_model_weights(self):
        model_a, optimizer_a = make_model_and_optimizer(seed=1)
        # Take one real step so the optimizer accumulates real state (Adam's
        # per-parameter moment estimates), not just its freshly-initialized defaults.
        loss = model_a.token_embedding.weight.sum()
        loss.backward()
        optimizer_a.step()
        self.manager.save(model_a, optimizer_a, "run-a", "main", global_step=1)

        model_b, optimizer_b = make_model_and_optimizer(seed=1)
        self.manager.restore("run-a", "main", 1, model_b, optimizer_b)

        state_a = optimizer_a.state_dict()["state"]
        state_b = optimizer_b.state_dict()["state"]
        self.assertEqual(set(state_a.keys()), set(state_b.keys()))
        for key in state_a:
            self.assertTrue(torch.equal(state_a[key]["exp_avg"], state_b[key]["exp_avg"]))

    def test_load_missing_checkpoint_raises(self):
        with self.assertRaises(FileNotFoundError):
            self.manager.load("run-a", "main", 0)

    def test_atomic_save_leaves_no_tmp_file_behind(self):
        model, optimizer = make_model_and_optimizer()
        path = self.manager.save(model, optimizer, "run-a", "main", global_step=0)
        self.assertTrue(path.exists())
        tmp_files = list(Path(self.tmpdir.name).rglob("*.tmp"))
        self.assertEqual(tmp_files, [])

    def test_latest_step_returns_max_and_none_when_empty(self):
        self.assertIsNone(self.manager.latest_step("run-a", "main"))
        model, optimizer = make_model_and_optimizer()
        self.manager.save(model, optimizer, "run-a", "main", global_step=3)
        self.manager.save(model, optimizer, "run-a", "main", global_step=7)
        self.assertEqual(self.manager.latest_step("run-a", "main"), 7)

    def test_latest_step_is_isolated_per_branch(self):
        model, optimizer = make_model_and_optimizer()
        self.manager.save(model, optimizer, "run-a", "main", global_step=10)
        self.manager.save(model, optimizer, "run-a", "fork-1", global_step=2)
        self.assertEqual(self.manager.latest_step("run-a", "main"), 10)
        self.assertEqual(self.manager.latest_step("run-a", "fork-1"), 2)

    def test_fork_metadata_roundtrips_when_provided(self):
        model, optimizer = make_model_and_optimizer()
        self.manager.save(
            model, optimizer, "run-a", "fork-1", global_step=10, parent_branch_id="main", fork_step=10
        )
        metadata = self.manager.restore("run-a", "fork-1", 10, model, optimizer)
        self.assertEqual(metadata.parent_branch_id, "main")
        self.assertEqual(metadata.fork_step, 10)


class TestNextStepAfterCheckpoint(unittest.TestCase):
    def test_is_exactly_one_past_the_checkpointed_step(self):
        from tds.checkpoint import CheckpointMetadata

        metadata = CheckpointMetadata(run_id="run-a", branch_id="main", global_step=41)
        self.assertEqual(next_step_after_checkpoint(metadata), 42)


if __name__ == "__main__":
    unittest.main()

import sys
import tempfile
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tds.batch_assembler import BatchAssembler  # noqa: E402
from tds.checkpoint import CheckpointManager  # noqa: E402
from tds.consumption_ledger import ConsumptionLedger, build_ledger_entry  # noqa: E402
from tds.fork import fork_branch  # noqa: E402
from tds.manifest_store import ManifestStore  # noqa: E402
from tds.mixture_compiler import MixtureStage, compile_curriculum  # noqa: E402
from tds.model import ToyTransformer, ToyTransformerConfig  # noqa: E402
from tds.packer import Packer  # noqa: E402
from tds.training_step import run_training_step  # noqa: E402

from test_packer import make_shard  # noqa: E402


def make_model_and_optimizer(seed=0, lr=1e-2):
    config = ToyTransformerConfig(vocab_size=180, max_sequence_length=8, d_model=8, n_layers=1, n_heads=2, d_ff=16)
    model = ToyTransformer(config, seed=seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    return model, optimizer


class TestForkBranchMechanics(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        self.checkpoints = CheckpointManager(Path(self.tmpdir.name) / "checkpoints")
        self.model, self.optimizer = make_model_and_optimizer(seed=1)
        self.checkpoints.save(self.model, self.optimizer, "run-a", "main", global_step=10)

    def test_fork_creates_a_checkpoint_with_lineage_metadata(self):
        fork_model, fork_optimizer = make_model_and_optimizer(seed=999)
        result = fork_branch(self.checkpoints, "run-a", "main", 10, "fork-1", fork_model, fork_optimizer)

        self.assertEqual(result.run_id, "run-a")
        self.assertEqual(result.parent_branch_id, "main")
        self.assertEqual(result.fork_step, 10)
        self.assertEqual(result.new_branch_id, "fork-1")
        self.assertTrue(result.checkpoint_path.exists())

        metadata = self.checkpoints.load("run-a", "fork-1", 10)
        self.assertEqual(metadata["parent_branch_id"], "main")
        self.assertEqual(metadata["fork_step"], 10)
        self.assertEqual(metadata["branch_id"], "fork-1")

    def test_fork_restores_the_parents_weights_into_the_new_branch(self):
        snapshot = [p.clone() for p in self.model.parameters()]
        fork_model, fork_optimizer = make_model_and_optimizer(seed=999)  # different, must be overwritten
        fork_branch(self.checkpoints, "run-a", "main", 10, "fork-1", fork_model, fork_optimizer)

        for expected, actual in zip(snapshot, fork_model.parameters()):
            self.assertTrue(torch.equal(expected, actual))

    def test_rejects_forking_to_the_same_branch_id(self):
        with self.assertRaises(ValueError):
            fork_branch(self.checkpoints, "run-a", "main", 10, "main", self.model, self.optimizer)


class TestBranchesDivergeAfterFork(unittest.TestCase):
    """The centerpiece property §5.13 exists to guarantee: two branches
    sharing a checkpoint produce genuinely different subsequent data when
    given a different seed, while the checkpoint itself proves exactly
    where and from what they diverged."""

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        d = Path(self.tmpdir.name)
        self.shards_dir = d / "shards"
        self.store = ManifestStore(d / "manifests")

        make_shard(
            self.shards_dir, self.store, "shard-000000", "code", list(range(60)),
            [
                {"document_id": "c0", "start_token": 0, "end_token": 20, "response_start_token": None},
                {"document_id": "c1", "start_token": 20, "end_token": 40, "response_start_token": None},
                {"document_id": "c2", "start_token": 40, "end_token": 60, "response_start_token": None},
            ],
        )
        make_shard(
            self.shards_dir, self.store, "shard-000001", "qa", list(range(100, 140)),
            [
                {"document_id": "q0", "start_token": 0, "end_token": 20, "response_start_token": None},
                {"document_id": "q1", "start_token": 20, "end_token": 40, "response_start_token": None},
            ],
        )
        self.lane_pools = self.store.document_pool_by_lane()

        stages = [
            MixtureStage(
                stage="mixed", token_start=0, token_end=400, sequence_length=8,
                mixture={"code": 0.5, "qa": 0.5},
            )
        ]
        self.schedule = compile_curriculum(
            stages, {"code": 60, "qa": 40}, global_batch_size=4, scarcity_policy="repeat"
        )
        self.microbatch_size = 2
        self.tokenizer_hash = "sha256:tok"
        self.run_id = "run-fork-test"

    def test_branches_share_history_up_to_fork_and_diverge_after(self):
        fork_step = 2
        seed_main = "seed-main"
        seed_fork = "seed-fork-different"

        ledger = ConsumptionLedger(Path(self.tmpdir.name) / "consumption")
        checkpoints = CheckpointManager(Path(self.tmpdir.name) / "checkpoints")

        # ---- Parent branch: run through fork_step, checkpoint, then keep going on its own seed ----
        main_model, main_optimizer = make_model_and_optimizer(seed=0)
        main_packer = Packer(seed_main, self.schedule, self.lane_pools, self.store, self.shards_dir)
        main_assembler = BatchAssembler(main_packer, self.microbatch_size)

        for step in range(fork_step + 1):
            mbs = main_assembler.assemble_step(step)
            run_training_step(main_model, main_optimizer, mbs)
            for mb in mbs:
                ledger.append(build_ledger_entry(self.run_id, "main", mb, self.schedule, self.tokenizer_hash))
        checkpoints.save(main_model, main_optimizer, self.run_id, "main", fork_step)
        snapshot_at_fork = [p.clone() for p in main_model.parameters()]

        for step in range(fork_step + 1, fork_step + 3):
            mbs = main_assembler.assemble_step(step)
            run_training_step(main_model, main_optimizer, mbs)
            for mb in mbs:
                ledger.append(build_ledger_entry(self.run_id, "main", mb, self.schedule, self.tokenizer_hash))

        # ---- Fork: brand-new model/optimizer, restore, diverge on a different seed ----
        fork_model, fork_optimizer = make_model_and_optimizer(seed=777)
        fork_result = fork_branch(
            checkpoints, self.run_id, "main", fork_step, "fork-1", fork_model, fork_optimizer
        )
        for expected, actual in zip(snapshot_at_fork, fork_model.parameters()):
            self.assertTrue(torch.equal(expected, actual))

        fork_packer = Packer(seed_fork, self.schedule, self.lane_pools, self.store, self.shards_dir)
        fork_assembler = BatchAssembler(fork_packer, self.microbatch_size)
        # Replay to the fork point first -- required for the same reason
        # resume needs it: Packer document-consumption position is
        # cumulative, not directly indexable by step number.
        for step in range(fork_step + 1):
            fork_assembler.assemble_step(step)
        for step in range(fork_step + 1, fork_step + 3):
            mbs = fork_assembler.assemble_step(step)
            run_training_step(fork_model, fork_optimizer, mbs)
            for mb in mbs:
                ledger.append(build_ledger_entry(self.run_id, "fork-1", mb, self.schedule, self.tokenizer_hash))

        # ---- Assertions ----
        metadata = checkpoints.load(self.run_id, "fork-1", fork_step)
        self.assertEqual(metadata["parent_branch_id"], "main")
        self.assertEqual(metadata["fork_step"], fork_step)

        main_after = [e for e in ledger.for_branch(self.run_id, "main") if e["global_step"] > fork_step]
        fork_after = [e for e in ledger.for_branch(self.run_id, "fork-1") if e["global_step"] > fork_step]
        self.assertEqual(len(main_after), len(fork_after))
        # A different seed drives a genuinely different data stream post-fork.
        main_signature = [(e["mixture_lane"], e["token_span_ids"]) for e in main_after]
        fork_signature = [(e["mixture_lane"], e["token_span_ids"]) for e in fork_after]
        self.assertNotEqual(main_signature, fork_signature)

        # The two branches remain independently distinguishable in the
        # ledger even though they share a run_id and a fork point: "fork-1"
        # has no entries at or before the fork step -- those live only
        # under "main", the branch that actually served them.
        fork_up_to_and_including_fork_point = [
            e for e in ledger.for_branch(self.run_id, "fork-1") if e["global_step"] <= fork_step
        ]
        self.assertEqual(fork_up_to_and_including_fork_point, [])


if __name__ == "__main__":
    unittest.main()

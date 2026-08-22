"""The centerpiece test: resuming from a checkpoint reproduces the exact
next batch an uninterrupted "control" run already produced, and does so
without ever touching the control run's live objects -- proving the
mechanism is genuine recompute-from-scratch, not a disguised continuation
of in-memory state. This is the property the assignment's crash-recovery
requirement ("prove that the next batch is exactly the expected batch")
rests on.
"""

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
from tds.manifest_store import ManifestStore  # noqa: E402
from tds.model import ToyTransformer, ToyTransformerConfig  # noqa: E402
from tds.packer import Packer  # noqa: E402
from tds.checkpoint import next_step_after_checkpoint  # noqa: E402
from tds.resume import verify_resume  # noqa: E402
from tds.training_step import run_training_step  # noqa: E402

from test_packer import make_shard  # noqa: E402


class TestFullCrashResumeCycle(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        d = Path(self.tmpdir.name)
        self.shards_dir = d / "shards"
        self.store = ManifestStore(d / "manifests")

        # Two lanes, several documents each -- enough real variety that a
        # single wrong seed or off-by-one step would visibly diverge.
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

        from tds.mixture_compiler import MixtureStage, compile_curriculum

        stages = [
            MixtureStage(
                stage="mixed", token_start=0, token_end=400, sequence_length=8,
                mixture={"code": 0.5, "qa": 0.5},
            )
        ]
        self.schedule = compile_curriculum(
            stages, {"code": 60, "qa": 40}, global_batch_size=4, scarcity_policy="repeat"
        )
        self.seed = "crash-resume-seed"
        self.microbatch_size = 2
        self.tokenizer_hash = "sha256:tok"
        self.run_id = "run-crash-test"
        self.branch_id = "main"
        self.config = ToyTransformerConfig(
            vocab_size=180, max_sequence_length=8, d_model=8, n_layers=1, n_heads=2, d_ff=16
        )

    def test_resume_after_simulated_crash_matches_control_run_and_restores_weights(self):
        crash_after_step = 2
        resume_step = crash_after_step + 1

        # ---- Control run: uninterrupted, writes the full consumption ledger ----
        control_model = ToyTransformer(self.config, seed=0)
        control_optimizer = torch.optim.Adam(control_model.parameters(), lr=1e-2)
        control_packer = Packer(self.seed, self.schedule, self.lane_pools, self.store, self.shards_dir)
        control_assembler = BatchAssembler(control_packer, self.microbatch_size)
        control_ledger = ConsumptionLedger(Path(self.tmpdir.name) / "consumption")
        checkpoints = CheckpointManager(Path(self.tmpdir.name) / "checkpoints")

        snapshot_at_crash = None
        for step in range(resume_step + 1):
            microbatches = control_assembler.assemble_step(step)
            run_training_step(control_model, control_optimizer, microbatches)
            for mb in microbatches:
                control_ledger.append(
                    build_ledger_entry(self.run_id, self.branch_id, mb, self.schedule, self.tokenizer_hash)
                )
            if step == crash_after_step:
                checkpoints.save(control_model, control_optimizer, self.run_id, self.branch_id, step)
                snapshot_at_crash = [p.clone() for p in control_model.parameters()]
                # From here on, pretend the control process is gone. We never
                # touch control_model/control_optimizer/control_packer again
                # for the resumed side -- only control_ledger, which is what
                # a real resumed process reads/appends to under the same
                # run_id/branch_id (never control_packer or control_model).

        # ---- Resume: brand-new objects, exactly as a genuinely new process would build ----
        resumed_model = ToyTransformer(self.config, seed=12345)  # deliberately different seed
        resumed_optimizer = torch.optim.Adam(resumed_model.parameters(), lr=1e-2)
        metadata = checkpoints.restore(
            self.run_id, self.branch_id, crash_after_step, resumed_model, resumed_optimizer
        )

        self.assertEqual(metadata.run_id, self.run_id)
        self.assertEqual(metadata.branch_id, self.branch_id)
        self.assertEqual(metadata.global_step, crash_after_step)

        # Weights genuinely restored to the exact point of the crash --
        # not left over from resumed_model's own (different) initial seed.
        for expected, actual in zip(snapshot_at_crash, resumed_model.parameters()):
            self.assertTrue(torch.equal(expected, actual))

        # Recompute-don't-restore: verify_resume never touches control_packer
        # or control_model, only the frozen config + the control ledger.
        result = verify_resume(
            self.run_id, self.branch_id, next_step_after_checkpoint(metadata),
            self.seed, self.schedule, self.lane_pools, self.store, self.shards_dir,
            self.microbatch_size, self.tokenizer_hash, control_ledger,
        )
        self.assertTrue(result.matched, result.mismatches)
        self.assertEqual(result.global_step, resume_step)
        self.assertEqual(result.mismatches, [])

    def test_verify_resume_detects_a_genuine_mismatch(self):
        # A different seed produces a different data stream -- verify_resume
        # must catch that, not rubber-stamp it. Proves the check has teeth.
        control_ledger = ConsumptionLedger(Path(self.tmpdir.name) / "consumption")
        packer = Packer(self.seed, self.schedule, self.lane_pools, self.store, self.shards_dir)
        assembler = BatchAssembler(packer, self.microbatch_size)
        for mb in assembler.assemble_step(0):
            control_ledger.append(
                build_ledger_entry(self.run_id, self.branch_id, mb, self.schedule, self.tokenizer_hash)
            )

        result = verify_resume(
            self.run_id, self.branch_id, 0,
            "a-completely-different-seed", self.schedule, self.lane_pools, self.store, self.shards_dir,
            self.microbatch_size, self.tokenizer_hash, control_ledger,
        )
        self.assertFalse(result.matched)
        self.assertTrue(result.mismatches)

    def test_verify_resume_reports_missing_control_entry(self):
        control_ledger = ConsumptionLedger(Path(self.tmpdir.name) / "consumption")  # empty
        result = verify_resume(
            self.run_id, self.branch_id, 0,
            self.seed, self.schedule, self.lane_pools, self.store, self.shards_dir,
            self.microbatch_size, self.tokenizer_hash, control_ledger,
        )
        self.assertFalse(result.matched)
        self.assertIn("no recorded entry to compare against", result.mismatches[0])


if __name__ == "__main__":
    unittest.main()

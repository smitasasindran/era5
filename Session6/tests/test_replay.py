import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tds.batch_assembler import BatchAssembler  # noqa: E402
from tds.consumption_ledger import ConsumptionLedger, build_ledger_entry  # noqa: E402
from tds.manifest_store import ManifestStore  # noqa: E402
from tds.mixture_compiler import MixtureStage, compile_curriculum  # noqa: E402
from tds.packer import Packer  # noqa: E402
from tds.replay import replay_range  # noqa: E402

from test_packer import make_shard  # noqa: E402


class TestReplayRange(unittest.TestCase):
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
        self.seed = "replay-test-seed"
        self.microbatch_size = 2
        self.tokenizer_hash = "sha256:tok"
        self.run_id = "run-a"
        self.branch_id = "main"

    def _write_control_ledger(self, num_steps):
        ledger = ConsumptionLedger(Path(self.tmpdir.name) / "consumption")
        packer = Packer(self.seed, self.schedule, self.lane_pools, self.store, self.shards_dir)
        assembler = BatchAssembler(packer, self.microbatch_size)
        for step in range(num_steps):
            for mb in assembler.assemble_step(step):
                ledger.append(build_ledger_entry(self.run_id, self.branch_id, mb, self.schedule, self.tokenizer_hash))
        return ledger

    def test_replaying_a_fully_recorded_range_matches(self):
        ledger = self._write_control_ledger(num_steps=6)
        result = replay_range(
            self.run_id, self.branch_id, 2, 5, self.seed, self.schedule,
            self.lane_pools, self.store, self.shards_dir, self.microbatch_size, self.tokenizer_hash, ledger,
        )
        self.assertTrue(result.matched)
        self.assertEqual(result.start_step, 2)
        self.assertEqual(result.end_step, 5)
        self.assertEqual([r.global_step for r in result.step_results], [2, 3, 4])
        self.assertEqual(result.mismatched_steps, [])

    def test_replaying_from_step_zero_matches(self):
        ledger = self._write_control_ledger(num_steps=3)
        result = replay_range(
            self.run_id, self.branch_id, 0, 3, self.seed, self.schedule,
            self.lane_pools, self.store, self.shards_dir, self.microbatch_size, self.tokenizer_hash, ledger,
        )
        self.assertTrue(result.matched)

    def test_a_different_seed_is_detected_across_the_whole_range(self):
        ledger = self._write_control_ledger(num_steps=6)
        result = replay_range(
            self.run_id, self.branch_id, 0, 6, "a-totally-different-seed", self.schedule,
            self.lane_pools, self.store, self.shards_dir, self.microbatch_size, self.tokenizer_hash, ledger,
        )
        self.assertFalse(result.matched)
        self.assertTrue(result.mismatched_steps)

    def test_range_extending_past_recorded_history_reports_missing_entries(self):
        ledger = self._write_control_ledger(num_steps=3)  # only steps 0..2 recorded
        result = replay_range(
            self.run_id, self.branch_id, 0, 5, self.seed, self.schedule,
            self.lane_pools, self.store, self.shards_dir, self.microbatch_size, self.tokenizer_hash, ledger,
        )
        self.assertFalse(result.matched)
        self.assertIn(3, result.mismatched_steps)
        self.assertIn(4, result.mismatched_steps)
        # steps actually recorded should still match
        matched_steps = [r.global_step for r in result.step_results if r.matched]
        self.assertEqual(matched_steps, [0, 1, 2])

    def test_rejects_invalid_ranges(self):
        ledger = self._write_control_ledger(num_steps=1)
        with self.assertRaises(ValueError):
            replay_range(
                self.run_id, self.branch_id, 3, 3, self.seed, self.schedule,
                self.lane_pools, self.store, self.shards_dir, self.microbatch_size, self.tokenizer_hash, ledger,
            )
        with self.assertRaises(ValueError):
            replay_range(
                self.run_id, self.branch_id, -1, 2, self.seed, self.schedule,
                self.lane_pools, self.store, self.shards_dir, self.microbatch_size, self.tokenizer_hash, ledger,
            )


if __name__ == "__main__":
    unittest.main()

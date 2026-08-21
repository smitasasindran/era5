import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tds.audit import (  # noqa: E402
    StepTiming,
    audit_range,
    exact_useful_tokens_range,
    mixture_compliance_report,
    throughput_report,
)
from tds.consumption_ledger import ConsumptionLedger, build_ledger_entry  # noqa: E402
from tds.batch_assembler import BatchAssembler  # noqa: E402
from tds.manifest_store import ManifestStore  # noqa: E402
from tds.mixture_compiler import MixtureStage, compile_curriculum  # noqa: E402
from tds.packer import Packer  # noqa: E402

from test_packer import EOS, make_shard, one_lane_schedule  # noqa: E402


class AuditFixture(unittest.TestCase):
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
        self.run_id = "run-a"
        self.branch_id = "main"

        self.ledger = ConsumptionLedger(d / "consumption")
        packer = Packer("seed-1", self.schedule, self.lane_pools, self.store, self.shards_dir)
        assembler = BatchAssembler(packer, self.microbatch_size)
        for step in range(6):
            for mb in assembler.assemble_step(step):
                self.ledger.append(build_ledger_entry(self.run_id, self.branch_id, mb, self.schedule, "sha256:tok"))


class TestAuditRange(AuditFixture):
    def test_reports_shards_lanes_and_stages_touched(self):
        report = audit_range(self.run_id, self.branch_id, 0, 6, self.ledger, self.schedule)
        self.assertEqual(report["shards_touched"], ["shard-000000", "shard-000001"])
        self.assertEqual(report["lanes_touched"], ["code", "qa"])
        self.assertEqual(report["curriculum_stages_touched"], ["mixed"])
        self.assertEqual(report["total_samples"], 6 * 4)  # 6 steps x global_batch_size=4

    def test_packing_utilization_is_exactly_one_by_design(self):
        # This project's Packer never pads -- every window is always
        # exactly full, which this check verifies purely from ledger
        # token-span records, no recomputation.
        report = audit_range(self.run_id, self.branch_id, 0, 6, self.ledger, self.schedule)
        self.assertAlmostEqual(report["packing_utilization"], 1.0)
        self.assertEqual(report["total_capacity_tokens"], report["total_served_tokens"])

    def test_estimated_useful_tokens_excludes_one_position_per_sample(self):
        report = audit_range(self.run_id, self.branch_id, 0, 6, self.ledger, self.schedule)
        # sequence_length=8, so each sample contributes at most 7 "useful"
        # positions under the concatenate_and_chop estimate (final position excluded).
        self.assertEqual(report["estimated_useful_tokens"], report["total_samples"] * 7)

    def test_narrower_range_only_includes_those_steps(self):
        full = audit_range(self.run_id, self.branch_id, 0, 6, self.ledger, self.schedule)
        partial = audit_range(self.run_id, self.branch_id, 2, 4, self.ledger, self.schedule)
        self.assertLess(partial["total_samples"], full["total_samples"])
        self.assertEqual(partial["total_samples"], 2 * 4)

    def test_rejects_invalid_range(self):
        with self.assertRaises(ValueError):
            audit_range(self.run_id, self.branch_id, 3, 3, self.ledger, self.schedule)

    def test_raises_when_nothing_recorded_for_the_range(self):
        with self.assertRaises(ValueError):
            audit_range(self.run_id, self.branch_id, 100, 200, self.ledger, self.schedule)

    def test_raises_for_an_unknown_branch(self):
        with self.assertRaises(ValueError):
            audit_range("run-a", "no-such-branch", 0, 6, self.ledger, self.schedule)


class TestExactUsefulTokensRange(AuditFixture):
    def test_matches_the_estimate_when_no_structure_preserving_masking_applies(self):
        # This fixture's lanes (code, qa) are both concatenate_and_chop --
        # audit_range's estimate is exact in that case, so the two should
        # agree precisely, not just approximately.
        estimate = audit_range(self.run_id, self.branch_id, 0, 6, self.ledger, self.schedule)
        exact = exact_useful_tokens_range(
            0, 6, "seed-1", self.schedule, self.lane_pools, self.store, self.shards_dir, self.microbatch_size,
        )
        self.assertEqual(exact["total_served_tokens"], estimate["total_served_tokens"])
        self.assertEqual(exact["exact_useful_tokens"], estimate["estimated_useful_tokens"])

    def test_narrower_range_only_includes_those_steps(self):
        full = exact_useful_tokens_range(
            0, 6, "seed-1", self.schedule, self.lane_pools, self.store, self.shards_dir, self.microbatch_size,
        )
        partial = exact_useful_tokens_range(
            2, 4, "seed-1", self.schedule, self.lane_pools, self.store, self.shards_dir, self.microbatch_size,
        )
        self.assertLess(partial["total_served_tokens"], full["total_served_tokens"])

    def test_rejects_invalid_range(self):
        with self.assertRaises(ValueError):
            exact_useful_tokens_range(
                3, 3, "seed-1", self.schedule, self.lane_pools, self.store, self.shards_dir, self.microbatch_size,
            )

    def test_breaks_down_by_shard_and_lane(self):
        report = exact_useful_tokens_range(
            0, 6, "seed-1", self.schedule, self.lane_pools, self.store, self.shards_dir, self.microbatch_size,
        )
        self.assertEqual(set(report["useful_tokens_by_lane"]), {"code", "qa"})
        self.assertEqual(set(report["useful_tokens_by_shard"]), {"shard-000000", "shard-000001"})
        self.assertEqual(
            sum(report["useful_tokens_by_lane"].values()), report["exact_useful_tokens"]
        )


class TestExactUsefulTokensRangeWithStructurePreservingMasking(unittest.TestCase):
    """Reproduces test_packer.py's TestStructurePreservingMasking fixture --
    the exact scenario audit_range's estimate_caveat warns about: a
    structure-preserving lane masks more than just the window's final
    position, and only a real recompute (not the ledger-derived estimate)
    counts it correctly."""

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        d = Path(self.tmpdir.name)
        self.shards_dir = d / "shards"
        self.store = ManifestStore(d / "manifests")

        # 3 prompt tokens (loss-masked) + 4 response tokens + eos (final
        # position always loss-masked) -- 8 tokens, exactly one window.
        tokens = [1, 2, 3, 10, 11, 12, 13, EOS]
        make_shard(
            self.shards_dir, self.store, "shard-000000", "instruction", tokens,
            [{"document_id": "docA", "start_token": 0, "end_token": 8, "response_start_token": 3}],
        )
        self.lane_pools = self.store.document_pool_by_lane()
        self.schedule = one_lane_schedule("instruction", sequence_length=8, available_tokens=8, num_steps=3)

    def test_exact_count_reflects_prompt_masking_the_estimate_would_miss(self):
        report = exact_useful_tokens_range(
            0, 3, "seed-1", self.schedule, self.lane_pools, self.store, self.shards_dir, microbatch_size=1,
        )
        # Same document every step (scarcity_policy="repeat"): loss_mask is
        # (0,0,0,1,1,1,1,0) each time -- 4 real useful positions, not the
        # concatenate_and_chop estimate of 7 (sequence_length - 1).
        self.assertEqual(report["exact_useful_tokens"], 3 * 4)
        self.assertEqual(report["total_served_tokens"], 3 * 8)
        self.assertLess(report["exact_useful_tokens"], report["total_served_tokens"] - 3)


class TestMixtureComplianceReport(AuditFixture):
    def test_planned_and_actual_shares_are_reported_per_lane(self):
        report = mixture_compliance_report(self.run_id, self.branch_id, 0, 6, self.ledger, self.schedule)
        self.assertIn("code", report["lanes"])
        self.assertIn("qa", report["lanes"])
        for lane, comparison in report["lanes"].items():
            self.assertAlmostEqual(
                comparison["actual_share"] - comparison["planned_share"], comparison["delta"]
            )
        # 50/50 mixture, ample supply -> actual should land in the
        # neighborhood of 0.5 each. pick_lane converges to its target over
        # *many* draws (see tds/cursor.py), not exactly over any one small
        # sample -- 24 samples here can reasonably land anywhere in a wide
        # band around 0.5, so this checks direction/magnitude, not an exact
        # match.
        self.assertAlmostEqual(report["lanes"]["code"]["actual_share"], 0.5, delta=0.25)
        self.assertAlmostEqual(report["lanes"]["qa"]["actual_share"], 0.5, delta=0.25)

    def test_rejects_invalid_range(self):
        with self.assertRaises(ValueError):
            mixture_compliance_report(self.run_id, self.branch_id, 5, 5, self.ledger, self.schedule)


class TestThroughputReport(unittest.TestCase):
    def test_aggregates_tokens_per_second(self):
        timings = [
            StepTiming(global_step=0, wall_seconds=1.0, total_tokens=100, useful_tokens=90),
            StepTiming(global_step=1, wall_seconds=1.0, total_tokens=100, useful_tokens=90),
        ]
        report = throughput_report(timings)
        self.assertEqual(report["num_steps"], 2)
        self.assertAlmostEqual(report["tokens_per_second"], 100.0)
        self.assertAlmostEqual(report["useful_tokens_per_second"], 90.0)

    def test_rejects_empty_input(self):
        with self.assertRaises(ValueError):
            throughput_report([])


if __name__ == "__main__":
    unittest.main()

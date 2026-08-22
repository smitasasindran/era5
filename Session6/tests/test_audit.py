import sys
import tempfile
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tds.audit import (  # noqa: E402
    LineageStep,
    StepTiming,
    audit_branch_lineage,
    audit_range,
    branch_lineage,
    exact_useful_tokens_range,
    mixture_compliance_report,
    throughput_report,
)
from tds.checkpoint import CheckpointManager  # noqa: E402
from tds.consumption_ledger import ConsumptionLedger, build_ledger_entry  # noqa: E402
from tds.batch_assembler import BatchAssembler  # noqa: E402
from tds.fork import fork_branch  # noqa: E402
from tds.manifest_store import ManifestStore  # noqa: E402
from tds.mixture_compiler import MixtureStage, compile_curriculum  # noqa: E402
from tds.model import ToyTransformer, ToyTransformerConfig  # noqa: E402
from tds.packer import Packer  # noqa: E402
from tds.training_step import run_training_step  # noqa: E402

from test_packer import EOS, make_shard, one_lane_schedule  # noqa: E402


def make_model_and_optimizer(seed=0, lr=1e-2):
    config = ToyTransformerConfig(vocab_size=180, max_sequence_length=8, d_model=8, n_layers=1, n_heads=2, d_ff=16)
    model = ToyTransformer(config, seed=seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    return model, optimizer


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


class TestBranchLineage(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        self.checkpoints = CheckpointManager(Path(self.tmpdir.name) / "checkpoints")
        self.model, self.optimizer = make_model_and_optimizer()

    def test_unforked_branch_is_its_own_single_element_lineage(self):
        self.checkpoints.save(self.model, self.optimizer, "run-a", "main", global_step=10)
        lineage = branch_lineage(self.checkpoints, "run-a", "main")
        self.assertEqual(lineage, [LineageStep(branch_id="main", parent_branch_id=None, fork_step=None)])

    def test_single_fork_lineage_is_root_then_child(self):
        self.checkpoints.save(self.model, self.optimizer, "run-a", "main", global_step=10)
        fork_model, fork_optimizer = make_model_and_optimizer(seed=999)
        fork_branch(self.checkpoints, "run-a", "main", 10, "fork-1", fork_model, fork_optimizer)

        lineage = branch_lineage(self.checkpoints, "run-a", "fork-1")
        self.assertEqual(
            lineage,
            [
                LineageStep(branch_id="main", parent_branch_id=None, fork_step=None),
                LineageStep(branch_id="fork-1", parent_branch_id="main", fork_step=10),
            ],
        )
        # The root branch's own lineage is unaffected by being forked from.
        self.assertEqual(
            branch_lineage(self.checkpoints, "run-a", "main"),
            [LineageStep(branch_id="main", parent_branch_id=None, fork_step=None)],
        )

    def test_chained_forks_produce_a_three_link_lineage(self):
        self.checkpoints.save(self.model, self.optimizer, "run-a", "main", global_step=10)
        fork1_model, fork1_optimizer = make_model_and_optimizer(seed=1)
        fork_branch(self.checkpoints, "run-a", "main", 10, "fork-1", fork1_model, fork1_optimizer)
        # fork-1 keeps training and gets forked again, later, from a later step.
        self.checkpoints.save(fork1_model, fork1_optimizer, "run-a", "fork-1", global_step=20)
        fork2_model, fork2_optimizer = make_model_and_optimizer(seed=2)
        fork_branch(self.checkpoints, "run-a", "fork-1", 20, "fork-2", fork2_model, fork2_optimizer)

        lineage = branch_lineage(self.checkpoints, "run-a", "fork-2")
        self.assertEqual([n.branch_id for n in lineage], ["main", "fork-1", "fork-2"])
        self.assertEqual([n.fork_step for n in lineage], [None, 10, 20])
        self.assertEqual([n.parent_branch_id for n in lineage], [None, "main", "fork-1"])

    def test_raises_for_a_branch_with_no_checkpoint(self):
        with self.assertRaises(ValueError):
            branch_lineage(self.checkpoints, "run-a", "no-such-branch")


class TestAuditBranchLineage(unittest.TestCase):
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
        self.run_id = "run-lineage-test"
        self.fork_step = 2

        self.ledger = ConsumptionLedger(d / "consumption")
        self.checkpoints = CheckpointManager(d / "checkpoints")

        main_model, main_optimizer = make_model_and_optimizer(seed=0)
        main_packer = Packer("seed-main", self.schedule, self.lane_pools, self.store, self.shards_dir)
        main_assembler = BatchAssembler(main_packer, self.microbatch_size)
        for step in range(self.fork_step + 4):  # main keeps going past the fork point too
            mbs = main_assembler.assemble_step(step)
            run_training_step(main_model, main_optimizer, mbs)
            for mb in mbs:
                self.ledger.append(build_ledger_entry(self.run_id, "main", mb, self.schedule, "sha256:tok"))
            if step == self.fork_step:
                self.checkpoints.save(main_model, main_optimizer, self.run_id, "main", self.fork_step)

        fork_model, fork_optimizer = make_model_and_optimizer(seed=777)
        fork_branch(self.checkpoints, self.run_id, "main", self.fork_step, "fork-1", fork_model, fork_optimizer)
        fork_packer = Packer("seed-fork", self.schedule, self.lane_pools, self.store, self.shards_dir)
        fork_assembler = BatchAssembler(fork_packer, self.microbatch_size)
        for step in range(self.fork_step + 1):
            fork_assembler.assemble_step(step)  # replay to the fork point, same as resume/fork always do
        for step in range(self.fork_step + 1, self.fork_step + 3):
            mbs = fork_assembler.assemble_step(step)
            run_training_step(fork_model, fork_optimizer, mbs)
            for mb in mbs:
                self.ledger.append(build_ledger_entry(self.run_id, "fork-1", mb, self.schedule, "sha256:tok"))

    def test_unforked_branch_matches_audit_range_exactly(self):
        end_step = self.fork_step + 4
        lineage_report = audit_branch_lineage(
            self.run_id, "main", end_step, self.ledger, self.checkpoints, self.schedule
        )
        plain_report = audit_range(self.run_id, "main", 0, end_step, self.ledger, self.schedule)
        for key in plain_report:
            self.assertEqual(lineage_report[key], plain_report[key])
        self.assertEqual(
            lineage_report["lineage"],
            [{"branch_id": "main", "parent_branch_id": None, "fork_step": None}],
        )

    def test_forked_branch_pulls_in_parent_history_before_the_fork(self):
        end_step = self.fork_step + 3  # covers fork-1's own 2 post-fork steps
        report = audit_branch_lineage(
            self.run_id, "fork-1", end_step, self.ledger, self.checkpoints, self.schedule
        )
        # (fork_step + 1) steps of shared pre-fork history from "main", plus
        # 2 steps of fork-1's own post-fork history -- each step contributes
        # global_batch_size=4 samples.
        expected_steps = (self.fork_step + 1) + 2
        self.assertEqual(report["total_samples"], expected_steps * 4)
        self.assertEqual(
            report["lineage"],
            [
                {"branch_id": "main", "parent_branch_id": None, "fork_step": None},
                {"branch_id": "fork-1", "parent_branch_id": "main", "fork_step": self.fork_step},
            ],
        )

    def test_forked_branch_history_differs_from_its_own_ledger_entries_alone(self):
        # fork-1's own ledger has no entries at or before fork_step (see
        # tests/test_fork.py) -- reading only its own branch would miss the
        # shared pre-fork history entirely and either raise or undercount.
        end_step = self.fork_step + 3
        lineage_report = audit_branch_lineage(
            self.run_id, "fork-1", end_step, self.ledger, self.checkpoints, self.schedule
        )
        own_branch_only = audit_range(
            self.run_id, "fork-1", self.fork_step + 1, end_step, self.ledger, self.schedule
        )
        self.assertGreater(lineage_report["total_samples"], own_branch_only["total_samples"])

    def test_rejects_non_positive_end_step(self):
        with self.assertRaises(ValueError):
            audit_branch_lineage(self.run_id, "main", 0, self.ledger, self.checkpoints, self.schedule)

    def test_raises_when_nothing_recorded_in_range(self):
        # A resolvable lineage (checkpoint exists) whose ledger simply has
        # no entries yet -- distinct from an unresolvable one (no checkpoint
        # at all, covered by TestBranchLineage.test_raises_for_a_branch_with_no_checkpoint).
        model, optimizer = make_model_and_optimizer()
        self.checkpoints.save(model, optimizer, "run-empty", "main", global_step=0)
        with self.assertRaises(ValueError):
            audit_branch_lineage("run-empty", "main", 5, self.ledger, self.checkpoints, self.schedule)


if __name__ == "__main__":
    unittest.main()

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tds.batch_assembler import BatchAssembler, Microbatch  # noqa: E402
from tds.consumption_ledger import ConsumptionLedger, build_ledger_entry  # noqa: E402
from tds.learning_ledger import (  # noqa: E402
    LearningLedger,
    LearningLedgerError,
    accumulate_per_shard_loss,
    build_learning_ledger_entries,
    classify_usefulness,
    model_phase_at_step,
)
from tds.manifest_store import ManifestStore  # noqa: E402
from tds.mixture_compiler import MixtureStage, compile_curriculum  # noqa: E402
from tds.model import ToyTransformer, ToyTransformerConfig  # noqa: E402
from tds.packer import PackedSample, Packer  # noqa: E402
from tds.training_step import run_training_step  # noqa: E402

from test_packer import make_shard, one_lane_schedule  # noqa: E402


def make_sample(global_step, slot, lane, boundaries, sequence_length):
    return PackedSample(
        global_step=global_step,
        slot=slot,
        lane=lane,
        packing_policy="concatenate_and_chop",
        sequence_packing_policy="greedy",
        token_ids=tuple(range(sequence_length)),
        segment_id=tuple([0] * sequence_length),
        position_id=tuple(range(sequence_length)),
        loss_mask=tuple([1] * (sequence_length - 1) + [0]),
        segment_boundaries=tuple(boundaries),
        loss_mask_hash="sha256:fake",
    )


class TestAccumulatePerShardLoss(unittest.TestCase):
    def test_single_sample_single_shard(self):
        sample = make_sample(0, 0, "code", [(0, "shard-a", "d0", 0, 4, 0, 4)], sequence_length=4)
        mb = Microbatch(
            global_step=0,
            microbatch_index=0,
            rank=0,
            token_ids=np.zeros((1, 4), dtype=np.int64),
            segment_id=np.zeros((1, 4), dtype=np.int64),
            position_id=np.arange(4).reshape(1, 4),
            loss_mask=np.array([[1, 1, 1, 0]], dtype=np.float32),
            samples=(sample,),
        )
        per_token_loss = np.array([[1.0, 2.0, 3.0]])  # shape (1, 3) == sequence_length - 1
        averages = accumulate_per_shard_loss([mb], [per_token_loss])
        self.assertAlmostEqual(averages["shard-a"], 2.0)  # (1+2+3)/3

    def test_sample_spanning_two_shards_attributes_correctly(self):
        # positions [0,2) from shard-a, [2,4) from shard-b within one 4-token window.
        sample = make_sample(
            0, 0, "code",
            [(0, "shard-a", "d0", 0, 2, 10, 12), (1, "shard-b", "d1", 2, 4, 0, 2)],
            sequence_length=4,
        )
        mb = Microbatch(
            global_step=0, microbatch_index=0, rank=0,
            token_ids=np.zeros((1, 4), dtype=np.int64),
            segment_id=np.array([[0, 0, 1, 1]]),
            position_id=np.array([[0, 1, 0, 1]]),
            loss_mask=np.array([[1, 1, 1, 0]], dtype=np.float32),
            samples=(sample,),
        )
        per_token_loss = np.array([[1.0, 2.0, 3.0]])  # 3 valid prediction positions
        averages = accumulate_per_shard_loss([mb], [per_token_loss])
        # positions 0,1 -> shard-a (both counted, loss_mask[:2]=[1,1]); position 2 -> shard-b
        self.assertAlmostEqual(averages["shard-a"], 1.5)  # (1+2)/2
        self.assertAlmostEqual(averages["shard-b"], 3.0)  # position 3 has mask 0, excluded

    def test_masked_positions_excluded(self):
        sample = make_sample(0, 0, "code", [(0, "shard-a", "d0", 0, 4, 0, 4)], sequence_length=4)
        mb = Microbatch(
            global_step=0, microbatch_index=0, rank=0,
            token_ids=np.zeros((1, 4), dtype=np.int64),
            segment_id=np.zeros((1, 4), dtype=np.int64),
            position_id=np.arange(4).reshape(1, 4),
            loss_mask=np.array([[1, 0, 1, 0]], dtype=np.float32),
            samples=(sample,),
        )
        per_token_loss = np.array([[10.0, 20.0, 30.0]])
        averages = accumulate_per_shard_loss([mb], [per_token_loss])
        self.assertAlmostEqual(averages["shard-a"], 20.0)  # (10+30)/2, position 1 excluded


class TestModelPhaseAtStep(unittest.TestCase):
    def setUp(self):
        stages = [
            MixtureStage(stage="foundation", token_start=0, token_end=900, sequence_length=1, mixture={"code": 1.0}),
            MixtureStage(stage="anneal", token_start=900, token_end=1200, sequence_length=1, mixture={"code": 1.0}),
        ]
        self.schedule = compile_curriculum(stages, {"code": 10_000}, global_batch_size=1)

    def test_early_mid_late_by_progress_fraction(self):
        # foundation stage steps [0, 900); anneal steps [900, 1200) -- total_steps=1200
        self.assertEqual(model_phase_at_step(self.schedule, 0), "early")
        self.assertEqual(model_phase_at_step(self.schedule, 500), "mid")
        self.assertEqual(model_phase_at_step(self.schedule, 850), "late")

    def test_anneal_stage_always_reports_anneal(self):
        self.assertEqual(model_phase_at_step(self.schedule, 900), "anneal")
        self.assertEqual(model_phase_at_step(self.schedule, 1199), "anneal")


class TestClassifyUsefulness(unittest.TestCase):
    def test_boundaries(self):
        self.assertEqual(classify_usefulness(-0.5), "useful")
        self.assertEqual(classify_usefulness(0.5), "harmful")
        self.assertEqual(classify_usefulness(0.0), "neutral")
        self.assertEqual(classify_usefulness(-1e-4), "neutral")  # within threshold


class TestBuildLearningLedgerEntriesEndToEnd(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        d = Path(self.tmpdir.name)
        self.shards_dir = d / "shards"
        self.store = ManifestStore(d / "manifests")

        make_shard(
            self.shards_dir, self.store, "shard-000000", "code", list(range(40)),
            [{"document_id": "d0", "start_token": 0, "end_token": 40, "response_start_token": None}],
        )
        self.pool = {"code": [("shard-000000", "d0", 0)]}
        self.schedule = one_lane_schedule("code", sequence_length=8, available_tokens=40, num_steps=5, batch_size=2)
        self.config = ToyTransformerConfig(
            vocab_size=48, max_sequence_length=8, d_model=8, n_layers=1, n_heads=2, d_ff=16
        )

    def run_step(self, model, optimizer, global_step, consumption_ledger):
        packer = Packer("seed-1", self.schedule, self.pool, self.store, self.shards_dir)
        assembler = BatchAssembler(packer, microbatch_size=2)
        # Advance the packer/assembler up to global_step (fresh instance each call
        # mirrors replay -- deterministic regardless of how many times this runs).
        for step in range(global_step + 1):
            microbatches = assembler.assemble_step(step)
        result = run_training_step(model, optimizer, microbatches)
        for mb in microbatches:
            consumption_ledger.append(
                build_ledger_entry("run-a", "main", mb, self.schedule, tokenizer_hash="sha256:tok")
            )
        return result

    def test_entries_reference_the_touched_shard_and_have_correct_delta(self):
        consumption_ledger = ConsumptionLedger(Path(self.tmpdir.name) / "consumption")
        model = ToyTransformer(self.config, seed=0)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.3)

        result = self.run_step(model, optimizer, 0, consumption_ledger)
        entries = build_learning_ledger_entries("run-a", "main", result, self.schedule, consumption_ledger)

        self.assertEqual(len(entries), 1)  # only shard-000000 exists
        entry = entries[0]
        self.assertEqual(entry["shard_id"], "shard-000000")
        self.assertEqual(entry["global_step"], 0)
        self.assertIsNone(entry["opus_score"])
        # Only one shard exists, so its delta must equal the whole step's
        # overall before/after delta exactly.
        self.assertAlmostEqual(
            entry["loss_delta_before_after"], result.avg_loss_after - result.avg_loss_before, places=5
        )
        self.assertEqual(entry["repeated_pass_number"], 1)

    def test_repeated_pass_number_increments_across_steps(self):
        consumption_ledger = ConsumptionLedger(Path(self.tmpdir.name) / "consumption")
        model = ToyTransformer(self.config, seed=0)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        result0 = self.run_step(model, optimizer, 0, consumption_ledger)
        entries0 = build_learning_ledger_entries("run-a", "main", result0, self.schedule, consumption_ledger)
        result1 = self.run_step(model, optimizer, 1, consumption_ledger)
        entries1 = build_learning_ledger_entries("run-a", "main", result1, self.schedule, consumption_ledger)

        self.assertEqual(entries0[0]["repeated_pass_number"], 1)
        self.assertEqual(entries1[0]["repeated_pass_number"], 2)


class TestLearningLedger(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        self.dir = Path(self.tmpdir.name)

    def make_entry(self, run_id="run-a", branch_id="main", global_step=0, shard_id="shard-000000", **overrides):
        entry = {
            "run_id": run_id,
            "branch_id": branch_id,
            "global_step": global_step,
            "shard_id": shard_id,
            "avg_token_loss": 2.0,
            "loss_delta_before_after": -0.1,
            "opus_score": None,
            "repeated_pass_number": 1,
            "model_phase": "early",
            "usefulness_classification": "useful",
        }
        entry.update(overrides)
        return entry

    def test_append_then_get(self):
        ledger = LearningLedger(self.dir)
        ledger.append(self.make_entry())
        self.assertEqual(ledger.get("run-a", "main", 0, "shard-000000")["avg_token_loss"], 2.0)

    def test_identical_reappend_is_idempotent(self):
        ledger = LearningLedger(self.dir)
        ledger.append(self.make_entry())
        ledger.append(self.make_entry())
        lines = (self.dir / "index.jsonl").read_text().strip().splitlines()
        self.assertEqual(len(lines), 1)

    def test_mismatched_reappend_is_rejected(self):
        ledger = LearningLedger(self.dir)
        ledger.append(self.make_entry(usefulness_classification="useful"))
        with self.assertRaises(LearningLedgerError):
            ledger.append(self.make_entry(usefulness_classification="harmful"))

    def test_reloading_from_disk_recovers_prior_entries(self):
        ledger = LearningLedger(self.dir)
        ledger.append(self.make_entry())
        reloaded = LearningLedger(self.dir)
        self.assertEqual(len(reloaded.all()), 1)

    def test_for_branch_filters(self):
        ledger = LearningLedger(self.dir)
        ledger.append(self.make_entry(branch_id="main"))
        ledger.append(self.make_entry(branch_id="fork-1"))
        self.assertEqual(len(ledger.for_branch("run-a", "main")), 1)
        self.assertEqual(len(ledger.for_branch("run-a", "fork-1")), 1)


if __name__ == "__main__":
    unittest.main()

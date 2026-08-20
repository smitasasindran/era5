import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tds.batch_assembler import BatchAssembler  # noqa: E402
from tds.consumption_ledger import (  # noqa: E402
    DATALOADER_VERSION,
    ConsumptionLedger,
    ConsumptionLedgerError,
    build_ledger_entry,
)
from tds.hashing import sha256_bytes  # noqa: E402
from tds.manifest_store import ManifestStore  # noqa: E402
from tds.packer import Packer  # noqa: E402

from test_packer import make_shard  # noqa: E402


class TestBuildLedgerEntry(unittest.TestCase):
    """Uses a real Packer/BatchAssembler over a fixture with two lanes and
    a document that spans two shards' worth of segments in one sample, so
    the per-sample list fields (mixture_lane, shard_ids, token_span_ids)
    actually exercise more than the trivial single-lane/single-shard case."""

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        d = Path(self.tmpdir.name)
        self.shards_dir = d / "shards"
        self.store = ManifestStore(d / "manifests")

        # code: one shard with two short documents -- one sample will pack both.
        make_shard(
            self.shards_dir,
            self.store,
            "shard-000000",
            "code",
            list(range(10)),
            [
                {"document_id": "c0", "start_token": 0, "end_token": 5, "response_start_token": None},
                {"document_id": "c1", "start_token": 5, "end_token": 10, "response_start_token": None},
            ],
        )
        # qa: a second shard, different lane.
        make_shard(
            self.shards_dir,
            self.store,
            "shard-000001",
            "qa",
            list(range(100, 108)),
            [{"document_id": "q0", "start_token": 0, "end_token": 8, "response_start_token": None}],
        )

        lane_pools = {
            "code": [("shard-000000", "c0", 0), ("shard-000000", "c1", 5)],
            "qa": [("shard-000001", "q0", 0)],
        }
        # Two-lane, batch-of-2 schedule so a single microbatch spans both lanes.
        from tds.mixture_compiler import MixtureStage, compile_curriculum

        stages = [
            MixtureStage(
                stage="mixed",
                token_start=0,
                token_end=20,
                sequence_length=10,
                mixture={"code": 0.5, "qa": 0.5},
            )
        ]
        self.schedule = compile_curriculum(
            stages, {"code": 10, "qa": 8}, global_batch_size=2, scarcity_policy="repeat"
        )
        self.packer = Packer("seed-1", self.schedule, lane_pools, self.store, self.shards_dir)
        self.assembler = BatchAssembler(self.packer, microbatch_size=2)

    def test_entry_shape_and_ids(self):
        mb = self.assembler.assemble_step(0)[0]
        entry = build_ledger_entry("run-a", "main", mb, self.schedule, tokenizer_hash="sha256:tok")

        self.assertEqual(entry["run_id"], "run-a")
        self.assertEqual(entry["branch_id"], "main")
        self.assertEqual(entry["global_step"], 0)
        self.assertEqual(entry["microbatch_index"], 0)
        self.assertEqual(entry["microbatch_id"], "mb-0-0")
        self.assertEqual(entry["rank"], 0)
        self.assertEqual(entry["curriculum_stage"], "mixed")
        self.assertEqual(entry["tokenizer_version"], "sha256:tok")
        self.assertEqual(entry["dataloader_version"], DATALOADER_VERSION)
        self.assertEqual(entry["attention_policy"], "causal+segment")
        self.assertEqual(entry["position_policy"], "reset_per_segment")

    def test_per_sample_fields_are_parallel_lists_matching_row_order(self):
        mb = self.assembler.assemble_step(0)[0]
        entry = build_ledger_entry("run-a", "main", mb, self.schedule, tokenizer_hash="sha256:tok")

        self.assertEqual(len(entry["packed_sample_ids"]), 2)
        self.assertEqual(len(entry["mixture_lane"]), 2)
        self.assertEqual(len(entry["shard_ids"]), 2)
        self.assertEqual(len(entry["token_span_ids"]), 2)
        self.assertEqual(len(entry["opus_decision_id"]), 2)

        for i, sample in enumerate(mb.samples):
            self.assertEqual(entry["packed_sample_ids"][i], f"ps-{sample.global_step}-{sample.slot}")
            self.assertEqual(entry["mixture_lane"][i], sample.lane)
            self.assertIsNone(entry["opus_decision_id"][i])

    def test_shard_ids_deduplicated_per_sample(self):
        # The code lane's window packs both c0 and c1, both from shard-000000
        # -- shard_ids for that sample must list it once, not twice.
        mb = self.assembler.assemble_step(0)[0]
        entry = build_ledger_entry("run-a", "main", mb, self.schedule, tokenizer_hash="sha256:tok")
        for sample_shard_ids in entry["shard_ids"]:
            self.assertEqual(len(sample_shard_ids), len(set(sample_shard_ids)))

    def test_token_span_ids_format(self):
        mb = self.assembler.assemble_step(0)[0]
        entry = build_ledger_entry("run-a", "main", mb, self.schedule, tokenizer_hash="sha256:tok")
        for i, sample in enumerate(mb.samples):
            expected = [f"{shard_id}:{s_start}-{s_end}" for _, shard_id, _, _, _, s_start, s_end in sample.segment_boundaries]
            self.assertEqual(entry["token_span_ids"][i], expected)

    def test_loss_mask_hash_covers_the_whole_stacked_array(self):
        mb = self.assembler.assemble_step(0)[0]
        entry = build_ledger_entry("run-a", "main", mb, self.schedule, tokenizer_hash="sha256:tok")
        self.assertEqual(entry["loss_mask_hash"], sha256_bytes(mb.loss_mask.tobytes()))

    def test_two_independent_packers_produce_identical_entries(self):
        # The property resume/replay ultimately rests on: recomputing from
        # scratch reproduces the exact same ledger entry.
        packer_b = Packer(
            "seed-1", self.schedule, self.packer.lane_pools, self.store, self.shards_dir
        )
        assembler_b = BatchAssembler(packer_b, microbatch_size=2)
        mb_a = self.assembler.assemble_step(0)[0]
        mb_b = assembler_b.assemble_step(0)[0]
        entry_a = build_ledger_entry("run-a", "main", mb_a, self.schedule, tokenizer_hash="sha256:tok")
        entry_b = build_ledger_entry("run-a", "main", mb_b, self.schedule, tokenizer_hash="sha256:tok")
        self.assertEqual(entry_a, entry_b)


class TestConsumptionLedger(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        self.dir = Path(self.tmpdir.name)

    def make_entry(self, run_id="run-a", branch_id="main", global_step=0, microbatch_index=0, **overrides):
        entry = {
            "run_id": run_id,
            "branch_id": branch_id,
            "global_step": global_step,
            "microbatch_index": microbatch_index,
            "microbatch_id": f"mb-{global_step}-{microbatch_index}",
            "rank": 0,
            "curriculum_stage": "foundation",
            "packed_sample_ids": ["ps-0-0", "ps-0-1"],
            "mixture_lane": ["code", "qa"],
            "shard_ids": [["shard-000000"], ["shard-000001"]],
            "token_span_ids": [["shard-000000:0-10"], ["shard-000001:0-10"]],
            "loss_mask_hash": "sha256:fake",
            "attention_policy": "causal+segment",
            "position_policy": "reset_per_segment",
            "tokenizer_version": "sha256:tok",
            "dataloader_version": "tds-v1",
            "opus_decision_id": [None, None],
        }
        entry.update(overrides)
        return entry

    def test_append_then_get(self):
        ledger = ConsumptionLedger(self.dir)
        ledger.append(self.make_entry())
        self.assertEqual(ledger.get("run-a", "main", "mb-0-0")["curriculum_stage"], "foundation")

    def test_identical_reappend_is_idempotent(self):
        ledger = ConsumptionLedger(self.dir)
        ledger.append(self.make_entry())
        ledger.append(self.make_entry())
        lines = (self.dir / "index.jsonl").read_text().strip().splitlines()
        self.assertEqual(len(lines), 1)

    def test_mismatched_reappend_is_rejected(self):
        ledger = ConsumptionLedger(self.dir)
        ledger.append(self.make_entry(curriculum_stage="foundation"))
        with self.assertRaises(ConsumptionLedgerError):
            ledger.append(self.make_entry(curriculum_stage="capability-expansion"))

    def test_reloading_from_disk_recovers_prior_entries(self):
        ledger = ConsumptionLedger(self.dir)
        ledger.append(self.make_entry())
        reloaded = ConsumptionLedger(self.dir)
        self.assertEqual(len(reloaded.all()), 1)
        self.assertIsNotNone(reloaded.get("run-a", "main", "mb-0-0"))

    def test_for_branch_filters_by_run_and_branch(self):
        ledger = ConsumptionLedger(self.dir)
        ledger.append(self.make_entry(run_id="run-a", branch_id="main", global_step=0, microbatch_index=0))
        ledger.append(self.make_entry(run_id="run-a", branch_id="fork-1", global_step=0, microbatch_index=0))
        ledger.append(self.make_entry(run_id="run-b", branch_id="main", global_step=0, microbatch_index=0))
        self.assertEqual(len(ledger.for_branch("run-a", "main")), 1)
        self.assertEqual(len(ledger.for_branch("run-a", "fork-1")), 1)
        self.assertEqual(len(ledger.for_branch("run-b", "main")), 1)

    def test_last_recorded_microbatch(self):
        ledger = ConsumptionLedger(self.dir)
        self.assertIsNone(ledger.last_recorded_microbatch("run-a", "main"))

        ledger.append(self.make_entry(global_step=0, microbatch_index=0))
        ledger.append(self.make_entry(global_step=0, microbatch_index=1))
        ledger.append(self.make_entry(global_step=1, microbatch_index=0))
        self.assertEqual(ledger.last_recorded_microbatch("run-a", "main"), (1, 0))

    def test_last_recorded_microbatch_ignores_other_branches(self):
        ledger = ConsumptionLedger(self.dir)
        ledger.append(self.make_entry(run_id="run-a", branch_id="main", global_step=5, microbatch_index=0))
        ledger.append(self.make_entry(run_id="run-a", branch_id="fork-1", global_step=99, microbatch_index=0))
        self.assertEqual(ledger.last_recorded_microbatch("run-a", "main"), (5, 0))


if __name__ == "__main__":
    unittest.main()

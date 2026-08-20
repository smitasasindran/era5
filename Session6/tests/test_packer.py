import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tds.hashing import sha256_bytes  # noqa: E402
from tds.manifest_store import ManifestStore  # noqa: E402
from tds.mixture_compiler import MixtureStage, compile_curriculum  # noqa: E402
from tds.packer import Packer, attention_bias_from_segments  # noqa: E402

EOS = 1


def make_shard(shards_dir, store, shard_id, lane, tokens, spans):
    arr = np.array(tokens, dtype="uint16")
    shards_dir.mkdir(parents=True, exist_ok=True)
    np.save(shards_dir / f"{shard_id}.npy", arr)
    store.append(
        {
            "shard_id": shard_id,
            "content_hash": sha256_bytes(arr.tobytes()),
            "tokenizer_hash": "sha256:test",
            "capability_lane": lane,
            "packing_policy": "greedy",
            "token_count": len(tokens),
            "document_count": len(spans),
            "document_spans": spans,
            "license_tier": "unspecified",
            "dedup_status": "not_checked",
            "contamination_status": "not_checked",
            "eval_overlap_status": "none",
            "cleaning_pipeline_hash": None,
            "parent_shard_ids": [],
        }
    )


def one_lane_schedule(lane, sequence_length, available_tokens, num_steps, batch_size=1):
    stages = [
        MixtureStage(
            stage="a",
            token_start=0,
            token_end=sequence_length * batch_size * num_steps,
            sequence_length=sequence_length,
            mixture={lane: 1.0},
        )
    ]
    return compile_curriculum(
        stages, {lane: available_tokens}, global_batch_size=batch_size, scarcity_policy="repeat"
    )


class TestConcatenateAndChopWorkedExample(unittest.TestCase):
    """Reproduces DATALOADER_DESIGN.md §5.6's worked example exactly: doc A
    (4 real tokens + eos) followed by doc B (5 real tokens + eos) packed
    into a 10-token window, with doc B's own trailing eos overflowing into
    the next window."""

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        d = Path(self.tmpdir.name)
        self.shards_dir = d / "shards"
        self.store = ManifestStore(d / "manifests")

        doc_a = [10, 11, 12, 13, EOS]  # 5 tokens: docA's 4 + its own eos
        doc_b = [20, 21, 22, 23, 24, EOS]  # 6 tokens: docB's 5 + its own eos
        make_shard(
            self.shards_dir,
            self.store,
            "shard-000000",
            "code",
            doc_a + doc_b,
            [
                {"document_id": "docA", "start_token": 0, "end_token": 5, "response_start_token": None},
                {"document_id": "docB", "start_token": 5, "end_token": 11, "response_start_token": None},
            ],
        )
        self.pool = {"code": [("shard-000000", "docA", 0), ("shard-000000", "docB", 5)]}
        self.schedule = one_lane_schedule("code", sequence_length=10, available_tokens=11, num_steps=2)
        # This seed happens to shuffle epoch 0 of this exact pool as [docA, docB].
        self.packer = Packer("test-seed-1", self.schedule, self.pool, self.store, self.shards_dir)

    def test_first_window_matches_the_design_docs_worked_example(self):
        sample = self.packer.pack_step(0)[0]
        self.assertEqual(sample.token_ids, (10, 11, 12, 13, EOS, 20, 21, 22, 23, 24))
        self.assertEqual(sample.segment_id, (0, 0, 0, 0, 0, 1, 1, 1, 1, 1))
        self.assertEqual(sample.position_id, (0, 1, 2, 3, 4, 0, 1, 2, 3, 4))
        # Every position loss-visible except the window's true final
        # position -- notably position 4 (the eos ending doc A) stays 1:
        # "a new document starts here" is itself useful pretraining signal.
        self.assertEqual(sample.loss_mask, (1, 1, 1, 1, 1, 1, 1, 1, 1, 0))
        self.assertEqual(
            sample.segment_boundaries,
            (
                (0, "shard-000000", "docA", 0, 5, 0, 5),
                (1, "shard-000000", "docB", 5, 10, 5, 10),
            ),
        )

    def test_second_window_carries_over_doc_bs_leftover_eos(self):
        self.packer.pack_step(0)  # advances the shared lane stream/carry state
        sample = self.packer.pack_step(1)[0]

        # doc B had 1 token (its own eos) left over after window 0.
        first_boundary = sample.segment_boundaries[0]
        self.assertEqual(first_boundary[:3], (0, "shard-000000", "docB"))
        self.assertEqual(first_boundary[3:], (0, 1, 10, 11))  # window[0:1) <- shard[10:11)
        self.assertEqual(sample.token_ids[0], EOS)
        self.assertEqual(len(sample.token_ids), 10)
        self.assertEqual(sample.loss_mask[-1], 0)


class TestCarryOverAcrossManyWindows(unittest.TestCase):
    """A single document longer than one window must be split across
    consecutive windows with no gap, overlap, or duplication -- and once
    exhausted, the (size-1) pool cycles back to the same document."""

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        d = Path(self.tmpdir.name)
        self.shards_dir = d / "shards"
        self.store = ManifestStore(d / "manifests")

        self.doc_tokens = list(range(100, 125))  # 25 tokens, one document
        make_shard(
            self.shards_dir,
            self.store,
            "shard-000000",
            "code",
            self.doc_tokens,
            [{"document_id": "docA", "start_token": 0, "end_token": 25, "response_start_token": None}],
        )
        self.pool = {"code": [("shard-000000", "docA", 0)]}
        self.schedule = one_lane_schedule("code", sequence_length=10, available_tokens=25, num_steps=3)
        self.packer = Packer("test-seed-1", self.schedule, self.pool, self.store, self.shards_dir)

    def test_reconstructs_the_document_exactly_then_repeats(self):
        w0 = self.packer.pack_step(0)[0].token_ids
        w1 = self.packer.pack_step(1)[0].token_ids
        w2 = self.packer.pack_step(2)[0].token_ids

        reconstructed = list(w0) + list(w1) + list(w2[:5])
        self.assertEqual(reconstructed, self.doc_tokens)
        # window 2's last 5 positions: the pool (size 1) cycled to a fresh
        # epoch and re-read the same document from its own start.
        self.assertEqual(list(w2[5:]), self.doc_tokens[:5])

    def test_no_window_ever_gains_or_loses_tokens(self):
        for step in range(3):
            sample = self.packer.pack_step(step)[0]
            self.assertEqual(len(sample.token_ids), 10)
            self.assertEqual(sum(end - start for *_, start, end in sample.segment_boundaries), 10)


class TestStructurePreservingMasking(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        d = Path(self.tmpdir.name)
        self.shards_dir = d / "shards"
        self.store = ManifestStore(d / "manifests")

        # 3 prompt tokens, 4 response tokens, 1 eos -- 8 total, exactly one window.
        tokens = [1, 2, 3, 10, 11, 12, 13, EOS]
        make_shard(
            self.shards_dir,
            self.store,
            "shard-000000",
            "instruction",
            tokens,
            [{"document_id": "docA", "start_token": 0, "end_token": 8, "response_start_token": 3}],
        )
        pool = {"instruction": [("shard-000000", "docA", 0)]}
        schedule = one_lane_schedule("instruction", sequence_length=8, available_tokens=8, num_steps=1)
        self.packer = Packer("test-seed-1", schedule, pool, self.store, self.shards_dir)

    def test_prompt_tokens_are_loss_masked_response_tokens_are_not(self):
        sample = self.packer.pack_step(0)[0]
        self.assertEqual(sample.packing_policy, "structure_preserving")
        # prompt (0,1,2) masked, response (3,4,5,6) visible, final position (7) masked.
        self.assertEqual(sample.loss_mask, (0, 0, 0, 1, 1, 1, 1, 0))


class TestStructurePreservingDegradesGracefullyWithoutAMarker(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        d = Path(self.tmpdir.name)
        self.shards_dir = d / "shards"
        self.store = ManifestStore(d / "manifests")

        tokens = [1, 2, 3, 4, EOS]
        make_shard(
            self.shards_dir,
            self.store,
            "shard-000000",
            "instruction",
            tokens,
            # response_start_token is None -- no marker was found for this document.
            [{"document_id": "docA", "start_token": 0, "end_token": 5, "response_start_token": None}],
        )
        pool = {"instruction": [("shard-000000", "docA", 0)]}
        schedule = one_lane_schedule("instruction", sequence_length=5, available_tokens=5, num_steps=1)
        self.packer = Packer("test-seed-1", schedule, pool, self.store, self.shards_dir)

    def test_whole_document_stays_loss_visible_except_the_final_position(self):
        sample = self.packer.pack_step(0)[0]
        self.assertEqual(sample.loss_mask, (1, 1, 1, 1, 0))


class TestAttentionBiasFromSegments(unittest.TestCase):
    def test_causal_and_same_segment(self):
        segment_id = [0, 0, 0, 1, 1]
        mask = attention_bias_from_segments(segment_id)
        expected = np.array(
            [
                [True, False, False, False, False],
                [True, True, False, False, False],
                [True, True, True, False, False],
                [False, False, False, True, False],
                [False, False, False, True, True],
            ]
        )
        np.testing.assert_array_equal(mask, expected)


class TestBestFitSequencePacking(unittest.TestCase):
    """best_fit picks whichever remaining document in the current epoch
    fits a window's leftover room most tightly, instead of taking whatever
    is next in stream order -- reducing how often documents get split."""

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        d = Path(self.tmpdir.name)
        self.shards_dir = d / "shards"
        self.store = ManifestStore(d / "manifests")

    def _one_doc_shard(self, doc_id, start, length):
        return {
            "document_id": doc_id,
            "start_token": start,
            "end_token": start + length,
            "response_start_token": None,
        }

    def test_rejects_unknown_policy(self):
        schedule = one_lane_schedule("code", sequence_length=10, available_tokens=10, num_steps=1)
        with self.assertRaises(ValueError):
            Packer(
                "seed",
                schedule,
                {"code": []},
                self.store,
                self.shards_dir,
                sequence_packing_policy="worst_fit",
            )

    def test_picks_the_tightest_fitting_document_over_stream_order(self):
        # Window budget 10, documents of length 10, 6, 4. best_fit picks
        # whichever single document fits the room most tightly at each
        # decision point: window 0 gets the exact 10-length match (zero
        # leftover, no split at all); window 1 then gets 6 followed by an
        # exact-fitting 4 -- both windows filled with no document ever
        # split, regardless of the documents' original stream order.
        tokens = list(range(20))  # 10 + 6 + 4 = 20 tokens total
        spans = [
            self._one_doc_shard("exact", 0, 10),
            self._one_doc_shard("medium", 10, 6),
            self._one_doc_shard("small", 16, 4),
        ]
        make_shard(self.shards_dir, self.store, "shard-000000", "code", tokens, spans)
        pool = {
            "code": [
                ("shard-000000", "medium", 10),  # deliberately not in best-fit order
                ("shard-000000", "small", 16),
                ("shard-000000", "exact", 0),
            ]
        }
        schedule = one_lane_schedule("code", sequence_length=10, available_tokens=20, num_steps=2)
        packer = Packer(
            "seed-order", schedule, pool, self.store, self.shards_dir, sequence_packing_policy="best_fit"
        )

        window0 = packer.pack_step(0)[0]
        self.assertEqual(len(window0.segment_boundaries), 1)
        self.assertEqual(window0.segment_boundaries[0][2], "exact")

        window1 = packer.pack_step(1)[0]
        doc_ids_in_window1 = [b[2] for b in window1.segment_boundaries]
        self.assertEqual(sorted(doc_ids_in_window1), ["medium", "small"])
        self.assertEqual(len(window1.segment_boundaries), 2)  # neither doc had to be split

    def test_falls_back_to_smallest_overflowing_document_when_nothing_fits(self):
        # Every remaining document is bigger than the room (5): best_fit
        # should pick the smallest of them (6) rather than the biggest (12).
        tokens = list(range(18))  # 6 + 12 = 18
        spans = [self._one_doc_shard("small", 0, 6), self._one_doc_shard("big", 6, 12)]
        make_shard(self.shards_dir, self.store, "shard-000000", "code", tokens, spans)
        pool = {"code": [("shard-000000", "small", 0), ("shard-000000", "big", 6)]}
        schedule = one_lane_schedule("code", sequence_length=5, available_tokens=18, num_steps=1)
        packer = Packer(
            "seed-1", schedule, pool, self.store, self.shards_dir, sequence_packing_policy="best_fit"
        )
        window0 = packer.pack_step(0)[0]
        self.assertEqual(window0.segment_boundaries[0][2], "small")

    def test_no_document_is_lost_or_duplicated_across_many_windows(self):
        lengths = [9, 6, 4, 11, 3, 7]
        offsets = [sum(lengths[:i]) for i in range(len(lengths))]
        tokens = list(range(sum(lengths)))
        spans = [
            self._one_doc_shard(f"d{i}", offsets[i], lengths[i]) for i in range(len(lengths))
        ]
        make_shard(self.shards_dir, self.store, "shard-000000", "code", tokens, spans)
        pool = {"code": [("shard-000000", f"d{i}", offsets[i]) for i in range(len(lengths))]}
        schedule = one_lane_schedule("code", sequence_length=10, available_tokens=sum(lengths), num_steps=6)
        packer = Packer(
            "seed-x", schedule, pool, self.store, self.shards_dir, sequence_packing_policy="best_fit"
        )

        all_tokens = []
        for step in range(6):
            all_tokens.extend(packer.pack_step(step)[0].token_ids)
        # 6 windows of 10 = 60 tokens; total document length is 40, so the
        # pool must have cycled into a second epoch partway through.
        self.assertEqual(len(all_tokens), 60)
        # best_fit reorders documents (by fit, not stream order), so the
        # first 40 tokens reconstruct every document's tokens as a *set*,
        # not necessarily in their original concatenation order.
        self.assertEqual(sorted(all_tokens[:40]), sorted(tokens))

    def test_records_the_policy_on_the_sample(self):
        tokens = list(range(10))
        make_shard(
            self.shards_dir, self.store, "shard-000000", "code", tokens,
            [self._one_doc_shard("d0", 0, 10)],
        )
        pool = {"code": [("shard-000000", "d0", 0)]}
        schedule = one_lane_schedule("code", sequence_length=10, available_tokens=10, num_steps=1)
        packer = Packer(
            "seed", schedule, pool, self.store, self.shards_dir, sequence_packing_policy="best_fit"
        )
        sample = packer.pack_step(0)[0]
        self.assertEqual(sample.sequence_packing_policy, "best_fit")
        self.assertEqual(sample.packing_policy, "concatenate_and_chop")  # loss-masking axis unaffected

    def test_default_policy_is_greedy_and_is_recorded(self):
        tokens = list(range(10))
        make_shard(
            self.shards_dir, self.store, "shard-000000", "code", tokens,
            [self._one_doc_shard("d0", 0, 10)],
        )
        pool = {"code": [("shard-000000", "d0", 0)]}
        schedule = one_lane_schedule("code", sequence_length=10, available_tokens=10, num_steps=1)
        packer = Packer("seed", schedule, pool, self.store, self.shards_dir)
        sample = packer.pack_step(0)[0]
        self.assertEqual(sample.sequence_packing_policy, "greedy")

    def test_two_fresh_best_fit_packers_produce_an_identical_stream(self):
        lengths = [9, 6, 4, 11, 3, 7]
        offsets = [sum(lengths[:i]) for i in range(len(lengths))]
        tokens = list(range(sum(lengths)))
        spans = [
            self._one_doc_shard(f"d{i}", offsets[i], lengths[i]) for i in range(len(lengths))
        ]
        make_shard(self.shards_dir, self.store, "shard-000000", "code", tokens, spans)
        pool = {"code": [("shard-000000", f"d{i}", offsets[i]) for i in range(len(lengths))]}
        schedule = one_lane_schedule("code", sequence_length=10, available_tokens=sum(lengths), num_steps=6)

        packer_a = Packer(
            "seed-det", schedule, pool, self.store, self.shards_dir, sequence_packing_policy="best_fit"
        )
        packer_b = Packer(
            "seed-det", schedule, pool, self.store, self.shards_dir, sequence_packing_policy="best_fit"
        )
        samples_a = [packer_a.pack_step(step)[0] for step in range(6)]
        samples_b = [packer_b.pack_step(step)[0] for step in range(6)]
        self.assertEqual(samples_a, samples_b)


class TestDeterminism(unittest.TestCase):
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
            list(range(50)),
            [
                {"document_id": "d0", "start_token": 0, "end_token": 12, "response_start_token": None},
                {"document_id": "d1", "start_token": 12, "end_token": 25, "response_start_token": None},
                {"document_id": "d2", "start_token": 25, "end_token": 37, "response_start_token": None},
                {"document_id": "d3", "start_token": 37, "end_token": 50, "response_start_token": None},
            ],
        )
        self.pool = {
            "code": [
                ("shard-000000", "d0", 0),
                ("shard-000000", "d1", 12),
                ("shard-000000", "d2", 25),
                ("shard-000000", "d3", 37),
            ]
        }
        self.schedule = one_lane_schedule("code", sequence_length=7, available_tokens=50, num_steps=10)

    def test_two_fresh_packers_produce_an_identical_stream(self):
        packer_a = Packer("seed-x", self.schedule, self.pool, self.store, self.shards_dir)
        packer_b = Packer("seed-x", self.schedule, self.pool, self.store, self.shards_dir)
        samples_a = [packer_a.pack_step(step)[0] for step in range(10)]
        samples_b = [packer_b.pack_step(step)[0] for step in range(10)]
        self.assertEqual(samples_a, samples_b)


if __name__ == "__main__":
    unittest.main()

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tds.batch_assembler import BatchAssembler  # noqa: E402
from tds.manifest_store import ManifestStore  # noqa: E402
from tds.model import ToyTransformer, ToyTransformerConfig  # noqa: E402
from tds.opus import (  # noqa: E402
    apply_opus_selection,
    filtered_pools_from_decisions,
    freeze_opus_selection,
    load_frozen_opus_selection,
    score_candidate,
)
from tds.packer import Packer  # noqa: E402

from test_packer import make_shard  # noqa: E402


class TestScoreCandidate(unittest.TestCase):
    def setUp(self):
        self.config = ToyTransformerConfig(
            vocab_size=32, max_sequence_length=16, d_model=8, n_layers=1, n_heads=2, d_ff=16
        )
        self.model = ToyTransformer(self.config, seed=0)

    def test_deterministic_for_the_same_tokens(self):
        tokens = np.arange(10)
        a = score_candidate(self.model, tokens, max_sequence_length=16)
        b = score_candidate(self.model, tokens, max_sequence_length=16)
        self.assertEqual(a, b)

    def test_too_short_document_scores_zero(self):
        self.assertEqual(score_candidate(self.model, np.array([1]), max_sequence_length=16), 0.0)
        self.assertEqual(score_candidate(self.model, np.array([]), max_sequence_length=16), 0.0)

    def test_returns_a_finite_float(self):
        score = score_candidate(self.model, np.arange(20), max_sequence_length=16)
        self.assertIsInstance(score, float)
        self.assertTrue(np.isfinite(score))


class OpusFixture(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        d = Path(self.tmpdir.name)
        self.shards_dir = d / "shards"
        self.store = ManifestStore(d / "manifests")

        make_shard(
            self.shards_dir, self.store, "shard-000000", "code", list(range(20)),
            [
                {"document_id": "c0", "start_token": 0, "end_token": 10, "response_start_token": None},
                {"document_id": "c1", "start_token": 10, "end_token": 20, "response_start_token": None},
            ],
        )
        make_shard(
            self.shards_dir, self.store, "shard-000001", "indic", list(range(100, 110)),
            [{"document_id": "i0", "start_token": 0, "end_token": 10, "response_start_token": None}],
        )
        self.lane_pools = self.store.document_pool_by_lane()


class TestApplyOpusSelectionDisabled(OpusFixture):
    def test_disabled_accepts_everything_without_a_model(self):
        filtered, decisions = apply_opus_selection(
            self.lane_pools, None, self.store, self.shards_dir, 16, 1.0, 8.0, enabled=False
        )
        self.assertEqual(filtered, self.lane_pools)
        self.assertTrue(all(d.status == "accepted" for d in decisions))
        self.assertTrue(all(d.opus_score is None for d in decisions))
        self.assertTrue(all(not d.protected_floor_override for d in decisions))

    def test_enabled_without_a_model_raises(self):
        with self.assertRaises(ValueError):
            apply_opus_selection(self.lane_pools, None, self.store, self.shards_dir, 16, 1.0, 8.0, enabled=True)


class TestApplyOpusSelectionThresholds(OpusFixture):
    """Uses a mocked score_candidate so the threshold/protection branching
    logic is tested deterministically, independent of what an actual
    model's forward pass happens to produce."""

    def setUp(self):
        super().setUp()
        self.fake_model = object()  # never actually called -- score_candidate is mocked

    @patch("tds.opus.score_candidate")
    def test_low_score_is_rejected_and_excluded_from_filtered_pool(self, mock_score):
        mock_score.return_value = 0.5  # below reject_below=1.0
        filtered, decisions = apply_opus_selection(
            self.lane_pools, self.fake_model, self.store, self.shards_dir, 16, 1.0, 8.0
        )
        self.assertTrue(all(d.status == "rejected" for d in decisions))
        self.assertTrue(all(d.rejection_reason == "low_proxy_utility" for d in decisions))
        self.assertTrue(all(d.effective_token_estimate == 0 for d in decisions))
        self.assertTrue(all(pool == [] for pool in filtered.values()))

    @patch("tds.opus.score_candidate")
    def test_high_score_is_deferred_and_excluded_from_filtered_pool(self, mock_score):
        mock_score.return_value = 20.0  # above defer_above=8.0
        filtered, decisions = apply_opus_selection(
            self.lane_pools, self.fake_model, self.store, self.shards_dir, 16, 1.0, 8.0
        )
        self.assertTrue(all(d.status == "deferred" for d in decisions))
        self.assertTrue(all(d.rejection_reason == "anomalous_high_loss" for d in decisions))
        self.assertTrue(all(pool == [] for pool in filtered.values()))

    @patch("tds.opus.score_candidate")
    def test_middle_score_is_accepted_with_real_token_estimate(self, mock_score):
        mock_score.return_value = 4.0
        filtered, decisions = apply_opus_selection(
            self.lane_pools, self.fake_model, self.store, self.shards_dir, 16, 1.0, 8.0
        )
        self.assertTrue(all(d.status == "accepted" for d in decisions))
        self.assertTrue(all(d.rejection_reason is None for d in decisions))
        code_decision = next(d for d in decisions if d.document_id == "c0")
        self.assertEqual(code_decision.effective_token_estimate, 10)
        self.assertEqual(filtered, self.lane_pools)

    @patch("tds.opus.score_candidate")
    def test_protected_lane_is_rescued_from_rejection(self, mock_score):
        mock_score.return_value = 0.5  # would be rejected everywhere
        filtered, decisions = apply_opus_selection(
            self.lane_pools, self.fake_model, self.store, self.shards_dir, 16, 1.0, 8.0,
            protected_lanes=frozenset({"indic"}),
        )
        indic_decisions = [d for d in decisions if d.capability_lane == "indic"]
        self.assertTrue(all(d.status == "accepted" for d in indic_decisions))
        self.assertTrue(all(d.protected_floor_override for d in indic_decisions))
        self.assertEqual(filtered["indic"], self.lane_pools["indic"])

        code_decisions = [d for d in decisions if d.capability_lane == "code"]
        self.assertTrue(all(d.status == "rejected" for d in code_decisions))
        self.assertEqual(filtered["code"], [])

    @patch("tds.opus.score_candidate")
    def test_candidate_id_format(self, mock_score):
        mock_score.return_value = 4.0
        _, decisions = apply_opus_selection(
            self.lane_pools, self.fake_model, self.store, self.shards_dir, 16, 1.0, 8.0
        )
        ids = {d.candidate_id for d in decisions}
        self.assertIn("opus-shard-000000-c0", ids)
        self.assertIn("opus-shard-000001-i0", ids)


class TestFilteredPoolsFromDecisions(OpusFixture):
    @patch("tds.opus.score_candidate")
    def test_reconstructs_the_same_filtered_pools(self, mock_score):
        mock_score.side_effect = lambda model, tokens, max_len: (
            0.5 if tokens[0] == 0 else 4.0
        )  # c0 (starts at token 0) rejected, everything else accepted
        filtered_live, decisions = apply_opus_selection(
            self.lane_pools, object(), self.store, self.shards_dir, 16, 1.0, 8.0
        )
        filtered_reconstructed = filtered_pools_from_decisions(self.lane_pools, decisions)
        self.assertEqual(filtered_live, filtered_reconstructed)
        self.assertNotIn(("shard-000000", "c0", 0), filtered_reconstructed["code"])


class TestFreezeAndLoad(OpusFixture):
    @patch("tds.opus.score_candidate")
    def test_roundtrip(self, mock_score):
        mock_score.return_value = 4.0
        _, decisions = apply_opus_selection(
            self.lane_pools, object(), self.store, self.shards_dir, 16, 1.0, 8.0
        )
        path = Path(self.tmpdir.name) / "opus_decisions.json"
        manifest = freeze_opus_selection(decisions, path)
        loaded, loaded_manifest = load_frozen_opus_selection(path)

        self.assertEqual(manifest["decisions_hash"], loaded_manifest["decisions_hash"])
        self.assertEqual(loaded, decisions)
        self.assertEqual(manifest["accepted_count"], len(decisions))

    def test_load_fails_without_a_frozen_selection(self):
        with self.assertRaises(FileNotFoundError):
            load_frozen_opus_selection(Path(self.tmpdir.name) / "missing.json")

    @patch("tds.opus.score_candidate")
    def test_load_detects_tampering(self, mock_score):
        mock_score.return_value = 4.0
        _, decisions = apply_opus_selection(
            self.lane_pools, object(), self.store, self.shards_dir, 16, 1.0, 8.0
        )
        path = Path(self.tmpdir.name) / "opus_decisions.json"
        freeze_opus_selection(decisions, path)
        with open(path, "a") as f:
            f.write(" ")
        with self.assertRaises(ValueError):
            load_frozen_opus_selection(path)


class TestOpusIntegratesWithPacker(OpusFixture):
    """A rejected document must never be drawable by the Packer at all --
    the whole point of filtering the pool before Packer/Cursor exist."""

    @patch("tds.opus.score_candidate")
    def test_rejected_document_never_appears_in_packed_output(self, mock_score):
        mock_score.side_effect = lambda model, tokens, max_len: 0.5 if tokens[0] == 0 else 4.0
        filtered, _ = apply_opus_selection(
            self.lane_pools, object(), self.store, self.shards_dir, 16, 1.0, 8.0
        )
        from tds.mixture_compiler import MixtureStage, compile_curriculum

        stages = [MixtureStage(stage="a", token_start=0, token_end=100, sequence_length=8, mixture={"code": 1.0})]
        schedule = compile_curriculum(stages, {"code": 10}, global_batch_size=1, scarcity_policy="repeat")
        packer = Packer("seed-1", schedule, filtered, self.store, self.shards_dir)
        assembler = BatchAssembler(packer, microbatch_size=1)

        seen_documents = set()
        for step in range(10):
            for mb in assembler.assemble_step(step):
                for sample in mb.samples:
                    for _seg, _shard, document_id, *_ in sample.segment_boundaries:
                        seen_documents.add(document_id)
        self.assertNotIn("c0", seen_documents)
        self.assertIn("c1", seen_documents)  # the only other code document -- must still be reachable


if __name__ == "__main__":
    unittest.main()

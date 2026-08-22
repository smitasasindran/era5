import itertools
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tds.cursor import (  # noqa: E402
    iter_lane_assignments,
    lane_document_stream,
    lane_epoch_stream,
    lane_for_slot,
    lane_sequence,
    pick_lane,
)
from tds.mixture_compiler import MixtureStage, compile_curriculum  # noqa: E402


def stage(name, start, end, mixture, floors=None, seq_len=8, warmup=0):
    return MixtureStage(
        stage=name,
        token_start=start,
        token_end=end,
        sequence_length=seq_len,
        mixture=mixture,
        protected_floors=floors or {},
        warmup_tokens=warmup,
    )


def pool(lane, n):
    return [(f"shard-{lane}-{i:03d}", f"doc-{lane}-{i:03d}", 0) for i in range(n)]


class TestPickLane(unittest.TestCase):
    def test_deterministic_for_same_inputs(self):
        weights = {"code": 0.5, "qa": 0.5}
        a = pick_lane("seed-1", weights, global_step=3, slot=1)
        b = pick_lane("seed-1", weights, global_step=3, slot=1)
        self.assertEqual(a, b)

    def test_never_picks_a_zero_weight_lane(self):
        weights = {"code": 1.0, "qa": 0.0}
        for step in range(200):
            self.assertEqual(pick_lane("seed-1", weights, step, 0), "code")

    def test_raises_when_no_lane_has_positive_weight(self):
        with self.assertRaises(ValueError):
            pick_lane("seed-1", {"code": 0.0, "qa": 0.0}, 0, 0)

    def test_renormalizes_when_weights_sum_below_one(self):
        # Only "code" has weight; a stage with unallocated_share must still
        # fill every slot, so "code" is picked every time despite its
        # weight not summing to 1.0 on its own.
        weights = {"code": 0.3}
        for step in range(50):
            self.assertEqual(pick_lane("seed-1", weights, step, 0), "code")

    def test_different_seeds_can_diverge(self):
        weights = {"code": 0.5, "qa": 0.5}
        picks_a = [pick_lane("seed-a", weights, step, 0) for step in range(50)]
        picks_b = [pick_lane("seed-b", weights, step, 0) for step in range(50)]
        self.assertNotEqual(picks_a, picks_b)


class TestLaneForSlotAndIterAssignments(unittest.TestCase):
    def setUp(self):
        stages = [stage("a", 0, 1600, {"code": 0.5, "qa": 0.5}, seq_len=8)]  # batch=2 -> 100 steps
        self.schedule = compile_curriculum(stages, {"code": 100_000, "qa": 100_000}, global_batch_size=2)

    def test_lane_for_slot_matches_bulk_iteration(self):
        bulk = list(itertools.islice(iter_lane_assignments("seed-1", self.schedule), 25))
        step, slot, lane = bulk[24]
        self.assertEqual(lane_for_slot("seed-1", self.schedule, step, slot), lane)

    def test_lane_for_slot_rejects_out_of_range_slot(self):
        with self.assertRaises(ValueError):
            lane_for_slot("seed-1", self.schedule, global_step=0, slot=2)

    def test_iter_lane_assignments_stops_at_schedule_end(self):
        items = list(iter_lane_assignments("seed-1", self.schedule))
        self.assertEqual(len(items), self.schedule.total_steps * self.schedule.global_batch_size)

    def test_two_fresh_runs_are_identical(self):
        run_a = list(itertools.islice(iter_lane_assignments("seed-1", self.schedule), 300))
        run_b = list(itertools.islice(iter_lane_assignments("seed-1", self.schedule), 300))
        self.assertEqual(run_a, run_b)


class TestLaneSequence(unittest.TestCase):
    def test_deterministic_for_same_inputs(self):
        p = pool("code", 10)
        a = lane_sequence("seed-1", "code", epoch=0, pool=p)
        b = lane_sequence("seed-1", "code", epoch=0, pool=p)
        self.assertEqual(a, b)

    def test_is_a_full_permutation_of_the_pool(self):
        p = pool("code", 10)
        shuffled = lane_sequence("seed-1", "code", epoch=0, pool=p)
        self.assertEqual(sorted(shuffled), sorted(p))

    def test_different_epochs_reshuffle(self):
        p = pool("code", 20)
        epoch0 = lane_sequence("seed-1", "code", epoch=0, pool=p)
        epoch1 = lane_sequence("seed-1", "code", epoch=1, pool=p)
        self.assertNotEqual(epoch0, epoch1)

    def test_raises_on_empty_pool(self):
        with self.assertRaises(ValueError):
            lane_sequence("seed-1", "code", epoch=0, pool=[])


class TestLaneDocumentStream(unittest.TestCase):
    def test_first_pool_size_items_cover_the_pool_exactly_once(self):
        p = pool("code", 7)
        items = list(itertools.islice(lane_document_stream("seed-1", "code", p), 7))
        self.assertEqual(sorted(items), sorted(p))

    def test_cycles_and_reshuffles_after_exhaustion(self):
        p = pool("code", 7)
        items = list(itertools.islice(lane_document_stream("seed-1", "code", p), 14))
        first_epoch, second_epoch = items[:7], items[7:14]
        self.assertEqual(sorted(first_epoch), sorted(second_epoch))  # same documents...
        self.assertNotEqual(first_epoch, second_epoch)  # ...but reshuffled

    def test_is_infinite_and_deterministic_across_fresh_generators(self):
        p = pool("code", 5)
        run_a = list(itertools.islice(lane_document_stream("seed-1", "code", p), 37))
        run_b = list(itertools.islice(lane_document_stream("seed-1", "code", p), 37))
        self.assertEqual(run_a, run_b)

    def test_raises_on_empty_pool_when_advanced(self):
        with self.assertRaises(ValueError):
            next(lane_document_stream("seed-1", "code", []))

    def test_different_seeds_diverge(self):
        p = pool("code", 20)
        run_a = list(itertools.islice(lane_document_stream("seed-1", "code", p), 20))
        run_b = list(itertools.islice(lane_document_stream("seed-2", "code", p), 20))
        self.assertNotEqual(run_a, run_b)


class TestLaneEpochStream(unittest.TestCase):
    def test_first_item_is_a_full_permutation_of_the_pool(self):
        p = pool("code", 7)
        first_epoch = next(lane_epoch_stream("seed-1", "code", p))
        self.assertEqual(sorted(first_epoch), sorted(p))

    def test_matches_lane_sequence_per_epoch(self):
        p = pool("code", 7)
        stream = lane_epoch_stream("seed-1", "code", p)
        for epoch in range(3):
            self.assertEqual(next(stream), lane_sequence("seed-1", "code", epoch, p))

    def test_successive_epochs_reshuffle(self):
        p = pool("code", 20)
        stream = lane_epoch_stream("seed-1", "code", p)
        epoch0, epoch1 = next(stream), next(stream)
        self.assertNotEqual(epoch0, epoch1)
        self.assertEqual(sorted(epoch0), sorted(epoch1))

    def test_deterministic_across_fresh_generators(self):
        p = pool("code", 10)
        run_a = list(itertools.islice(lane_epoch_stream("seed-1", "code", p), 4))
        run_b = list(itertools.islice(lane_epoch_stream("seed-1", "code", p), 4))
        self.assertEqual(run_a, run_b)


class TestScarcityAwareLaneAssignment(unittest.TestCase):
    def test_defer_policy_lane_is_never_assigned(self):
        stages = [stage("a", 0, 800, {"code": 0.5, "qa": 0.5}, seq_len=8)]
        schedule = compile_curriculum(
            stages, {"code": 10, "qa": 100_000}, global_batch_size=1, scarcity_policy="defer"
        )
        assignments = list(iter_lane_assignments("seed-1", schedule))
        self.assertTrue(all(lane == "qa" for _, _, lane in assignments))


if __name__ == "__main__":
    unittest.main()

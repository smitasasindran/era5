import itertools
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tds.cursor import (  # noqa: E402
    CandidateItem,
    cursor,
    iter_candidates,
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


class TestIterCandidatesSingleLane(unittest.TestCase):
    def setUp(self):
        stages = [stage("a", 0, 800, {"code": 1.0}, seq_len=8)]  # 8 tok/step, batch=1 -> 100 steps
        self.schedule = compile_curriculum(stages, {"code": 100_000}, global_batch_size=1)
        self.pool = pool("code", 7)
        self.lane_pools = {"code": self.pool}

    def test_first_pool_size_candidates_cover_the_pool_exactly_once(self):
        items = list(itertools.islice(iter_candidates("seed-1", self.schedule, self.lane_pools), 7))
        seen = {(item.shard_id, item.document_id) for item in items}
        expected = {(shard_id, doc_id) for shard_id, doc_id, _ in self.pool}
        self.assertEqual(seen, expected)
        self.assertTrue(all(item.epoch == 0 for item in items))
        self.assertEqual([item.pick_index for item in items], list(range(7)))

    def test_pool_reshuffles_and_repeats_after_exhaustion(self):
        items = list(itertools.islice(iter_candidates("seed-1", self.schedule, self.lane_pools), 14))
        first_epoch = [(i.shard_id, i.document_id) for i in items[:7]]
        second_epoch = [(i.shard_id, i.document_id) for i in items[7:14]]
        self.assertEqual(sorted(first_epoch), sorted(second_epoch))  # same documents...
        self.assertNotEqual(first_epoch, second_epoch)  # ...but reshuffled
        self.assertTrue(all(item.epoch == 1 for item in items[7:14]))

    def test_stops_at_schedule_end(self):
        items = list(iter_candidates("seed-1", self.schedule, self.lane_pools))
        self.assertEqual(len(items), self.schedule.total_steps * self.schedule.global_batch_size)


class TestCursorPointQuery(unittest.TestCase):
    def setUp(self):
        stages = [stage("a", 0, 1600, {"code": 0.5, "qa": 0.5}, seq_len=8)]  # batch=2 -> 100 steps
        self.schedule = compile_curriculum(stages, {"code": 100_000, "qa": 100_000}, global_batch_size=2)
        self.lane_pools = {"code": pool("code", 15), "qa": pool("qa", 15)}

    def test_matches_the_corresponding_bulk_iteration_item(self):
        bulk = list(itertools.islice(iter_candidates("seed-1", self.schedule, self.lane_pools), 25))
        point = cursor("seed-1", self.schedule, self.lane_pools, global_step=12, slot=0)
        self.assertEqual(point, bulk[24])

    def test_rejects_out_of_range_slot(self):
        with self.assertRaises(ValueError):
            cursor("seed-1", self.schedule, self.lane_pools, global_step=0, slot=2)


class TestReplayDeterminism(unittest.TestCase):
    """The property the crash/resume/replay requirement ultimately rests
    on: two independent runs of iter_candidates from scratch, given the
    same (seed, schedule, lane_pools), must produce an identical stream --
    with no persisted state shared between them."""

    def setUp(self):
        stages = [
            stage("a", 0, 800, {"code": 0.6, "qa": 0.4}, seq_len=8),
            stage("b", 800, 1600, {"code": 0.2, "qa": 0.8}, seq_len=8, warmup=80),
        ]
        self.schedule = compile_curriculum(stages, {"code": 100_000, "qa": 100_000}, global_batch_size=2)
        self.lane_pools = {"code": pool("code", 11), "qa": pool("qa", 13)}

    def test_two_fresh_runs_are_identical(self):
        run_a = list(itertools.islice(iter_candidates("seed-1", self.schedule, self.lane_pools), 300))
        run_b = list(itertools.islice(iter_candidates("seed-1", self.schedule, self.lane_pools), 300))
        self.assertEqual(run_a, run_b)

    def test_replaying_a_historical_interval_matches_the_original(self):
        full_run = list(itertools.islice(iter_candidates("seed-1", self.schedule, self.lane_pools), 300))
        # "Replay steps 50..60": recompute from scratch and slice out the
        # same interval, exactly like a fresh process reconstructing a
        # historical window rather than trusting any carried-over state.
        replay = list(
            itertools.islice(iter_candidates("seed-1", self.schedule, self.lane_pools), 100, 120)
        )
        self.assertEqual(replay, full_run[100:120])

    def test_different_seed_diverges_somewhere(self):
        run_a = list(itertools.islice(iter_candidates("seed-1", self.schedule, self.lane_pools), 300))
        run_b = list(itertools.islice(iter_candidates("seed-2", self.schedule, self.lane_pools), 300))
        self.assertNotEqual(run_a, run_b)


class TestScarcityAwareSampling(unittest.TestCase):
    def test_defer_policy_lane_is_never_drawn(self):
        stages = [stage("a", 0, 800, {"code": 0.5, "qa": 0.5}, seq_len=8)]
        schedule = compile_curriculum(
            stages, {"code": 10, "qa": 100_000}, global_batch_size=1, scarcity_policy="defer"
        )
        lane_pools = {"code": pool("code", 5), "qa": pool("qa", 200)}
        items = list(iter_candidates("seed-1", schedule, lane_pools))
        self.assertTrue(all(item.lane == "qa" for item in items))


if __name__ == "__main__":
    unittest.main()

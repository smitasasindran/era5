import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tds.mixture_compiler import (  # noqa: E402
    MixtureStage,
    ScarcityError,
    compile_curriculum,
)


def stage(name, start, end, mixture, floors=None, seq_len=8, warmup=0, scarcity_policy=None):
    return MixtureStage(
        stage=name,
        token_start=start,
        token_end=end,
        sequence_length=seq_len,
        mixture=mixture,
        protected_floors=floors or {},
        warmup_tokens=warmup,
        scarcity_policy=scarcity_policy,
    )


class TestStageValidation(unittest.TestCase):
    def test_no_stages_is_rejected(self):
        with self.assertRaises(ValueError):
            compile_curriculum([], {}, global_batch_size=1)

    def test_mixture_not_summing_to_one_is_rejected(self):
        stages = [stage("a", 0, 100, {"code": 0.5, "qa": 0.4})]
        with self.assertRaises(ValueError):
            compile_curriculum(stages, {"code": 1000, "qa": 1000}, global_batch_size=1)

    def test_floor_exceeding_weight_is_rejected(self):
        stages = [stage("a", 0, 100, {"code": 1.0}, floors={"code": 1.5})]
        with self.assertRaises(ValueError):
            compile_curriculum(stages, {"code": 1000}, global_batch_size=1)

    def test_floor_for_lane_not_in_mixture_is_rejected(self):
        stages = [stage("a", 0, 100, {"code": 1.0}, floors={"qa": 0.1})]
        with self.assertRaises(ValueError):
            compile_curriculum(stages, {"code": 1000, "qa": 1000}, global_batch_size=1)

    def test_floors_summing_over_one_is_rejected(self):
        stages = [stage("a", 0, 100, {"code": 0.5, "qa": 0.5}, floors={"code": 0.6, "qa": 0.6})]
        with self.assertRaises(ValueError):
            compile_curriculum(stages, {"code": 1000, "qa": 1000}, global_batch_size=1)

    def test_non_contiguous_stages_are_rejected(self):
        stages = [
            stage("a", 0, 100, {"code": 1.0}),
            stage("b", 150, 250, {"code": 1.0}),  # gap between 100 and 150
        ]
        with self.assertRaises(ValueError):
            compile_curriculum(stages, {"code": 1000}, global_batch_size=1)

    def test_overlapping_stages_are_rejected(self):
        stages = [
            stage("a", 0, 100, {"code": 1.0}),
            stage("b", 50, 150, {"code": 1.0}),  # overlaps with "a"
        ]
        with self.assertRaises(ValueError):
            compile_curriculum(stages, {"code": 1000}, global_batch_size=1)

    def test_out_of_order_stages_are_rejected(self):
        stages = [
            stage("b", 100, 200, {"code": 1.0}),
            stage("a", 0, 100, {"code": 1.0}),
        ]
        with self.assertRaises(ValueError):
            compile_curriculum(stages, {"code": 1000}, global_batch_size=1)

    def test_zero_or_negative_span_is_rejected(self):
        stages = [stage("a", 100, 100, {"code": 1.0})]
        with self.assertRaises(ValueError):
            compile_curriculum(stages, {"code": 1000}, global_batch_size=1)


class TestAmpleSupply(unittest.TestCase):
    def test_all_lanes_ok_when_supply_is_ample(self):
        stages = [stage("a", 0, 1000, {"code": 0.5, "qa": 0.5})]
        schedule = compile_curriculum(stages, {"code": 10_000, "qa": 10_000}, global_batch_size=1)
        cs = schedule.stages[0]
        for lane in ("code", "qa"):
            plan = cs.lane_plans[lane]
            self.assertEqual(plan.status, "ok")
            self.assertEqual(plan.effective_weight, plan.target_weight)
        self.assertAlmostEqual(cs.unallocated_share, 0.0)

    def test_step_range_uses_floor_division_of_span_by_tokens_per_step(self):
        # sequence_length=8, global_batch_size=1 -> 8 tokens/step; span=1000 -> 125 steps
        stages = [stage("a", 0, 1000, {"code": 1.0}, seq_len=8)]
        schedule = compile_curriculum(stages, {"code": 10_000}, global_batch_size=1)
        cs = schedule.stages[0]
        self.assertEqual(cs.step_start, 0)
        self.assertEqual(cs.step_end, 125)


class TestScarcityRepeatPolicy(unittest.TestCase):
    def test_repeat_fully_serves_target_and_reports_repeat_factor(self):
        stages = [stage("a", 0, 1000, {"code": 1.0})]  # needs 1000 tokens
        schedule = compile_curriculum(stages, {"code": 400}, global_batch_size=1, scarcity_policy="repeat")
        plan = schedule.stages[0].lane_plans["code"]
        self.assertEqual(plan.status, "scarce_repeat")
        self.assertEqual(plan.effective_tokens, 1000)
        self.assertAlmostEqual(plan.repeat_factor, 1000 / 400)

    def test_repeat_with_zero_available_raises(self):
        stages = [stage("a", 0, 1000, {"code": 1.0})]
        with self.assertRaises(ScarcityError):
            compile_curriculum(stages, {"code": 0}, global_batch_size=1, scarcity_policy="repeat")


class TestScarcityReduceSharePolicy(unittest.TestCase):
    def test_reduces_to_available_when_no_floor(self):
        stages = [stage("a", 0, 1000, {"code": 1.0})]  # needs 1000, only 400 available
        schedule = compile_curriculum(
            stages, {"code": 400}, global_batch_size=1, scarcity_policy="reduce_share"
        )
        plan = schedule.stages[0].lane_plans["code"]
        self.assertEqual(plan.status, "scarce_reduced")
        self.assertEqual(plan.effective_tokens, 400)
        self.assertAlmostEqual(plan.effective_weight, 0.4)
        self.assertAlmostEqual(schedule.stages[0].unallocated_share, 0.6)

    def test_floor_is_honored_even_beyond_remaining_capacity(self):
        # code needs 1000, only 100 tokens remain unclaimed, but floor guarantees 300
        stages = [stage("a", 0, 1000, {"code": 1.0}, floors={"code": 0.3})]
        schedule = compile_curriculum(
            stages, {"code": 100}, global_batch_size=1, scarcity_policy="reduce_share"
        )
        plan = schedule.stages[0].lane_plans["code"]
        self.assertEqual(plan.effective_tokens, 300)  # the floor wins, via implied repeat
        self.assertGreater(plan.repeat_factor, 1.0)  # 300 tokens planned from only 100 distinct

    def test_floor_unmeetable_even_by_itself_raises(self):
        stages = [stage("a", 0, 1000, {"code": 1.0}, floors={"code": 0.5})]
        with self.assertRaises(ScarcityError):
            compile_curriculum(stages, {"code": 0}, global_batch_size=1, scarcity_policy="reduce_share")


class TestScarcityDeferPolicy(unittest.TestCase):
    def test_defer_gives_zero_effective_tokens(self):
        stages = [stage("a", 0, 1000, {"code": 0.5, "qa": 0.5})]
        schedule = compile_curriculum(
            stages, {"code": 10, "qa": 10_000}, global_batch_size=1, scarcity_policy="defer"
        )
        plan = schedule.stages[0].lane_plans["code"]
        self.assertEqual(plan.status, "deferred")
        self.assertEqual(plan.effective_tokens, 0.0)
        self.assertGreater(schedule.stages[0].unallocated_share, 0.0)


class TestCumulativeScarcityAcrossStages(unittest.TestCase):
    def test_second_stage_accounts_for_first_stages_consumption(self):
        # 1000 tokens of "code" available total. Stage 1 alone asks for 700
        # (fine in isolation); stage 2 alone asks for another 500 (also
        # "fine in isolation" if checked independently) -- but cumulatively
        # 700 + 500 = 1200 > 1000, so stage 2 must come up scarce.
        stages = [
            stage("a", 0, 700, {"code": 1.0}),
            stage("b", 700, 1200, {"code": 1.0}),
        ]
        schedule = compile_curriculum(
            stages, {"code": 1000}, global_batch_size=1, scarcity_policy="reduce_share"
        )
        first, second = schedule.stages
        self.assertEqual(first.lane_plans["code"].status, "ok")
        self.assertEqual(first.lane_plans["code"].effective_tokens, 700)
        self.assertEqual(second.lane_plans["code"].status, "scarce_reduced")
        self.assertEqual(second.lane_plans["code"].effective_tokens, 300)  # only 300 left of 1000
        self.assertAlmostEqual(second.lane_plans["code"].cumulative_tokens_after, 1000)

    def test_stage_looking_fine_alone_can_still_be_scarce_cumulatively_under_repeat(self):
        stages = [
            stage("a", 0, 700, {"code": 1.0}),
            stage("b", 700, 1200, {"code": 1.0}),
        ]
        schedule = compile_curriculum(stages, {"code": 1000}, global_batch_size=1, scarcity_policy="repeat")
        second = schedule.stages[1]
        self.assertEqual(second.lane_plans["code"].status, "scarce_repeat")
        self.assertAlmostEqual(second.lane_plans["code"].repeat_factor, 1200 / 1000)


class TestStepNumberingAcrossSequenceLengthChanges(unittest.TestCase):
    def test_steps_stay_monotonic_when_sequence_length_changes_between_stages(self):
        # stage a: seq_len=8, batch=1 -> 8 tok/step, span=800 -> 100 steps
        # stage b: seq_len=32, batch=1 -> 32 tok/step, span=800 -> 25 steps
        stages = [
            stage("a", 0, 800, {"code": 1.0}, seq_len=8),
            stage("b", 800, 1600, {"code": 1.0}, seq_len=32),
        ]
        schedule = compile_curriculum(stages, {"code": 100_000}, global_batch_size=1)
        first, second = schedule.stages
        self.assertEqual(first.step_start, 0)
        self.assertEqual(first.step_end, 100)
        self.assertEqual(second.step_start, 100)  # continues exactly where stage 1 left off
        self.assertEqual(second.step_end, 125)
        self.assertLess(first.step_end, second.step_end)


class TestMixtureAtStepAndWarmup(unittest.TestCase):
    def setUp(self):
        stages = [
            stage("a", 0, 800, {"code": 1.0}, seq_len=8),
            stage("b", 800, 1600, {"code": 0.0, "qa": 1.0}, seq_len=8, warmup=80),
        ]
        self.schedule = compile_curriculum(stages, {"code": 100_000, "qa": 100_000}, global_batch_size=1)

    def test_stage_at_step_out_of_range_raises(self):
        with self.assertRaises(ValueError):
            self.schedule.stage_at_step(-1)
        with self.assertRaises(ValueError):
            self.schedule.stage_at_step(10_000)

    def test_first_stage_never_ramps(self):
        mixture = self.schedule.mixture_at_step(0)
        self.assertAlmostEqual(mixture["code"], 1.0)

    def test_mixture_at_stage_start_equals_previous_stage(self):
        # warmup=80 tokens, 8 tok/step -> 10 warmup steps; stage b starts at step 100
        mixture = self.schedule.mixture_at_step(100)
        self.assertAlmostEqual(mixture["code"], 1.0)
        self.assertAlmostEqual(mixture.get("qa", 0.0), 0.0)

    def test_mixture_after_warmup_window_equals_current_stage(self):
        mixture = self.schedule.mixture_at_step(110)  # 100 + 10 warmup steps
        self.assertAlmostEqual(mixture["qa"], 1.0)
        self.assertAlmostEqual(mixture.get("code", 0.0), 0.0)

    def test_mixture_midway_through_warmup_is_a_linear_blend(self):
        mixture = self.schedule.mixture_at_step(105)  # halfway through the 10-step warmup
        self.assertAlmostEqual(mixture["code"], 0.5, places=6)
        self.assertAlmostEqual(mixture["qa"], 0.5, places=6)


if __name__ == "__main__":
    unittest.main()

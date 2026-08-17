"""Curriculum & Mixture Compiler: human-authored stages -> an executable schedule.

Converts a list of curriculum stages (token ranges, lane weights, protected
floors, warmup windows) into a `CompiledSchedule` that can answer, for any
global_step, "which stage is this, and what mixture of lanes should this
step draw from" -- accounting for whether each stage's planned lane shares
are actually satisfiable from the shard supply that exists (per
`ManifestStore.lane_token_totals()`), not just assuming they are.

Scarcity is tracked *cumulatively* across stages, not per-stage in
isolation: a lane's supply is a shared resource across the whole
curriculum, so a stage that looks comfortably satisfiable on its own can
still be scarce once you account for what earlier stages already drew from
the same lane. Each lane's `repeat_factor` (cumulative tokens planned so
far / available distinct tokens) is exactly the "how many times through
this lane's content have we gone by now" number the design's learning
ledger will eventually want per shard.

When a stage's plan can't be fully satisfied, the compiler does not
silently redistribute the shortfall to other lanes -- that's an operator
decision the design doc explicitly frames as a choice (repeat, generate
synthetic data, reduce the lane's share, or move it to a later stage), not
something to automate away. Instead it applies the *configured*
scarcity_policy for the shortfall and reports the residual gap explicitly
as `unallocated_share`, so it's visible rather than papered over.
"""

from __future__ import annotations

import dataclasses
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .hashing import sha256_file


class ScarcityError(Exception):
    """A lane's protected floor (or, under "repeat", its entire target)
    cannot be satisfied even in principle -- e.g. zero tokens available."""


@dataclass
class MixtureStage:
    stage: str
    token_start: int
    token_end: int
    sequence_length: int
    mixture: Dict[str, float]
    protected_floors: Dict[str, float] = field(default_factory=dict)
    warmup_tokens: int = 0
    scarcity_policy: Optional[str] = None  # None -> use the curriculum-wide default


@dataclass
class LanePlan:
    lane: str
    target_weight: float
    effective_weight: float
    target_tokens: float
    effective_tokens: float
    available_tokens: int
    cumulative_tokens_after: float
    status: str  # "ok" | "scarce_repeat" | "scarce_reduced" | "deferred"
    repeat_factor: Optional[float]  # cumulative_tokens_after / available_tokens


@dataclass
class CompiledStage:
    stage: MixtureStage
    step_start: int
    step_end: int
    stage_token_span: int
    lane_plans: Dict[str, LanePlan]

    def effective_mixture(self) -> Dict[str, float]:
        return {lane: plan.effective_weight for lane, plan in self.lane_plans.items()}

    @property
    def total_effective_weight(self) -> float:
        return sum(plan.effective_weight for plan in self.lane_plans.values())

    @property
    def unallocated_share(self) -> float:
        return max(0.0, 1.0 - self.total_effective_weight)


@dataclass
class CompiledSchedule:
    global_batch_size: int
    scarcity_policy: str
    stages: List[CompiledStage]

    def _stage_index_at_step(self, step: int) -> int:
        for idx, cs in enumerate(self.stages):
            if cs.step_start <= step < cs.step_end:
                return idx
        if step < self.stages[0].step_start:
            raise ValueError(
                f"step {step} precedes the first stage, which starts at {self.stages[0].step_start}"
            )
        raise ValueError(
            f"step {step} is beyond the last configured stage, which ends at {self.stages[-1].step_end}"
        )

    def stage_at_step(self, step: int) -> CompiledStage:
        return self.stages[self._stage_index_at_step(step)]

    def mixture_at_step(self, step: int) -> Dict[str, float]:
        """The effective mixture at `step`, linearly ramping from the
        previous stage's mixture to this stage's over the first
        warmup_tokens of the stage (converted to steps using this stage's
        own sequence_length). The first stage never ramps -- there is
        nothing to ramp from."""
        idx = self._stage_index_at_step(step)
        stage = self.stages[idx]

        if idx == 0 or stage.stage.warmup_tokens <= 0:
            return stage.effective_mixture()

        tokens_per_step = stage.stage.sequence_length * self.global_batch_size
        warmup_steps = stage.stage.warmup_tokens // tokens_per_step
        if warmup_steps <= 0:
            return stage.effective_mixture()

        progress = min(1.0, (step - stage.step_start) / warmup_steps)
        prev_mixture = self.stages[idx - 1].effective_mixture()
        curr_mixture = stage.effective_mixture()
        lanes = set(prev_mixture) | set(curr_mixture)
        return {
            lane: (1 - progress) * prev_mixture.get(lane, 0.0) + progress * curr_mixture.get(lane, 0.0)
            for lane in lanes
        }


def _validate_stage_ordering(stages: List[MixtureStage]) -> None:
    if not stages:
        raise ValueError("no stages provided")
    for stage in stages:
        if stage.token_end <= stage.token_start:
            raise ValueError(f"stage {stage.stage!r}: token_end must be greater than token_start")

    ordered = sorted(stages, key=lambda s: s.token_start)
    if [s.stage for s in ordered] != [s.stage for s in stages]:
        raise ValueError("stages must be listed in ascending token_start order")

    for prev, curr in zip(ordered, ordered[1:]):
        if prev.token_end != curr.token_start:
            raise ValueError(
                f"stage {curr.stage!r} starts at token {curr.token_start}, but the previous stage "
                f"{prev.stage!r} ends at token {prev.token_end} -- stages must tile the training "
                f"token budget contiguously, with no gaps or overlaps"
            )


def _validate_stage_shares(stage: MixtureStage) -> None:
    weight_sum = sum(stage.mixture.values())
    if not math.isclose(weight_sum, 1.0, abs_tol=1e-6):
        raise ValueError(f"stage {stage.stage!r}: mixture weights sum to {weight_sum}, expected 1.0")

    for lane, floor in stage.protected_floors.items():
        if lane not in stage.mixture:
            raise ValueError(
                f"stage {stage.stage!r}: protected_floors names lane {lane!r}, which has no "
                f"entry in mixture"
            )
        weight = stage.mixture[lane]
        if floor > weight + 1e-9:
            raise ValueError(
                f"stage {stage.stage!r}: protected floor for {lane!r} ({floor}) exceeds its "
                f"target weight ({weight}) -- a floor cannot promise more than the stage plans "
                f"to give the lane in the first place"
            )

    floor_sum = sum(stage.protected_floors.values())
    if floor_sum > 1.0 + 1e-9:
        raise ValueError(f"stage {stage.stage!r}: protected floors sum to {floor_sum}, more than 100%")


def _compile_stage(
    stage: MixtureStage,
    lane_available_tokens: Dict[str, int],
    cumulative: Dict[str, float],
    global_batch_size: int,
    default_scarcity_policy: str,
    step_start: int,
) -> CompiledStage:
    """`step_start` is a running counter passed in from compile_curriculum,
    not derived from this stage's absolute token position -- it has to be,
    since sequence_length (and so tokens-per-step) can differ between
    stages, e.g. a long-context annealing stage. Deriving step_start from
    token_start directly would make steps overlap or run backwards the
    moment sequence_length changes between stages, since the same absolute
    token position means a different step number under a different
    tokens-per-step ratio."""
    _validate_stage_shares(stage)

    span = stage.token_end - stage.token_start
    tokens_per_step = stage.sequence_length * global_batch_size
    step_end = step_start + span // tokens_per_step

    policy = stage.scarcity_policy or default_scarcity_policy

    lane_plans: Dict[str, LanePlan] = {}
    for lane, weight in stage.mixture.items():
        target_tokens = weight * span
        floor_tokens = stage.protected_floors.get(lane, 0.0) * span
        available = lane_available_tokens.get(lane, 0)
        prior = cumulative[lane]
        projected_total = prior + target_tokens

        if projected_total <= available + 1e-9:
            status = "ok"
            effective_tokens = target_tokens
        elif policy == "repeat":
            if available <= 0:
                raise ScarcityError(
                    f"stage {stage.stage!r}: lane {lane!r} has zero available tokens; "
                    f"cannot satisfy any share of it via repetition"
                )
            status = "scarce_repeat"
            effective_tokens = target_tokens  # fully served, just by cycling shards more than once
        elif policy == "reduce_share":
            if floor_tokens > 1e-9 and available <= 0:
                # A floor can always be met via repetition as long as *some*
                # tokens exist for the lane, however extreme the resulting
                # repeat_factor -- that's a number for the operator to judge,
                # not something this compiler silently vetoes. It's only
                # truly impossible when there's nothing at all to repeat.
                raise ScarcityError(
                    f"stage {stage.stage!r}: lane {lane!r} has a protected floor of "
                    f"{floor_tokens:.0f} tokens but zero tokens are available for it at all"
                )
            remaining_capacity = max(0.0, available - prior)
            effective_tokens = max(remaining_capacity, floor_tokens)
            status = "ok" if effective_tokens >= target_tokens - 1e-6 else "scarce_reduced"
        elif policy == "defer":
            status = "deferred"
            effective_tokens = 0.0
        else:
            raise ValueError(f"unknown scarcity_policy {policy!r}")

        cumulative[lane] = prior + effective_tokens
        repeat_factor = (cumulative[lane] / available) if available > 0 else None

        lane_plans[lane] = LanePlan(
            lane=lane,
            target_weight=weight,
            effective_weight=effective_tokens / span,
            target_tokens=target_tokens,
            effective_tokens=effective_tokens,
            available_tokens=available,
            cumulative_tokens_after=cumulative[lane],
            status=status,
            repeat_factor=repeat_factor,
        )

    return CompiledStage(
        stage=stage, step_start=step_start, step_end=step_end, stage_token_span=span, lane_plans=lane_plans
    )


def compile_curriculum(
    stages: List[MixtureStage],
    lane_available_tokens: Dict[str, int],
    global_batch_size: int,
    scarcity_policy: str = "reduce_share",
) -> CompiledSchedule:
    _validate_stage_ordering(stages)

    cumulative: Dict[str, float] = {lane: 0.0 for lane in lane_available_tokens}
    compiled_stages = []
    step_cursor = 0
    for stage in stages:
        for lane in stage.mixture:
            cumulative.setdefault(lane, 0.0)
        compiled_stage = _compile_stage(
            stage, lane_available_tokens, cumulative, global_batch_size, scarcity_policy, step_cursor
        )
        compiled_stages.append(compiled_stage)
        step_cursor = compiled_stage.step_end

    return CompiledSchedule(
        global_batch_size=global_batch_size, scarcity_policy=scarcity_policy, stages=compiled_stages
    )


def print_schedule_report(schedule: CompiledSchedule) -> None:
    """Human-readable stage-by-stage report, shared by run_pipeline.py (which
    compiles the schedule as its last step) and the standalone
    compile_mixture.py (for recompiling a curriculum without rebuilding
    shards) so the two never drift into differently-formatted output."""
    for cs in schedule.stages:
        print(
            f"Stage {cs.stage.stage!r}: steps [{cs.step_start}, {cs.step_end}), "
            f"span {cs.stage_token_span} tokens, sequence_length={cs.stage.sequence_length}"
        )
        for lane, plan in sorted(cs.lane_plans.items()):
            repeat_note = (
                f", repeat_factor={plan.repeat_factor:.2f}x"
                if plan.repeat_factor is not None and plan.repeat_factor > 1.0 + 1e-9
                else ""
            )
            print(
                f"    {lane:<14} target={plan.target_weight:.2%} effective={plan.effective_weight:.2%} "
                f"[{plan.status}]{repeat_note}"
            )
        if cs.unallocated_share > 1e-9:
            print(f"    [WARN] unallocated_share={cs.unallocated_share:.2%} of this stage is unmet")
        print()


def _manifest_path_for(schedule_path: Path) -> Path:
    return schedule_path.with_name(schedule_path.stem + "_manifest" + schedule_path.suffix)


def freeze_schedule(schedule: CompiledSchedule, schedule_path: str | Path) -> dict:
    """Write the compiled schedule plus a hash-pinned manifest.

    The resume/replay guarantee ("training stream = pure function of frozen
    manifests, mixture config, seed, step") only holds if the "mixture
    config" a resumed run reads back is the *exact* schedule compiled at the
    start of the run -- not whatever compile_curriculum() would produce if
    re-run against data/manifests/ as it exists *today*, which may have
    grown new shards since. So this schedule gets frozen the same way the
    tokenizer does: hash-verified on every load, never silently
    recomputed live by a downstream consumer (e.g. the Cursor).
    """
    schedule_path = Path(schedule_path)
    schedule_path.parent.mkdir(parents=True, exist_ok=True)

    payload = dataclasses.asdict(schedule)
    with open(schedule_path, "w") as f:
        json.dump(payload, f, indent=2)

    schedule_hash = sha256_file(schedule_path)
    manifest = {
        "schedule_hash": schedule_hash,
        "global_batch_size": schedule.global_batch_size,
        "scarcity_policy": schedule.scarcity_policy,
        "stage_names": [cs.stage.stage for cs in schedule.stages],
    }
    manifest_path = _manifest_path_for(schedule_path)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    return manifest


def load_frozen_schedule(schedule_path: str | Path) -> Tuple[CompiledSchedule, dict]:
    """Load a previously frozen schedule, verifying its hash hasn't drifted."""
    schedule_path = Path(schedule_path)
    manifest_path = _manifest_path_for(schedule_path)

    if not schedule_path.exists() or not manifest_path.exists():
        raise FileNotFoundError(
            f"No frozen mixture schedule at {schedule_path}. Run scripts/compile_mixture.py first."
        )

    with open(manifest_path) as f:
        manifest = json.load(f)

    actual_hash = sha256_file(schedule_path)
    if actual_hash != manifest["schedule_hash"]:
        raise ValueError(
            "Frozen mixture schedule has changed since it was compiled: "
            f"expected {manifest['schedule_hash']}, found {actual_hash}. "
            "A changed schedule is a new artifact -- recompile to get a new "
            "schedule_hash rather than editing this one in place."
        )

    with open(schedule_path) as f:
        payload = json.load(f)

    return _schedule_from_dict(payload), manifest


def _schedule_from_dict(payload: dict) -> CompiledSchedule:
    return CompiledSchedule(
        global_batch_size=payload["global_batch_size"],
        scarcity_policy=payload["scarcity_policy"],
        stages=[_compiled_stage_from_dict(cs) for cs in payload["stages"]],
    )


def _compiled_stage_from_dict(d: dict) -> CompiledStage:
    return CompiledStage(
        stage=MixtureStage(**d["stage"]),
        step_start=d["step_start"],
        step_end=d["step_end"],
        stage_token_span=d["stage_token_span"],
        lane_plans={lane: LanePlan(**plan) for lane, plan in d["lane_plans"].items()},
    )

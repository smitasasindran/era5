"""Audit: reconstruct which shards/lanes/stages contributed over a step
range, packing utilization, and throughput -- purely by reading the
consumption + learning ledgers, no re-running of training required.

Per DATALOADER_DESIGN.md §5.14. Deliberately reads *only* what the
ledgers already recorded (`token_span_ids`, `mixture_lane`, `shard_ids`),
never the live model or a fresh Packer -- that's what makes this "audit,"
distinct from replay (tds/replay.py), which *does* recompute from scratch
to prove reconstructability. The two are complementary, not overlapping:
replay proves the stream is reproducible; audit reports on a stream that
already happened, cheaply, from records alone.

One real limitation of "ledger-only": the consumption ledger stores
`loss_mask_hash`, not the actual `loss_mask` array (deliberately -- see
§5.8, storing every position's mask for a real run would dwarf the rest
of the ledger). So "useful loss-bearing tokens" can only be *estimated*
from ledger data alone: token_span_ids give the exact token count served
per sample, and the one position every packed window always masks (its
final position, §5.6) is known structurally -- but additional
loss-masking from a structure-preserving lane's prompt portion is not
ledger-visible without recomputing the actual sample (which `tds.replay`
can do, at the cost of no longer being "purely from records"). This
module reports the honest, ledger-derivable estimate and says so.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from .batch_assembler import BatchAssembler
from .checkpoint import CheckpointManager
from .consumption_ledger import ConsumptionLedger
from .cursor import LanePool
from .manifest_store import ManifestStore
from .mixture_compiler import CompiledSchedule
from .packer import Packer


def _span_length(span: str) -> int:
    """"shard-000001:42204-42332" -> 128"""
    _, token_range = span.rsplit(":", 1)
    start, end = token_range.split("-")
    return int(end) - int(start)


def _aggregate_consumption_entries(entries: List[dict], schedule: CompiledSchedule) -> dict:
    """The actual audit computation -- shards/lanes/stages touched, packing
    utilization, estimated useful tokens -- factored out of `audit_range`
    so `audit_branch_lineage` can run the identical aggregation over a
    stitched multi-branch entry list instead of one branch's own."""
    shards_touched: set = set()
    lanes_touched: set = set()
    stages_touched: set = set()
    lane_sample_counts: Dict[str, int] = {}
    total_capacity_tokens = 0
    total_served_tokens = 0
    total_samples = 0
    total_segments = 0
    estimated_useful_tokens = 0

    for entry in entries:
        sequence_length = schedule.stage_at_step(entry["global_step"]).stage.sequence_length
        stages_touched.add(entry["curriculum_stage"])

        for i, lane in enumerate(entry["mixture_lane"]):
            lanes_touched.add(lane)
            lane_sample_counts[lane] = lane_sample_counts.get(lane, 0) + 1
            total_samples += 1
            total_capacity_tokens += sequence_length

            shards_touched.update(entry["shard_ids"][i])
            spans = entry["token_span_ids"][i]
            total_segments += len(spans)
            served = sum(_span_length(span) for span in spans)
            total_served_tokens += served
            estimated_useful_tokens += max(served - 1, 0)  # final position always loss-masked

    return {
        "shards_touched": sorted(shards_touched),
        "lanes_touched": sorted(lanes_touched),
        "curriculum_stages_touched": sorted(stages_touched),
        "lane_sample_counts": lane_sample_counts,
        "total_samples": total_samples,
        "total_capacity_tokens": total_capacity_tokens,
        "total_served_tokens": total_served_tokens,
        "packing_utilization": total_served_tokens / total_capacity_tokens if total_capacity_tokens else 0.0,
        "avg_segments_per_sample": total_segments / total_samples if total_samples else 0.0,
        "estimated_useful_tokens": estimated_useful_tokens,
        "estimate_caveat": (
            "estimated_useful_tokens assumes concatenate_and_chop masking (only the window's "
            "final position excluded); structure-preserving lanes may mask additional prompt "
            "tokens not visible from ledger records alone -- see tds/replay.py to recompute exactly"
        ),
    }


def audit_range(
    run_id: str,
    branch_id: str,
    start_step: int,
    end_step: int,
    consumption_ledger: ConsumptionLedger,
    schedule: CompiledSchedule,
) -> dict:
    """Which shards/lanes/curriculum stages contributed to steps
    [start_step, end_step), with packing utilization (served tokens /
    allocated window capacity -- always ~1.0 by this project's packing
    design, which never pads; reported as a provable fact, not assumed)
    and an estimate of useful loss-bearing tokens.

    Reports on `branch_id`'s own ledger entries only -- a forked branch's
    pre-fork history lives under its parent's branch_id instead (see
    `tds.fork`), so this alone never reconstructs a forked branch's full
    history. See `audit_branch_lineage` for that."""
    if end_step <= start_step:
        raise ValueError(f"invalid step range [{start_step}, {end_step})")

    entries = [
        e
        for e in consumption_ledger.for_branch(run_id, branch_id)
        if start_step <= e["global_step"] < end_step
    ]
    if not entries:
        raise ValueError(
            f"no consumption ledger entries recorded for (run_id={run_id!r}, "
            f"branch_id={branch_id!r}) in [{start_step}, {end_step})"
        )

    return {
        "run_id": run_id,
        "branch_id": branch_id,
        "start_step": start_step,
        "end_step": end_step,
        **_aggregate_consumption_entries(entries, schedule),
    }


@dataclass(frozen=True)
class LineageStep:
    """One link in a branch's fork ancestry. The root branch of a run
    (never forked from anything) has `parent_branch_id=None`,
    `fork_step=None`."""

    branch_id: str
    parent_branch_id: Optional[str]
    fork_step: Optional[int]


def branch_lineage(checkpoints: CheckpointManager, run_id: str, branch_id: str) -> List[LineageStep]:
    """Root-to-`branch_id` chain of forks, ordered oldest first.

    Per `tds.fork`'s own docstring, reconstructing a forked branch's full
    lineage was explicitly punted to this module: `fork_branch` never
    duplicates a parent's pre-fork ledger history, it only tags the new
    branch's *first* checkpoint with `parent_branch_id`/`fork_step`. So
    walking that chain backwards -- from `branch_id`, to whatever branch
    it was forked from, to whatever branch *that* was forked from, until a
    branch with no parent -- is the only way to know which other
    branch_ids a full history of `branch_id` needs to borrow entries from,
    and at which step boundaries.

    A branch's *first* checkpoint (`CheckpointManager.earliest_step`) is
    always the one carrying its lineage metadata: `fork_branch` is the
    only thing that ever sets `parent_branch_id`/`fork_step` when saving,
    and it only ever does so once, for the very first checkpoint under a
    new branch_id."""
    chain: List[LineageStep] = []
    seen = set()
    current = branch_id
    while True:
        if current in seen:
            raise ValueError(f"cycle detected in branch lineage of {branch_id!r} at {current!r}")
        seen.add(current)

        step = checkpoints.earliest_step(run_id, current)
        if step is None:
            raise ValueError(f"no checkpoint recorded for (run_id={run_id!r}, branch_id={current!r})")
        payload = checkpoints.load(run_id, current, step)
        parent_branch_id = payload.get("parent_branch_id")
        fork_step = payload.get("fork_step")
        chain.append(LineageStep(branch_id=current, parent_branch_id=parent_branch_id, fork_step=fork_step))

        if parent_branch_id is None:
            break
        current = parent_branch_id

    chain.reverse()
    return chain


def _branch_lineage_entries(
    lineage: List[LineageStep], run_id: str, end_step: int, consumption_ledger: ConsumptionLedger
) -> List[dict]:
    """Stitches consumption-ledger entries across every branch_id in a
    lineage chain, splitting at each fork boundary exactly where
    `tds.fork.fork_branch` itself splits history: steps [0, fork_step] of
    a child come from its parent's own ledger entries (the child's ledger
    has none there -- see tests/test_fork.py's own assertion of this),
    steps after that come from the child's own entries, up to whichever
    step is next: the *next* fork in the chain, or `end_step` for the
    final (target) branch."""
    entries: List[dict] = []
    n = len(lineage)
    for i, node in enumerate(lineage):
        segment_start = 0 if i == 0 else node.fork_step + 1
        segment_end = end_step if i == n - 1 else lineage[i + 1].fork_step + 1
        entries.extend(
            e
            for e in consumption_ledger.for_branch(run_id, node.branch_id)
            if segment_start <= e["global_step"] < segment_end
        )
    return entries


def audit_branch_lineage(
    run_id: str,
    branch_id: str,
    end_step: int,
    consumption_ledger: ConsumptionLedger,
    checkpoints: CheckpointManager,
    schedule: CompiledSchedule,
) -> dict:
    """`audit_range`'s report, but for `branch_id`'s *complete* history
    from step 0 -- across every ancestor branch its checkpoint lineage
    passes through, not just `branch_id`'s own ledger entries. For a
    branch that was never forked, this is identical to
    `audit_range(run_id, branch_id, 0, end_step, ...)`; for a forked
    branch, it also pulls in the parent's (and, transitively, the
    parent's parent's, ...) entries for the steps before the fork that
    only ever lived under those branch_ids.

    Always starts at step 0 -- unlike `audit_range`, this reconstructs a
    branch's *complete* history, so there's no independent `start_step`
    to choose."""
    if end_step <= 0:
        raise ValueError(f"invalid end_step {end_step}")

    lineage = branch_lineage(checkpoints, run_id, branch_id)
    entries = _branch_lineage_entries(lineage, run_id, end_step, consumption_ledger)
    if not entries:
        raise ValueError(
            f"no consumption ledger entries recorded across the lineage of "
            f"(run_id={run_id!r}, branch_id={branch_id!r}) in [0, {end_step})"
        )

    return {
        "run_id": run_id,
        "branch_id": branch_id,
        "start_step": 0,
        "end_step": end_step,
        "lineage": [
            {"branch_id": n.branch_id, "parent_branch_id": n.parent_branch_id, "fork_step": n.fork_step}
            for n in lineage
        ],
        **_aggregate_consumption_entries(entries, schedule),
    }


def exact_useful_tokens_range(
    start_step: int,
    end_step: int,
    seed: str,
    schedule: CompiledSchedule,
    lane_pools: Dict[str, LanePool],
    manifest_store: ManifestStore,
    shards_dir: str | Path,
    microbatch_size: int,
) -> dict:
    """The exact useful (loss-bearing) token count for [start_step, end_step),
    recomputed from a fresh Packer/BatchAssembler walk instead of estimated
    from ledger records -- this is the "see tds/replay.py to recompute
    exactly" escape hatch `audit_range`'s own `estimate_caveat` points to.

    Real `loss_mask` arrays aren't in the consumption ledger (only their
    hash is, per §5.8), so `audit_range` can only *estimate* useful tokens
    from `token_span_ids` -- correct for concatenate_and_chop lanes, an
    undercount-of-masking (i.e. an overcount of "useful") for
    structure-preserving lanes, whose prompt-portion masking isn't
    ledger-visible. This function sidesteps the ledger entirely: it
    rebuilds the same deterministic stream (same seed, schedule, lane
    pools -- same reasoning as tds.replay.replay_range, including why it
    must walk from step 0 rather than jump to start_step) and sums the
    *real* `loss_mask` array Packer actually produced, so structure-
    preserving masking is counted exactly, not estimated.

    Unlike `audit_range`, this takes no `run_id`/`branch_id` and never
    touches the consumption ledger -- it's a pure recomputation, not a
    record read, so there's nothing to look up by run/branch. Its
    `total_served_tokens` should equal `audit_range`'s over the same
    range: both describe the same deterministic stream, one read from
    records, one recomputed from scratch.
    """
    if start_step < 0 or end_step <= start_step:
        raise ValueError(f"invalid step range [{start_step}, {end_step})")

    packer = Packer(seed, schedule, lane_pools, manifest_store, shards_dir)
    assembler = BatchAssembler(packer, microbatch_size)

    total_served_tokens = 0
    exact_useful_tokens = 0
    useful_tokens_by_shard: Dict[str, int] = {}
    useful_tokens_by_lane: Dict[str, int] = {}

    for step in range(end_step):
        microbatches = assembler.assemble_step(step)
        if step < start_step:
            continue

        for mb in microbatches:
            total_served_tokens += mb.token_ids.size
            exact_useful_tokens += int(mb.loss_mask.sum())
            for row, sample in enumerate(mb.samples):
                useful_tokens_by_lane[sample.lane] = (
                    useful_tokens_by_lane.get(sample.lane, 0) + int(mb.loss_mask[row].sum())
                )
                for _local_seg, shard_id, _doc_id, w_start, w_end, _s_start, _s_end in sample.segment_boundaries:
                    count = int(mb.loss_mask[row, w_start:w_end].sum())
                    useful_tokens_by_shard[shard_id] = useful_tokens_by_shard.get(shard_id, 0) + count

    return {
        "start_step": start_step,
        "end_step": end_step,
        "total_served_tokens": total_served_tokens,
        "exact_useful_tokens": exact_useful_tokens,
        "useful_tokens_by_shard": useful_tokens_by_shard,
        "useful_tokens_by_lane": useful_tokens_by_lane,
    }


def mixture_compliance_report(
    run_id: str,
    branch_id: str,
    start_step: int,
    end_step: int,
    consumption_ledger: ConsumptionLedger,
    schedule: CompiledSchedule,
) -> dict:
    """Planned (the compiled schedule's own effective_mixture, averaged
    per-step across the range) versus actual (realized lane shares from
    the consumption ledger) -- the "Planned versus actual shares"
    evidence the assignment's Mixture compliance row asks for."""
    if end_step <= start_step:
        raise ValueError(f"invalid step range [{start_step}, {end_step})")

    entries = [
        e
        for e in consumption_ledger.for_branch(run_id, branch_id)
        if start_step <= e["global_step"] < end_step
    ]

    realized_counts: Dict[str, int] = {}
    total_samples = 0
    for entry in entries:
        for lane in entry["mixture_lane"]:
            realized_counts[lane] = realized_counts.get(lane, 0) + 1
            total_samples += 1
    realized_shares = {lane: count / total_samples for lane, count in realized_counts.items()} if total_samples else {}

    planned_sums: Dict[str, float] = {}
    num_steps = end_step - start_step
    for step in range(start_step, end_step):
        for lane, weight in schedule.mixture_at_step(step).items():
            planned_sums[lane] = planned_sums.get(lane, 0.0) + weight
    planned_shares = {lane: total / num_steps for lane, total in planned_sums.items()}

    lanes = sorted(set(planned_shares) | set(realized_shares))
    return {
        "start_step": start_step,
        "end_step": end_step,
        "total_samples": total_samples,
        "lanes": {
            lane: {
                "planned_share": planned_shares.get(lane, 0.0),
                "actual_share": realized_shares.get(lane, 0.0),
                "delta": realized_shares.get(lane, 0.0) - planned_shares.get(lane, 0.0),
            }
            for lane in lanes
        },
    }


@dataclass(frozen=True)
class StepTiming:
    global_step: int
    wall_seconds: float
    total_tokens: int
    useful_tokens: int


def throughput_report(timings: List[StepTiming]) -> dict:
    """Aggregate tokens/sec and useful-loss-bearing-tokens/sec from
    wall-clock timings recorded live during a run -- timing can only ever
    come from an actual run, never be reconstructed from ledger records
    after the fact, unlike everything else in this module."""
    if not timings:
        raise ValueError("throughput_report requires at least one timing sample")

    total_wall = sum(t.wall_seconds for t in timings)
    total_tokens = sum(t.total_tokens for t in timings)
    total_useful = sum(t.useful_tokens for t in timings)
    return {
        "num_steps": len(timings),
        "total_wall_seconds": total_wall,
        "total_tokens": total_tokens,
        "total_useful_tokens": total_useful,
        "tokens_per_second": total_tokens / total_wall if total_wall > 0 else 0.0,
        "useful_tokens_per_second": total_useful / total_wall if total_wall > 0 else 0.0,
    }

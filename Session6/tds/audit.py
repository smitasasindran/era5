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
from typing import Dict, List

from .consumption_ledger import ConsumptionLedger
from .mixture_compiler import CompiledSchedule


def _span_length(span: str) -> int:
    """"shard-000001:42204-42332" -> 128"""
    _, token_range = span.rsplit(":", 1)
    start, end = token_range.split("-")
    return int(end) - int(start)


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
    and an estimate of useful loss-bearing tokens."""
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
        "run_id": run_id,
        "branch_id": branch_id,
        "start_step": start_step,
        "end_step": end_step,
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

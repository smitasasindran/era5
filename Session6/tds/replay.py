"""Replay: recompute a historical step range from frozen config alone,
and hash-compare against the ledger already recorded for it.

Per DATALOADER_DESIGN.md §5.12, and the §6 walkthrough's own framing:
"replay and audit are nearly free extensions of [resume's] mechanism:
replay is [recompute] run for an arbitrary historical range with an
equality assertion; audit is the same recompute run for a range with a
report instead of an assertion." tds/resume.py's single-step
`verify_resume` is now built on this module's `replay_range`, rather than
duplicating the recompute-and-compare logic in two places.

`replay_range` walks *one* fresh Packer/BatchAssembler sequentially from
step 0 through the end of the requested range, regardless of where the
range starts -- required because the Packer's per-lane document
consumption position is cumulative (see tds/resume.py's note on this),
so there's no way to jump straight to an arbitrary `start_step`. What
this function does avoid is the O(n^2) trap of checking each step in the
range independently (which would replay from 0 separately for every
single step): one continuous walk means replaying a whole range costs
O(end_step) total, not O(end_step^2).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

from .batch_assembler import BatchAssembler
from .consumption_ledger import ConsumptionLedger, build_ledger_entry
from .cursor import LanePool
from .manifest_store import ManifestStore
from .mixture_compiler import CompiledSchedule
from .packer import Packer

# The fields the consumption ledger stores specifically for verification
# (§5.8) -- not the full token arrays.
VERIFIED_FIELDS = ("shard_ids", "token_span_ids", "loss_mask_hash")


@dataclass
class StepReplayResult:
    global_step: int
    matched: bool
    mismatches: List[str] = field(default_factory=list)


@dataclass
class ReplayResult:
    start_step: int
    end_step: int  # exclusive, matching CompiledSchedule's own [start, end) convention
    step_results: List[StepReplayResult]

    @property
    def matched(self) -> bool:
        return all(r.matched for r in self.step_results)

    @property
    def mismatched_steps(self) -> List[int]:
        return [r.global_step for r in self.step_results if not r.matched]


def replay_range(
    run_id: str,
    branch_id: str,
    start_step: int,
    end_step: int,
    seed: str,
    schedule: CompiledSchedule,
    lane_pools: Dict[str, LanePool],
    manifest_store: ManifestStore,
    shards_dir: str | Path,
    microbatch_size: int,
    tokenizer_hash: str,
    consumption_ledger: ConsumptionLedger,
) -> ReplayResult:
    """Recompute every step in [start_step, end_step) from a single fresh
    Packer/BatchAssembler, comparing each against what the consumption
    ledger already has on record. Steps before start_step are still
    walked (to advance the Packer to a correct position) but not
    compared or included in the result."""
    if start_step < 0 or end_step <= start_step:
        raise ValueError(f"invalid step range [{start_step}, {end_step})")

    packer = Packer(seed, schedule, lane_pools, manifest_store, shards_dir)
    assembler = BatchAssembler(packer, microbatch_size)

    step_results: List[StepReplayResult] = []
    for step in range(end_step):
        microbatches = assembler.assemble_step(step)
        if step < start_step:
            continue

        mismatches: List[str] = []
        for mb in microbatches:
            recomputed = build_ledger_entry(run_id, branch_id, mb, schedule, tokenizer_hash)
            recorded = consumption_ledger.get(run_id, branch_id, recomputed["microbatch_id"])
            if recorded is None:
                mismatches.append(f"{recomputed['microbatch_id']}: no recorded entry to compare against")
                continue
            for key in VERIFIED_FIELDS:
                if recomputed[key] != recorded[key]:
                    mismatches.append(
                        f"{recomputed['microbatch_id']}: {key} mismatch -- "
                        f"recomputed={recomputed[key]!r} recorded={recorded[key]!r}"
                    )
        step_results.append(StepReplayResult(global_step=step, matched=not mismatches, mismatches=mismatches))

    return ReplayResult(start_step=start_step, end_step=end_step, step_results=step_results)

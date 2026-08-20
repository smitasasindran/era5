"""Crash/Resume: recompute, don't restore.

Per DATALOADER_DESIGN.md §6. A checkpoint stores only (run_id, branch_id,
global_step) about the dataloader (see tds/checkpoint.py) -- resuming
means recomputing the next step's batch from a brand-new Packer/
BatchAssembler, exactly as a genuinely new process would, then proving
that recomputed batch matches what an uninterrupted "control" run already
recorded in the consumption ledger for that same step. That comparison --
not "the process didn't crash again" -- is what actually certifies
correct resume.

`verify_resume` is a thin, single-step wrapper around
`tds.replay.replay_range` -- resume is exactly a one-step replay with an
equality assertion, per §6's own framing, so this module doesn't
duplicate that recompute-and-compare logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

from .batch_assembler import BatchAssembler, Microbatch
from .consumption_ledger import ConsumptionLedger
from .cursor import LanePool
from .manifest_store import ManifestStore
from .mixture_compiler import CompiledSchedule
from .packer import Packer
from .replay import replay_range


def recompute_step(
    seed: str,
    schedule: CompiledSchedule,
    lane_pools: Dict[str, LanePool],
    manifest_store: ManifestStore,
    shards_dir: str | Path,
    microbatch_size: int,
    global_step: int,
) -> List[Microbatch]:
    """Rebuilds one step's microbatches from a brand-new Packer/
    BatchAssembler -- no state from any prior run is used or assumed, so
    this is exactly what a genuinely fresh process (possibly on different
    hardware) would produce.

    This replays every step from 0 up to and including `global_step` on
    that fresh instance, discarding all but the last. That's not an
    oversight: the Packer's per-lane document position is *cumulative* --
    which document is next depends on exactly how much of that lane's
    stream every earlier step already consumed (carry-over spans,
    document lengths, etc.) -- so unlike lane *assignment* (pick_lane,
    genuinely O(1) per step, see tds/cursor.py), there is no formula that
    jumps straight to "the state at step N" without walking through
    0..N-1 first. This is the real cost of a checkpoint that stores
    nothing but (run_id, branch_id, global_step): a small, fixed-size
    checkpoint, traded for O(global_step) resume-time recomputation --
    fine at this project's step counts, and still strictly cheaper than
    re-tokenizing or re-reading the corpus."""
    packer = Packer(seed, schedule, lane_pools, manifest_store, shards_dir)
    assembler = BatchAssembler(packer, microbatch_size)
    for step in range(global_step):
        assembler.assemble_step(step)
    return assembler.assemble_step(global_step)


@dataclass
class ResumeVerificationResult:
    global_step: int
    matched: bool
    mismatches: List[str] = field(default_factory=list)  # empty iff matched


def verify_resume(
    run_id: str,
    branch_id: str,
    resume_step: int,
    seed: str,
    schedule: CompiledSchedule,
    lane_pools: Dict[str, LanePool],
    manifest_store: ManifestStore,
    shards_dir: str | Path,
    microbatch_size: int,
    tokenizer_hash: str,
    consumption_ledger: ConsumptionLedger,
) -> ResumeVerificationResult:
    """Recompute `resume_step`'s batch from scratch and compare it,
    microbatch by microbatch, against what the consumption ledger already
    has on record for (run_id, branch_id, resume_step) -- the "control run"
    entry §6 describes. A missing control entry is reported as a mismatch
    rather than silently skipped: there's nothing to prove resume against
    if the step was never actually served by an uninterrupted run."""
    result = replay_range(
        run_id,
        branch_id,
        resume_step,
        resume_step + 1,
        seed,
        schedule,
        lane_pools,
        manifest_store,
        shards_dir,
        microbatch_size,
        tokenizer_hash,
        consumption_ledger,
    )
    step_result = result.step_results[0]
    return ResumeVerificationResult(
        global_step=step_result.global_step, matched=step_result.matched, mismatches=step_result.mismatches
    )

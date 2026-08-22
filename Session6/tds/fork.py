"""Fork: branch a new, independently-evolving stream from an existing
checkpoint.

Per DATALOADER_DESIGN.md §5.13: "Restore a checkpoint, assign a new
branch_id, and diverge the schedule (new seed component, different
mixture config, or later stage) from that exact global_step. The ledger
records the fork point (parent_branch_id, fork_step) so two branches
sharing a checkpoint but different subsequent data are always
distinguishable."

There's very little new mechanism here -- `tds.checkpoint.CheckpointManager`
already carries `parent_branch_id`/`fork_step` fields (added when that
component was built, specifically so this wouldn't need retrofitting),
and `tds.consumption_ledger.ConsumptionLedger` already keys entries by
`(run_id, branch_id, microbatch_id)` and filters via `for_branch` -- a
forked branch just writes its own new entries under its own branch_id
into the very same ledger the parent already uses. `fork_branch` is the
one genuinely new piece: restore the parent's checkpoint, then
immediately re-save it under the new branch_id with fork lineage
attached, so the new branch has its own self-contained checkpoint
history starting exactly at the fork point.

What this deliberately does *not* do: duplicate the parent's pre-fork
consumption-ledger history under the new branch_id. Steps
[0, fork_step] are identical for every branch that shares this fork point
(same seed/schedule up to here, by construction) and already live under
the parent's branch_id -- re-recording them would be redundant, and
reconstructing "one branch's full lineage across a fork" is an Audit-level
concern (walking parent_branch_id/fork_step across branches), not
something this module needs to solve. See
`tds.audit.branch_lineage`/`tds.audit.audit_branch_lineage`, which do
exactly that walk, using the very `parent_branch_id`/`fork_step` fields
this module writes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from .checkpoint import CheckpointManager


@dataclass(frozen=True)
class ForkResult:
    run_id: str
    parent_branch_id: str
    fork_step: int
    new_branch_id: str
    checkpoint_path: Path


def fork_branch(
    checkpoints: CheckpointManager,
    run_id: str,
    parent_branch_id: str,
    fork_step: int,
    new_branch_id: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> ForkResult:
    """Restores the parent branch's checkpoint at `fork_step` into
    `model`/`optimizer` (in place), then saves a new checkpoint under
    `new_branch_id` at the same `global_step`, tagged with
    `parent_branch_id`/`fork_step`. From here, the caller trains under
    `(run_id, new_branch_id)` exactly like any other resume --
    `next_step_after_checkpoint` on the returned checkpoint's metadata
    gives the first step the new branch is free to diverge on (a
    different seed, mixture config, or curriculum stage)."""
    if new_branch_id == parent_branch_id:
        raise ValueError("a fork must use a new branch_id, not the parent branch's own")

    checkpoints.restore(run_id, parent_branch_id, fork_step, model, optimizer)
    path = checkpoints.save(
        model,
        optimizer,
        run_id,
        new_branch_id,
        fork_step,
        parent_branch_id=parent_branch_id,
        fork_step=fork_step,
    )
    return ForkResult(
        run_id=run_id,
        parent_branch_id=parent_branch_id,
        fork_step=fork_step,
        new_branch_id=new_branch_id,
        checkpoint_path=path,
    )

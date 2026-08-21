"""Checkpoint Manager: atomically saves model/optimizer/RNG state plus
exactly (run_id, branch_id, global_step) -- nothing about the dataloader.

Per DATALOADER_DESIGN.md §5.10 and §6's detailed walkthrough. The whole
point: "the cursor is a pure function of (seed, mixture_config,
global_step)... there is no mutable state to lose." A checkpoint that
tried to also serialize a dataloader iterator (a shuffle buffer, a file
cursor) would be saving something crash/resume doesn't need and can't
safely trust anyway -- see tds/resume.py for the "recompute, don't
restore" half of this story.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch


@dataclass(frozen=True)
class CheckpointMetadata:
    run_id: str
    branch_id: str
    global_step: int
    parent_branch_id: Optional[str] = None  # set only for a checkpoint that originated a fork
    fork_step: Optional[int] = None


def next_step_after_checkpoint(metadata: CheckpointMetadata) -> int:
    """Named explicitly rather than inlining `+ 1` everywhere -- resume
    correctness (no skipped or repeated batch) hinges entirely on this
    being exactly one past the last *fully completed* optimizer update a
    checkpoint represents, never the checkpointed step itself (repeat) or
    two past it (skip)."""
    return metadata.global_step + 1


class CheckpointManager:
    """One checkpoint file per (run_id, branch_id, global_step), written
    atomically (temp file + os.replace, so a crash mid-write can never
    leave a corrupt checkpoint at the target path)."""

    def __init__(self, checkpoint_dir: str | Path):
        self.dir = Path(checkpoint_dir)
        self.dir.mkdir(parents=True, exist_ok=True)

    def _path(self, run_id: str, branch_id: str, global_step: int) -> Path:
        branch_dir = self.dir / run_id / branch_id
        branch_dir.mkdir(parents=True, exist_ok=True)
        return branch_dir / f"step-{global_step:012d}.pt"

    def save(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        run_id: str,
        branch_id: str,
        global_step: int,
        parent_branch_id: Optional[str] = None,
        fork_step: Optional[int] = None,
    ) -> Path:
        payload = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "rng_state": torch.get_rng_state(),
            "run_id": run_id,
            "branch_id": branch_id,
            "global_step": global_step,
            "parent_branch_id": parent_branch_id,
            "fork_step": fork_step,
        }
        path = self._path(run_id, branch_id, global_step)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, tmp_path)
        os.replace(tmp_path, path)  # atomic within the same directory/filesystem
        return path

    def load(self, run_id: str, branch_id: str, global_step: int) -> dict:
        path = self._path(run_id, branch_id, global_step)
        if not path.exists():
            raise FileNotFoundError(f"no checkpoint at {path}")
        # weights_only=False: this payload deliberately carries non-tensor
        # metadata (run_id/branch_id/global_step, optimizer state) alongside
        # tensors -- safe here because these are files this project's own
        # CheckpointManager wrote, never an untrusted external checkpoint.
        return torch.load(path, weights_only=False)

    def restore(
        self,
        run_id: str,
        branch_id: str,
        global_step: int,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
    ) -> CheckpointMetadata:
        """Loads a checkpoint and applies it to `model`/`optimizer` in
        place (including torch's global RNG state), returning just the
        three (plus fork) values the dataloader side actually needs."""
        payload = self.load(run_id, branch_id, global_step)
        model.load_state_dict(payload["model_state_dict"])
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        torch.set_rng_state(payload["rng_state"])
        return CheckpointMetadata(
            run_id=payload["run_id"],
            branch_id=payload["branch_id"],
            global_step=payload["global_step"],
            parent_branch_id=payload.get("parent_branch_id"),
            fork_step=payload.get("fork_step"),
        )

    def latest_step(self, run_id: str, branch_id: str) -> Optional[int]:
        branch_dir = self.dir / run_id / branch_id
        if not branch_dir.exists():
            return None
        steps = [int(p.stem.split("-")[1]) for p in branch_dir.glob("step-*.pt")]
        return max(steps) if steps else None

    def earliest_step(self, run_id: str, branch_id: str) -> Optional[int]:
        """The step a branch's own checkpoint history *starts* at -- for a
        forked branch this is always `fork_step` (the only checkpoint
        `fork_branch` ever writes with lineage metadata attached), which is
        exactly what `tds.audit.branch_lineage` needs to walk
        `parent_branch_id`/`fork_step` back across branches without
        needing to know which step to look at in advance."""
        branch_dir = self.dir / run_id / branch_id
        if not branch_dir.exists():
            return None
        steps = [int(p.stem.split("-")[1]) for p in branch_dir.glob("step-*.pt")]
        return min(steps) if steps else None

"""One training step: forward+backward across a global_step's microbatches,
one optimizer update, and a before/after re-evaluation of the same data.

The Learning Ledger's `avg_token_loss` and `loss_delta_before_after`
(DATALOADER_DESIGN.md §5.9, "Loss delta before and after exposure" per the
course notes' Learning Ledger section) are properties of one optimizer
update, not of one microbatch -- gradient accumulation across a step's
microbatches happens *before* the update, and the "after" comparison has
to use the *same* data the model was just exposed to. So this module
owns exactly one training step's worth of orchestration: nothing about
ledgers, checkpoints, or crash/resume lives here -- see tds/learning_ledger.py
for what turns this module's output into ledger entries.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import torch

from .batch_assembler import Microbatch
from .model import ToyTransformer, compute_batch_loss, forward_and_masked_loss


@dataclass
class TrainingStepResult:
    global_step: int
    microbatches: List[Microbatch]
    per_token_loss_before: List[np.ndarray]  # aligned with microbatches
    per_token_loss_after: List[np.ndarray]
    avg_loss_before: float
    avg_loss_after: float


def _validate(microbatches: List[Microbatch]) -> int:
    if not microbatches:
        raise ValueError("run_training_step requires at least one microbatch")
    global_step = microbatches[0].global_step
    if any(mb.global_step != global_step for mb in microbatches):
        raise ValueError("all microbatches in one training step must share the same global_step")
    return global_step


def _evaluate(model: ToyTransformer, microbatches: List[Microbatch]):
    """Read-only forward pass over every microbatch. Returns
    (per_token_loss list, overall avg_loss across all of them combined)."""
    model.eval()
    per_token_loss_list = []
    total_loss, total_count = 0.0, 0
    with torch.no_grad():
        for mb in microbatches:
            per_token_loss, _ = compute_batch_loss(model, mb)
            per_token_loss_list.append(per_token_loss)
            mask = mb.loss_mask[:, :-1]
            total_loss += float(per_token_loss.sum())
            total_count += int((mask > 0).sum())
    avg_loss = total_loss / max(total_count, 1)
    return per_token_loss_list, avg_loss


def run_training_step(
    model: ToyTransformer, optimizer: torch.optim.Optimizer, microbatches: List[Microbatch]
) -> TrainingStepResult:
    """One optimizer update: evaluate before, accumulate gradients across
    every microbatch (each scaled by 1/len(microbatches), the standard
    gradient-accumulation normalization), step, then evaluate the exact
    same data again with the updated weights."""
    global_step = _validate(microbatches)

    per_token_loss_before, avg_loss_before = _evaluate(model, microbatches)

    model.train()
    optimizer.zero_grad()
    for mb in microbatches:
        masked_loss, mask = forward_and_masked_loss(model, mb)
        denom = mask.sum().clamp(min=1)
        loss = masked_loss.sum() / denom / len(microbatches)
        loss.backward()
    optimizer.step()

    per_token_loss_after, avg_loss_after = _evaluate(model, microbatches)

    return TrainingStepResult(
        global_step=global_step,
        microbatches=list(microbatches),
        per_token_loss_before=per_token_loss_before,
        per_token_loss_after=per_token_loss_after,
        avg_loss_before=avg_loss_before,
        avg_loss_after=avg_loss_after,
    )

"""Learning Ledger: attaches training outcomes back to the data that
produced them.

Per DATALOADER_DESIGN.md §5.9. Where the consumption ledger records *what*
the model saw, this records *how it responded* -- per shard, tied to
consumption entries by `global_step`. Per the course notes' framing: "the
consumption ledger records what the model saw; the learning ledger
attaches the outcome back to the data," turning the data system into a
measuring instrument rather than just a record-keeper.

The design doc's own example entry has no run_id/branch_id -- an
intentional-looking omission, but one that would let two different runs'
(or two forked branches') entries silently collide on the same
`(global_step, shard_id)` key. Extended here the same way the consumption
ledger already was: keyed by `(run_id, branch_id, global_step, shard_id)`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .batch_assembler import Microbatch
from .consumption_ledger import ConsumptionLedger
from .mixture_compiler import CompiledSchedule
from .training_step import TrainingStepResult

USEFULNESS_THRESHOLD = 1e-3  # |loss_delta| below this counts as "neutral"


class LearningLedgerError(Exception):
    """Raised on a mismatched re-append -- see ConsumptionLedgerError for
    the identical reasoning: append-only, idempotent on identical content,
    loud on divergence."""


def accumulate_per_shard_loss(
    microbatches: List[Microbatch], per_token_loss_list: List[np.ndarray]
) -> Dict[str, float]:
    """Average loss per shard_id, aggregated across every sample in every
    microbatch. Uses each sample's segment_boundaries to attribute the
    right slice of its per_token_loss to the right shard -- a single
    sample can span multiple shards (a document that started in one shard
    and carried over into a window built from the next), and this
    attributes each shard only the positions it actually contributed."""
    totals: Dict[str, List[float]] = {}  # shard_id -> [loss_sum, count]
    for mb, per_token_loss in zip(microbatches, per_token_loss_list):
        length_minus_one = per_token_loss.shape[1]
        mask = mb.loss_mask[:, :length_minus_one] > 0
        for row, sample in enumerate(mb.samples):
            for _local_seg, shard_id, _doc_id, w_start, w_end, _s_start, _s_end in sample.segment_boundaries:
                end = min(w_end, length_minus_one)
                if end <= w_start:
                    continue
                seg_mask = mask[row, w_start:end]
                count = int(seg_mask.sum())
                if count == 0:
                    continue
                seg_losses = per_token_loss[row, w_start:end]
                loss_sum = float(seg_losses[seg_mask].sum())
                entry = totals.setdefault(shard_id, [0.0, 0])
                entry[0] += loss_sum
                entry[1] += count

    return {shard_id: loss_sum / count for shard_id, (loss_sum, count) in totals.items() if count > 0}


def model_phase_at_step(schedule: CompiledSchedule, global_step: int) -> str:
    """early / mid / late, by fraction of the schedule's total_steps
    completed -- or "anneal" outright when the current stage's name says
    so (our curriculum configs literally name a stage "anneal"), since
    that's a real, meaningful phase distinct from mechanical progress
    fraction."""
    stage_name = schedule.stage_at_step(global_step).stage.stage
    if "anneal" in stage_name.lower():
        return "anneal"
    progress = global_step / schedule.total_steps
    if progress < 1 / 3:
        return "early"
    if progress < 2 / 3:
        return "mid"
    return "late"


def classify_usefulness(loss_delta: float, threshold: float = USEFULNESS_THRESHOLD) -> str:
    """"useful" if this step's exposure measurably reduced the shard's own
    loss, "harmful" if it measurably increased it (e.g. a gradient spike),
    "neutral" otherwise. A simple, inspectable heuristic -- deliberately
    not acted on automatically in this phase, per §5.9: "recorded for the
    next version to consume."""
    if loss_delta <= -threshold:
        return "useful"
    if loss_delta >= threshold:
        return "harmful"
    return "neutral"


def _repeated_pass_number(
    consumption_ledger: ConsumptionLedger, run_id: str, branch_id: str, global_step: int, shard_id: str
) -> int:
    """How many consumption-ledger entries at or before global_step (for
    this run/branch) touched this shard -- the first appearance is pass 1.
    Computed from the consumption ledger rather than tracked separately,
    so it can never drift from what was actually recorded as served."""
    count = 0
    for entry in consumption_ledger.for_branch(run_id, branch_id):
        if entry["global_step"] > global_step:
            continue
        touched = {shard for shard_list in entry["shard_ids"] for shard in shard_list}
        if shard_id in touched:
            count += 1
    return count


def build_learning_ledger_entries(
    run_id: str,
    branch_id: str,
    result: TrainingStepResult,
    schedule: CompiledSchedule,
    consumption_ledger: ConsumptionLedger,
) -> List[dict]:
    """One entry per shard touched by this training step. Requires the
    consumption ledger to already hold this step's entries (repeated_pass_number
    counts include the current step) -- callers should write the
    consumption ledger entries for a step before building its learning
    ledger entries."""
    loss_before = accumulate_per_shard_loss(result.microbatches, result.per_token_loss_before)
    loss_after = accumulate_per_shard_loss(result.microbatches, result.per_token_loss_after)
    phase = model_phase_at_step(schedule, result.global_step)

    entries = []
    for shard_id in sorted(loss_before):
        if shard_id not in loss_after:
            continue  # defensive only -- same data evaluated twice always touches the same shards
        delta = loss_after[shard_id] - loss_before[shard_id]
        entries.append(
            {
                "run_id": run_id,
                "branch_id": branch_id,
                "global_step": result.global_step,
                "shard_id": shard_id,
                "avg_token_loss": loss_before[shard_id],
                "loss_delta_before_after": delta,
                # OPUS (tds/opus.py) now does real scoring, but that score
                # is per-document, computed once before packing, and not
                # threaded through Microbatch/TrainingStepResult here --
                # left None rather than fabricating a shard-level rollup
                # this function was never given the inputs to compute.
                "opus_score": None,
                "repeated_pass_number": _repeated_pass_number(
                    consumption_ledger, run_id, branch_id, result.global_step, shard_id
                ),
                "model_phase": phase,
                "usefulness_classification": classify_usefulness(delta),
            }
        )
    return entries


class LearningLedger:
    """Append-only registry of learning-ledger entries, keyed by
    (run_id, branch_id, global_step, shard_id) -- same append-only,
    idempotent-on-identical-reappend semantics as ConsumptionLedger."""

    def __init__(self, ledger_dir: str | Path):
        self.dir = Path(ledger_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.dir / "index.jsonl"

        self._by_key: Dict[Tuple[str, str, int, str], dict] = {}
        self._order: List[Tuple[str, str, int, str]] = []
        if self.index_path.exists():
            with open(self.index_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    entry = json.loads(line)
                    key = self._key(entry)
                    self._by_key[key] = entry
                    self._order.append(key)

    @staticmethod
    def _key(entry: dict) -> Tuple[str, str, int, str]:
        return (entry["run_id"], entry["branch_id"], entry["global_step"], entry["shard_id"])

    def append(self, entry: dict) -> None:
        key = self._key(entry)
        existing = self._by_key.get(key)
        if existing is not None:
            if existing != entry:
                raise LearningLedgerError(
                    f"Refusing to record {key}: an entry already exists with different "
                    "content. The ledger is append-only, and the same (global_step, shard_id) "
                    "must never be recorded two different ways."
                )
            return

        with open(self.index_path, "a") as f:
            f.write(json.dumps(entry) + "\n")
        self._by_key[key] = entry
        self._order.append(key)

    def get(self, run_id: str, branch_id: str, global_step: int, shard_id: str) -> Optional[dict]:
        return self._by_key.get((run_id, branch_id, global_step, shard_id))

    def all(self) -> List[dict]:
        return [self._by_key[key] for key in self._order]

    def for_branch(self, run_id: str, branch_id: str) -> List[dict]:
        return [e for e in self.all() if e["run_id"] == run_id and e["branch_id"] == branch_id]

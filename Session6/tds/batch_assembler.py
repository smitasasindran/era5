"""Batch Assembler: groups one step's packed sequences into microbatches.

Per DATALOADER_DESIGN.md §5.7. Single-GPU, so there's no rank/worker
partitioning to do: global_batch_size (already fixed by the compiled
schedule) = microbatch_size * gradient_accumulation_steps. The Packer
already produces exactly one global batch's worth of PackedSamples per
`pack_step()` call, in slot order (0..global_batch_size-1) -- this
module's only job is to slice that ordered list into consecutive,
equal-sized microbatches and stack each one's parallel arrays
(token_ids, segment_id, position_id, loss_mask) into the shape a model
actually consumes: (microbatch_size, sequence_length).

Order is preserved exactly: row i of microbatch k is
packed_samples[k * microbatch_size + i]. Concatenating a step's
microbatches back together reproduces the Packer's own slot order, which
is what makes recomputing a historical step from scratch (replay) directly
comparable, slot-for-slot, against what a live run's consumption ledger
recorded.

Attention is deliberately not materialized here either, for the same
reason as in tds/packer.py: `segment_id` is stacked and handed to the
model as an O(microbatch_size * sequence_length) array; an attention
implementation biases from it directly rather than this module building
an explicit per-sample L x L mask.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from .packer import PackedSample, Packer


@dataclass(frozen=True, eq=False)
class Microbatch:
    global_step: int
    microbatch_index: int  # 0-indexed among this step's gradient-accumulation slices
    rank: int  # always 0 in this single-GPU scope; kept for ledger schema forward-compatibility
    token_ids: np.ndarray  # (microbatch_size, sequence_length), int64
    segment_id: np.ndarray  # same shape, int64
    position_id: np.ndarray  # same shape, int64
    loss_mask: np.ndarray  # same shape, float32 -- multiplies elementwise against per-token loss
    samples: Tuple[PackedSample, ...]  # provenance: one entry per row, same order as the arrays

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Microbatch):
            return NotImplemented
        return (
            self.global_step == other.global_step
            and self.microbatch_index == other.microbatch_index
            and self.rank == other.rank
            and self.samples == other.samples
            and np.array_equal(self.token_ids, other.token_ids)
            and np.array_equal(self.segment_id, other.segment_id)
            and np.array_equal(self.position_id, other.position_id)
            and np.array_equal(self.loss_mask, other.loss_mask)
        )


def assemble_batches(packed_samples: List[PackedSample], microbatch_size: int) -> List[Microbatch]:
    """Slice one step's ordered PackedSamples into consecutive microbatches
    and stack each into model-ready arrays."""
    if microbatch_size <= 0:
        raise ValueError(f"microbatch_size must be positive, got {microbatch_size}")
    if not packed_samples:
        return []

    global_step = packed_samples[0].global_step
    if any(s.global_step != global_step for s in packed_samples):
        raise ValueError("assemble_batches expects samples from a single global_step's pack_step() call")

    global_batch_size = len(packed_samples)
    if global_batch_size % microbatch_size != 0:
        raise ValueError(
            f"global_batch_size={global_batch_size} is not evenly divisible by "
            f"microbatch_size={microbatch_size} -- gradient_accumulation_steps must be a whole number"
        )

    microbatches = []
    for microbatch_index, start in enumerate(range(0, global_batch_size, microbatch_size)):
        chunk = packed_samples[start : start + microbatch_size]
        microbatches.append(
            Microbatch(
                global_step=global_step,
                microbatch_index=microbatch_index,
                rank=0,
                token_ids=np.array([s.token_ids for s in chunk], dtype=np.int64),
                segment_id=np.array([s.segment_id for s in chunk], dtype=np.int64),
                position_id=np.array([s.position_id for s in chunk], dtype=np.int64),
                loss_mask=np.array([s.loss_mask for s in chunk], dtype=np.float32),
                samples=tuple(chunk),
            )
        )
    return microbatches


class BatchAssembler:
    """Convenience wrapper composing a Packer with a fixed microbatch_size,
    for the common case of walking a run step by step."""

    def __init__(self, packer: Packer, microbatch_size: int):
        self.packer = packer
        self.microbatch_size = microbatch_size

    def assemble_step(self, global_step: int) -> List[Microbatch]:
        return assemble_batches(self.packer.pack_step(global_step), self.microbatch_size)

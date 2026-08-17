"""Cursor: (seed, schedule, lane_pools, global_step, slot) -> CandidateItem.

Per DATALOADER_DESIGN.md §5.4, the cursor is a pure function with no
persisted dataloader state -- the training stream is fully determined by
(seed, mixture schedule, step), so a crash never loses "where we were" and
replay never needs to re-run everything from step 0 to reproduce one
interval. This module refines the design doc's shorthand pseudocode in two
concrete ways, both explained inline below: what a "candidate" actually
points at (a document, not a shard or a pre-cut window), and how a lane's
weighted turn is decided (a counter-based hash pick, not a live shuffle).

A "slot" is a document's position within one training step's batch:
slot_index = global_step * global_batch_size + slot, slot in
[0, global_batch_size). CompiledSchedule's own step numbering already
represents one training step as tokens_per_step = sequence_length *
global_batch_size tokens (see mixture_compiler.py), i.e. one step = one
full batch of global_batch_size sequences -- so the cursor needs a slot
index *within* a step, which the design doc's single-argument
`cursor(run_config, global_step)` pseudocode doesn't show but which its own
step-sizing already implies is necessary.

What a CandidateItem points at: DATALOADER_DESIGN.md §5.6 (Packer) "fills
fixed-length sequences from accepted candidates" -- candidates are the raw
material the packer concatenates and chops, not already-fixed-length
windows. That means a candidate is a *document* reference, not a shard or a
sequence_length-sized window: `lane_sequence(seed, lane)[k]` unpacks to
`(shard_id, offset)`, which is exactly a document's (shard_id, start_token)
location in that shard's manifest.document_spans. This also decouples the
cursor entirely from sequence_length (which varies by stage) -- windowing
is the packer's job, done later from whatever documents the cursor handed
it.

How a lane's turn is picked: instead of literal live weighted random
sampling (stateful, not point-queryable) or an exact proportional
round-robin (no simple closed form once lane weights change every step
during a warmup ramp), `pick_lane` uses a counter-based hash: a uniform
value in [0, 1) computed purely from (seed, global_step, slot) via SHA-256,
compared against the target mixture's cumulative weight thresholds. This
is a genuinely stateless, O(1)-per-query pick (any (step, slot) can be
evaluated in isolation, in any order), at the cost of realized lane shares
only *converging* to the target mixture rather than exactly matching it
over a small number of draws -- an honest, reproducible, auditable
trade-off explained further in IMPLEMENTATION_NOTES.md.

If a stage's effective mixture doesn't sum to 1.0 (unallocated_share > 0,
i.e. the mixture compiler couldn't fully satisfy this stage's plan even
after applying its scarcity_policy), pick_lane renormalizes across the
lanes that *do* have positive weight -- every batch slot needs some real
document, so the shortfall can't literally mean "leave this slot empty".
The unallocated_share stays a legitimate pre-flight warning for the
operator to notice; the cursor does not silently mask it, it just still
has to produce a full batch.

`picks_so_far_in_lane` (how many draws a lane has had before slot n, used
to index into its shuffled document pool) has no similarly cheap closed
form -- unlike proportional round-robin, counting how many of a hash-based
sequence's outcomes matched a given lane genuinely requires enumerating
them. `iter_candidates` does this the efficient way: a single forward pass
maintaining a running per-lane counter (O(1) amortized per slot) rather
than replaying picks_so_far_in_lane from scratch at every slot (which would
be O(n) per slot, O(n^2) overall). Resuming after a crash or replaying a
historical interval both mean the same thing here: start `iter_candidates`
fresh from slot 0 and advance it -- an O(n) one-time replay, not a restored
serialized iterator. At this project's step counts (low thousands) that
replay is a sub-second operation.
"""

from __future__ import annotations

import hashlib
import itertools
import random
from dataclasses import dataclass
from typing import Dict, Iterator, List, Tuple

from .mixture_compiler import CompiledSchedule

LanePool = List[Tuple[str, str, int]]  # (shard_id, document_id, token_offset)


@dataclass(frozen=True)
class CandidateItem:
    global_step: int
    slot: int
    lane: str
    shard_id: str
    document_id: str
    token_offset: int
    pick_index: int  # k -- how many prior slots also picked this lane
    epoch: int  # which pass through this lane's document pool this draw belongs to


def _uniform_unit_interval(seed: str, *parts) -> float:
    """A deterministic pseudo-random float in [0, 1), purely a function of
    its inputs -- a counter-based PRNG rather than a stateful stream, so any
    (seed, ...parts) can be evaluated in isolation without replaying
    anything that came before it."""
    key = ":".join([str(seed), *(str(p) for p in parts)]).encode("utf-8")
    digest = hashlib.sha256(key).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def _derived_seed(seed: str, *parts) -> int:
    key = ":".join([str(seed), *(str(p) for p in parts)]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(key).digest(), "big")


def pick_lane(seed: str, weights: Dict[str, float], global_step: int, slot: int) -> str:
    """Deterministic weighted pick among lanes with positive weight at this
    slot. Renormalizes across only the positive-weight lanes -- see module
    docstring for why a mixture that doesn't sum to 1.0 doesn't mean a slot
    goes unfilled."""
    positive = {lane: w for lane, w in weights.items() if w > 0}
    total = sum(positive.values())
    if total <= 0:
        raise ValueError(
            f"step {global_step}: no lane has positive effective weight -- "
            "this stage's mixture cannot fill any batch slot"
        )

    threshold = _uniform_unit_interval(seed, "lane", global_step, slot) * total
    cumulative = 0.0
    for lane in sorted(positive):  # deterministic order, independent of dict insertion order
        cumulative += positive[lane]
        if threshold < cumulative:
            return lane
    return sorted(positive)[-1]  # floating-point edge case: threshold landed exactly on the total


def lane_sequence(seed: str, lane: str, epoch: int, pool: LanePool) -> LanePool:
    """A seeded deterministic permutation of `pool` (all of a lane's
    documents) for the given epoch -- reshuffled with a fresh derived seed
    each time the pool is fully cycled through, rather than repeating the
    same order every epoch. "Epoch" here means a full pass through *this
    lane's* document pool specifically, not a pass over the whole corpus."""
    if not pool:
        raise ValueError(f"lane {lane!r} has no documents to draw from")
    rng = random.Random(_derived_seed(seed, "lane_sequence", lane, epoch))
    shuffled = list(pool)
    rng.shuffle(shuffled)
    return shuffled


def iter_candidates(
    seed: str, schedule: CompiledSchedule, lane_pools: Dict[str, LanePool]
) -> Iterator[CandidateItem]:
    """Yields CandidateItems for slot_index = 0, 1, 2, ... in order, up to
    schedule.total_steps, maintaining a running per-lane pick count
    incrementally. This is the efficient path for consuming many steps in
    order (a live training loop, or replaying/resuming by advancing a fresh
    generator up to the point of interest) -- see module docstring."""
    batch_size = schedule.global_batch_size
    max_step = schedule.total_steps

    counts: Dict[str, int] = {}
    step = None
    weights: Dict[str, float] = {}
    n = 0
    while True:
        this_step, slot = divmod(n, batch_size)
        if this_step >= max_step:
            return
        if this_step != step:
            step, weights = this_step, schedule.mixture_at_step(this_step)

        lane = pick_lane(seed, weights, this_step, slot)
        pool = lane_pools.get(lane, [])
        if not pool:
            raise ValueError(
                f"lane {lane!r} was picked at step {this_step} slot {slot} but has no "
                "documents in lane_pools -- schedule and shard manifests are out of sync"
            )

        k = counts.get(lane, 0)
        epoch, position = divmod(k, len(pool))
        shard_id, document_id, token_offset = lane_sequence(seed, lane, epoch, pool)[position]
        counts[lane] = k + 1

        yield CandidateItem(
            global_step=this_step,
            slot=slot,
            lane=lane,
            shard_id=shard_id,
            document_id=document_id,
            token_offset=token_offset,
            pick_index=k,
            epoch=epoch,
        )
        n += 1


def cursor(
    seed: str,
    schedule: CompiledSchedule,
    lane_pools: Dict[str, LanePool],
    global_step: int,
    slot: int,
) -> CandidateItem:
    """Point-query convenience wrapper -- what the design doc's pseudocode
    names `cursor(run_config, global_step)`. Implemented as a single slice
    into iter_candidates, so it can never disagree with bulk iteration; for
    consuming many steps in order, call iter_candidates directly instead of
    calling this in a loop (which would replay from 0 every time)."""
    batch_size = schedule.global_batch_size
    if not (0 <= slot < batch_size):
        raise ValueError(f"slot {slot} out of range for global_batch_size={batch_size}")

    slot_index = global_step * batch_size + slot
    return next(itertools.islice(iter_candidates(seed, schedule, lane_pools), slot_index, None))

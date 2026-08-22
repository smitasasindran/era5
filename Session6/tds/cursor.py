"""Cursor: deterministic (seed, schedule, step, slot) -> lane, and a
deterministic per-lane document stream -- no persisted dataloader state.

Per DATALOADER_DESIGN.md §5.4, the training stream is fully determined by
(seed, mixture schedule, step), so a crash never loses "where we were" and
replay never needs to re-run everything from step 0 to reproduce one
interval. This module has two genuinely separate jobs, previously
conflated into one `cursor()`/`iter_candidates()` pair (see below for why
that didn't survive contact with the Packer):

1. **Which lane does output sequence (global_step, slot) draw from?**
   `pick_lane` / `lane_for_slot` / `iter_lane_assignments`. This is a pure,
   O(1)-per-query decision -- it depends only on (seed, global_step, slot)
   and the compiled mixture at that step, never on document-consumption
   history.
2. **Given a lane, what's the next document to pull from it?**
   `lane_document_stream` (strict order) / `lane_epoch_stream` (a whole
   epoch's documents at once, for lookahead). Infinite, seeded, per-lane
   streams -- reshuffled each time they cycle -- that the Packer advances
   at its *own* pace, not once per slot.

Why these had to be split: a packed training sequence is filled by
concatenating as many documents as needed from one lane (DATALOADER_DESIGN.md
§5.6, "the packer fills fixed-length sequences *from* candidates" --
candidates are raw material, not already-cut windows). A short document
doesn't use up a whole sequence by itself, and a long document can span
several. So "how many documents does output sequence N consume" is not 1 --
it depends on document lengths, which only the Packer, actually walking
the token stream, can determine. Baking a fixed "one document per slot"
assumption into the cursor (an earlier version of this module did exactly
that) cannot represent that many-to-one / one-to-many relationship. Lane
*assignment* per slot has no such problem -- it never depended on document
lengths in the first place -- so it stays here, cheap and stateless; document
*consumption* moved to whatever actually paces it (the Packer).

How a lane's turn is picked: instead of live weighted random sampling
(stateful, not point-queryable) or an exact proportional round-robin (no
simple closed form once lane weights change every step during a warmup
ramp), `pick_lane` uses a counter-based hash: a uniform value in [0, 1)
computed purely from (seed, global_step, slot) via SHA-256, compared
against the target mixture's cumulative weight thresholds. This is a
genuinely stateless, O(1)-per-query pick (any (step, slot) can be evaluated
in isolation, in any order), at the cost of realized lane shares only
*converging* to the target mixture rather than exactly matching it over a
small number of draws -- an honest, reproducible, auditable trade-off.

If a stage's effective mixture doesn't sum to 1.0 (unallocated_share > 0,
i.e. the mixture compiler couldn't fully satisfy this stage's plan even
after applying its scarcity_policy), pick_lane renormalizes across the
lanes that *do* have positive weight -- every batch slot needs some real
document, so the shortfall can't literally mean "leave this slot empty".
The unallocated_share stays a legitimate pre-flight warning for the
operator to notice; the cursor does not silently mask it, it just still
has to produce a full batch.
"""

from __future__ import annotations

import hashlib
import random
from typing import Dict, Iterator, List, Tuple

from .mixture_compiler import CompiledSchedule

LanePool = List[Tuple[str, str, int]]  # (shard_id, document_id, token_offset)


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


def lane_for_slot(seed: str, schedule: CompiledSchedule, global_step: int, slot: int) -> str:
    """Point-query convenience: which lane does (global_step, slot) draw
    from? O(1) -- pick_lane has no history dependency, so this never needs
    to replay anything before it."""
    batch_size = schedule.global_batch_size
    if not (0 <= slot < batch_size):
        raise ValueError(f"slot {slot} out of range for global_batch_size={batch_size}")
    weights = schedule.mixture_at_step(global_step)
    return pick_lane(seed, weights, global_step, slot)


def iter_lane_assignments(
    seed: str, schedule: CompiledSchedule
) -> Iterator[Tuple[int, int, str]]:
    """Yields (global_step, slot, lane) for slot_index = 0, 1, 2, ... up to
    schedule.total_steps * global_batch_size. A thin, stateless convenience
    for walking every slot in order (e.g. driving the Packer one step at a
    time) -- equivalent to calling lane_for_slot for every (step, slot) in
    order, just without recomputing mixture_at_step redundantly for every
    slot in the same step."""
    batch_size = schedule.global_batch_size
    max_step = schedule.total_steps
    step = None
    weights: Dict[str, float] = {}
    n = 0
    while True:
        this_step, slot = divmod(n, batch_size)
        if this_step >= max_step:
            return
        if this_step != step:
            step, weights = this_step, schedule.mixture_at_step(this_step)
        yield this_step, slot, pick_lane(seed, weights, this_step, slot)
        n += 1


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


def lane_document_stream(seed: str, lane: str, pool: LanePool) -> Iterator[Tuple[str, str, int]]:
    """An infinite generator of (shard_id, document_id, token_offset) for
    `lane`, cycling through a freshly-reshuffled permutation of `pool` each
    time it's exhausted. The Packer advances this at its own pace -- one
    step per document consumed, not one per (global_step, slot) -- since
    how many documents a packed window needs depends on their lengths.

    "Recompute from scratch" for this stream means: start a fresh generator
    from the same (seed, lane, pool) and advance it the same number of
    times. There's no shortcut to "the k-th document" other than counting,
    exactly like lane_for_slot's counterpart for candidates in the earlier
    design -- except here that counting is the Packer's job (it already
    walks documents one at a time to fill windows), not this module's.

    This is the right access pattern for "greedy" sequence packing (take
    documents strictly in stream order). A "best_fit" packer instead needs
    a whole epoch's documents visible at once, to pick whichever fits a
    window's remaining room best rather than whatever's next in line --
    see lane_epoch_stream below."""
    epoch = 0
    while True:
        for item in lane_sequence(seed, lane, epoch, pool):
            yield item
        epoch += 1


def lane_epoch_stream(seed: str, lane: str, pool: LanePool) -> Iterator[LanePool]:
    """An infinite generator yielding one whole epoch's shuffled document
    list at a time -- epoch 0's permutation, then epoch 1's, then epoch
    2's, forever. The per-epoch counterpart to lane_document_stream (which
    flattens the same epochs into a single infinite per-document stream):
    used by packing policies that need lookahead across a whole epoch's
    documents (e.g. best-fit sequence packing choosing the tightest-fitting
    remaining document for a window's remaining room), rather than
    consuming one document at a time in a fixed order."""
    epoch = 0
    while True:
        yield lane_sequence(seed, lane, epoch, pool)
        epoch += 1

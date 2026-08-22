"""Packer: fills fixed-length sequences from candidate documents.

Per DATALOADER_DESIGN.md §5.6. A packed sequence is built by pulling
documents from one lane's document stream (tds/cursor.py:lane_document_stream)
and concatenating their tokens (each document's own trailing <eos>, baked
in at shard-build time, already marks its boundary) until the window is
full. Two things follow directly from "pulling documents until full,"
neither of which a fixed one-document-per-sequence design could represent:

- A short document doesn't use up a whole window by itself -- several get
  concatenated into one.
- A long document can overflow a window -- it gets sliced, and the
  remainder is *carried over* into the lane's next window rather than
  dropped or re-read from its own start. No document is skipped, split
  incorrectly, or duplicated across windows.

Per packed sequence, four parallel arrays are produced alongside
`token_ids` (the packed sample record -- see PackedSample below):

- `segment_id`: a *local* index (0, 1, 2, ...) identifying which document's
  contribution a token belongs to -- only needs to be locally unique
  within this one window, never persisted/compared across windows.
- `position_id`: resets to 0 at each segment boundary (correct for RoPE).
- `loss_mask`: 1 by default; 0 at the window's final position (no
  next-token target within this window); 0 over the *prompt* portion of a
  segment in a "structure_preserving"-policy lane. Padding (0 for
  positions the stream couldn't fill) doesn't arise in this design:
  lane_document_stream cycles forever, so a window is always fully
  fillable -- there is no "ran out of data" case to pad for.
- `attention_rule`: causal AND same-segment. Per the design doc, this is
  never materialized as an explicit L x L matrix in training -- segment_id
  is the O(L) representation an attention implementation would bias from
  directly. `attention_bias_from_segments` below materializes the boolean
  matrix anyway, but only as a verification/test utility, not part of
  PackedSample.

Loss-masking policy is selected per lane (STRUCTURE_PRESERVING_LANES from
tds/shard_builder.py):

- "concatenate_and_chop" (default): every position's loss_mask is 1 except
  the window's final position -- plain-pretraining scope, "a new document
  starts here" is itself useful signal, so even an EOS-to-next-document
  transition stays loss-visible.
- "structure_preserving": for lanes with a prompt/response split
  (`response_start_token` on the document's manifest span, set by the
  shard builder), positions before the response boundary are additionally
  loss-masked. A document in a structure-preserving lane without a
  detectable split (`response_start_token is None`, or degraded to its own
  start when no marker was found) is treated as fully response --
  no extra masking, same as concatenate_and_chop for that one document.

Orthogonal to that: `sequence_packing_policy` (a Packer-wide setting,
mirroring the shard builder's `packing_policy`) decides *which* document
gets pulled next to fill a window's remaining room, not what gets
loss-masked:

- "greedy" (default): consume `lane_document_stream` in strict order --
  whatever's next, regardless of length. Simple, order-faithful, but a
  document much bigger than the remaining room forces an immediate
  carry-over/split.
- "best_fit": look across the current epoch's not-yet-consumed documents
  (`lane_epoch_stream`) and pick whichever fits the remaining room most
  tightly -- the largest document that still fits, falling back to the
  *smallest* remaining document (to minimize how much overflows) only when
  none fits at all. Reduces how often documents get split across windows,
  at the cost of no longer preserving the epoch's shuffle order. Carry-over
  still exists and behaves identically either way -- best_fit only changes
  *which* document gets picked when there's no active carry, never how an
  overflowing one gets sliced.

Either sequence_packing_policy combines freely with either loss-masking
policy -- e.g. an "instruction" lane can be packed best_fit *and*
structure_preserving at once.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from .cursor import LanePool, lane_document_stream, lane_epoch_stream, lane_for_slot
from .hashing import sha256_bytes
from .manifest_store import ManifestStore
from .mixture_compiler import CompiledSchedule
from .shard_builder import STRUCTURE_PRESERVING_LANES

# (local_segment_id, shard_id, document_id,
#  window_start, window_end,      -- this segment's token range within the packed window
#  shard_start, shard_end)        -- the absolute shard-token range it was read from
SegmentBoundary = Tuple[int, str, str, int, int, int, int]

SEQUENCE_PACKING_POLICIES = ("greedy", "best_fit")

# (shard_id, document_id, token_offset, length)
PendingDocument = Tuple[str, str, int, int]


@dataclass(frozen=True)
class PackedSample:
    global_step: int
    slot: int
    lane: str
    packing_policy: str  # "concatenate_and_chop" | "structure_preserving" -- loss masking
    sequence_packing_policy: str  # "greedy" | "best_fit" -- document selection
    token_ids: Tuple[int, ...]
    segment_id: Tuple[int, ...]
    position_id: Tuple[int, ...]
    loss_mask: Tuple[int, ...]
    segment_boundaries: Tuple[SegmentBoundary, ...]
    loss_mask_hash: str


def attention_bias_from_segments(segment_id: Sequence[int]) -> np.ndarray:
    """Materializes the boolean causal+same-segment attention mask (True =
    may attend) as an explicit L x L array, purely for verification/tests --
    real attention computation should bias directly from segment_id (O(L)),
    never build this O(L^2) matrix. See module docstring."""
    seg = np.asarray(segment_id)
    length = len(seg)
    causal = np.tril(np.ones((length, length), dtype=bool))
    same_segment = seg[:, None] == seg[None, :]
    return causal & same_segment


class Packer:
    """Stateful only in the sense a live training loop's dataloader is
    stateful: a running per-lane document-stream position and a possible
    carried-over partial document. All of that state is itself a pure
    function of (seed, lane_pools) plus "how many documents has this lane's
    stream produced so far" -- reconstructible by building a fresh Packer
    and replaying pack_step calls from global_step 0, exactly like Cursor's
    replay/resume story."""

    def __init__(
        self,
        seed: str,
        schedule: CompiledSchedule,
        lane_pools: Dict[str, LanePool],
        manifest_store: ManifestStore,
        shards_dir: str | Path,
        structure_preserving_lanes: set = STRUCTURE_PRESERVING_LANES,
        sequence_packing_policy: str = "greedy",
    ):
        if sequence_packing_policy not in SEQUENCE_PACKING_POLICIES:
            raise ValueError(
                f"unknown sequence_packing_policy={sequence_packing_policy!r}; "
                f"must be one of {SEQUENCE_PACKING_POLICIES}"
            )

        self.seed = seed
        self.schedule = schedule
        self.lane_pools = lane_pools
        self.store = manifest_store
        self.shards_dir = Path(shards_dir)
        self.structure_preserving_lanes = structure_preserving_lanes
        self.sequence_packing_policy = sequence_packing_policy

        self._streams: Dict[str, Iterator[Tuple[str, str, int]]] = {}
        self._epoch_streams: Dict[str, Iterator[LanePool]] = {}
        self._epoch_pending: Dict[str, List[PendingDocument]] = {}
        # (shard_id, document_id, next_read_pos, doc_end_token) or None
        self._carry: Dict[str, Optional[Tuple[str, str, int, int]]] = {}
        self._token_cache: Dict[str, np.ndarray] = {}
        self._span_cache: Dict[Tuple[str, str], dict] = {}

    def pack_step(self, global_step: int) -> List[PackedSample]:
        """One full batch's worth of packed sequences for this step."""
        stage = self.schedule.stage_at_step(global_step)
        sequence_length = stage.stage.sequence_length
        return [
            self._pack_one(global_step, slot, sequence_length)
            for slot in range(self.schedule.global_batch_size)
        ]

    def _pack_one(self, global_step: int, slot: int, sequence_length: int) -> PackedSample:
        lane = lane_for_slot(self.seed, self.schedule, global_step, slot)
        loss_masking_policy = (
            "structure_preserving" if lane in self.structure_preserving_lanes else "concatenate_and_chop"
        )
        token_ids, boundaries = self._fill_window(lane, sequence_length)

        segment_id: List[int] = []
        position_id: List[int] = []
        for local_seg, _, _, w_start, w_end, _, _ in boundaries:
            length = w_end - w_start
            segment_id.extend([local_seg] * length)
            position_id.extend(range(length))

        loss_mask = self._loss_mask(boundaries, sequence_length, loss_masking_policy)
        loss_mask_hash = sha256_bytes(np.array(loss_mask, dtype=np.uint8).tobytes())

        return PackedSample(
            global_step=global_step,
            slot=slot,
            lane=lane,
            packing_policy=loss_masking_policy,
            sequence_packing_policy=self.sequence_packing_policy,
            token_ids=tuple(token_ids),
            segment_id=tuple(segment_id),
            position_id=tuple(position_id),
            loss_mask=tuple(loss_mask),
            segment_boundaries=tuple(boundaries),
            loss_mask_hash=loss_mask_hash,
        )

    def _loss_mask(
        self, boundaries: List[SegmentBoundary], sequence_length: int, policy: str
    ) -> List[int]:
        loss_mask = [1] * sequence_length
        if policy == "structure_preserving":
            for _, shard_id, document_id, w_start, _, s_start, s_end in boundaries:
                response_start = self._span_for(shard_id, document_id).get("response_start_token")
                if response_start is None or response_start <= s_start:
                    continue  # fully response -- no extra masking for this segment
                prompt_len = min(response_start, s_end) - s_start
                for pos in range(w_start, w_start + prompt_len):
                    loss_mask[pos] = 0
        loss_mask[-1] = 0  # window's final position has no next-token target
        return loss_mask

    def _fill_window(
        self, lane: str, sequence_length: int
    ) -> Tuple[List[int], List[SegmentBoundary]]:
        """Shared filling loop for both sequence_packing_policy values --
        they differ only in *which* document gets pulled next when there's
        no carry-over in progress (`next_document`), never in how an
        overflowing document gets sliced/carried."""
        next_document = self._next_document_picker(lane)
        token_ids: List[int] = []
        boundaries: List[SegmentBoundary] = []
        local_seg = 0

        while len(token_ids) < sequence_length:
            carry = self._carry.get(lane)
            if carry is not None:
                shard_id, document_id, read_pos, doc_end = carry
            else:
                room = sequence_length - len(token_ids)
                shard_id, document_id, doc_start = next_document(room)
                doc_end = self._span_for(shard_id, document_id)["end_token"]
                read_pos = doc_start

            arr = self._tokens_for(shard_id)
            room = sequence_length - len(token_ids)
            take = min(room, doc_end - read_pos)

            w_start = len(token_ids)
            token_ids.extend(int(x) for x in arr[read_pos : read_pos + take])
            w_end = len(token_ids)
            s_start, s_end = read_pos, read_pos + take
            boundaries.append((local_seg, shard_id, document_id, w_start, w_end, s_start, s_end))
            local_seg += 1

            next_read_pos = read_pos + take
            self._carry[lane] = (
                (shard_id, document_id, next_read_pos, doc_end) if next_read_pos < doc_end else None
            )

        return token_ids, boundaries

    def _next_document_picker(self, lane: str) -> Callable[[int], Tuple[str, str, int]]:
        if self.sequence_packing_policy == "greedy":
            stream = self._stream_for(lane)
            return lambda room: next(stream)
        return lambda room: self._pick_best_fit_document(lane, room)

    def _pick_best_fit_document(self, lane: str, room: int) -> Tuple[str, str, int]:
        """Among the current epoch's not-yet-consumed documents, pick the
        largest one that still fits within `room` (tightest fit, minimizing
        leftover room) -- or, if none fits at all, the smallest one overall
        (minimizing how much has to carry over into the next window).
        Ties break toward whichever appears earliest in the epoch's
        shuffle order, for determinism."""
        pending = self._epoch_pending.get(lane)
        if not pending:
            pending = self._next_epoch_pending(lane)

        fits = [i for i, item in enumerate(pending) if item[3] <= room]
        if fits:
            best_idx = max(fits, key=lambda i: pending[i][3])
        else:
            best_idx = min(range(len(pending)), key=lambda i: pending[i][3])

        shard_id, document_id, offset, _length = pending.pop(best_idx)
        return shard_id, document_id, offset

    def _next_epoch_pending(self, lane: str) -> List[PendingDocument]:
        epoch_documents = next(self._epoch_stream_for(lane))
        pending = [
            (shard_id, document_id, offset, self._span_for(shard_id, document_id)["end_token"] - offset)
            for shard_id, document_id, offset in epoch_documents
        ]
        self._epoch_pending[lane] = pending
        return pending

    def _stream_for(self, lane: str) -> Iterator[Tuple[str, str, int]]:
        if lane not in self._streams:
            self._streams[lane] = lane_document_stream(self.seed, lane, self.lane_pools.get(lane, []))
        return self._streams[lane]

    def _epoch_stream_for(self, lane: str) -> Iterator[LanePool]:
        if lane not in self._epoch_streams:
            self._epoch_streams[lane] = lane_epoch_stream(self.seed, lane, self.lane_pools.get(lane, []))
        return self._epoch_streams[lane]

    def _tokens_for(self, shard_id: str) -> np.ndarray:
        if shard_id not in self._token_cache:
            self._token_cache[shard_id] = np.load(self.shards_dir / f"{shard_id}.npy")
        return self._token_cache[shard_id]

    def _span_for(self, shard_id: str, document_id: str) -> dict:
        key = (shard_id, document_id)
        if key not in self._span_cache:
            manifest = self.store.get(shard_id)
            for span in manifest["document_spans"]:
                self._span_cache[(shard_id, span["document_id"])] = span
        return self._span_cache[key]

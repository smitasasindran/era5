"""OPUS Selector: proxy-scored accept/reject/defer over candidate
documents, upgraded here from the identity pass-through stub to a real,
config-toggleable selector (see tds/config.py's OpusConfig).

Per DATALOADER_DESIGN.md §5.5, OPUS "sits between the cursor and the
packer." Concretely here: filtering happens *once*, before any Packer or
Cursor exists, over the raw `lane_pools` (`ManifestStore.document_pool_by_lane()`)
-- producing a filtered `lane_pools` dict that gets handed to
`Packer(..., lane_pools=filtered_pools)` with zero changes to Packer or
Cursor. Every guarantee those two components already have (model-oblivious,
pure functions of frozen inputs; replay/resume never touching a live
model) survives OPUS's introduction completely intact, because OPUS's
output *is* just another frozen input, exactly like the tokenizer or the
compiled mixture schedule.

Scoring: each candidate document's average per-token loss under a given
model snapshot (up to `max_sequence_length` tokens, truncated; no
backward pass). The course notes describe OPUS as "estimating which
updates would be most useful against a proxy direction" -- realized here
as the simplest honest version of that idea: how surprising is this
document to the model right now. High loss is treated as scarce/valuable
(genuinely novel content), *unless* it's an extreme outlier (more likely
corrupted or degenerate text than a rare gem), which gets deferred rather
than trusted outright:

    score < reject_below  -> "rejected"  (low_proxy_utility)
    score > defer_above   -> "deferred"  (anomalous_high_loss)
    otherwise             -> "accepted"

A document whose lane is in `protected_lanes` is always accepted
(`protected_floor_override=True` if it would otherwise have been
rejected/deferred) -- a scarce, protected capability lane must never be
starved by the proxy's own judgment, matching the design's own framing of
floor rescue.

Because scoring depends on a *specific* model snapshot, the result is not
purely reproducible from frozen config alone the way the tokenizer or
mixture schedule are -- a model that's since been trained further would
score every candidate differently. So the decision log itself must be
frozen and loaded back explicitly (`freeze_opus_selection` /
`load_frozen_opus_selection`), never silently re-run against whatever the
"current" model happens to be at some later point.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Tuple

import numpy as np

from .batch_assembler import Microbatch
from .cursor import LanePool
from .hashing import sha256_file
from .manifest_store import ManifestStore
from .model import ToyTransformer, compute_batch_loss

STATUSES = ("accepted", "rejected", "deferred")


@dataclass(frozen=True)
class OpusDecision:
    candidate_id: str
    shard_id: str
    document_id: str
    capability_lane: str
    opus_score: Optional[float]  # None when OPUS is disabled -- no scoring was performed
    status: str  # "accepted" | "rejected" | "deferred"
    rejection_reason: Optional[str]
    protected_floor_override: bool
    effective_token_estimate: int


def score_candidate(model: ToyTransformer, tokens: np.ndarray, max_sequence_length: int) -> float:
    """Average per-token cross-entropy loss of `tokens` under `model`,
    truncated to `max_sequence_length`. Reuses tds.model.compute_batch_loss
    via a single-row, single-segment Microbatch, rather than re-deriving
    attention/position handling here -- a lone candidate document is just
    the degenerate case of one segment spanning the whole window."""
    window = tokens[:max_sequence_length].astype(np.int64)
    length = len(window)
    if length < 2:
        return 0.0  # too short to have any next-token prediction at all
    mb = Microbatch(
        global_step=-1,
        microbatch_index=-1,
        rank=0,
        token_ids=window.reshape(1, length),
        segment_id=np.zeros((1, length), dtype=np.int64),
        position_id=np.arange(length, dtype=np.int64).reshape(1, length),
        loss_mask=np.ones((1, length), dtype=np.float32),
        samples=(),
    )
    _, avg_loss = compute_batch_loss(model, mb)
    return avg_loss


def apply_opus_selection(
    lane_pools: Dict[str, LanePool],
    model: Optional[ToyTransformer],
    manifest_store: ManifestStore,
    shards_dir: str | Path,
    max_sequence_length: int,
    reject_below: float,
    defer_above: float,
    protected_lanes: FrozenSet[str] = frozenset(),
    enabled: bool = True,
) -> Tuple[Dict[str, LanePool], List[OpusDecision]]:
    """Scores every document in every lane's pool once, returning
    (filtered_pools, decisions). `model` may be None when `enabled=False`
    -- disabled selection performs no forward passes at all and accepts
    every candidate, so no model is needed to run the pass-through path."""
    if enabled and model is None:
        raise ValueError("apply_opus_selection requires a model when enabled=True")

    shards_dir = Path(shards_dir)
    token_cache: Dict[str, np.ndarray] = {}
    span_cache: Dict[Tuple[str, str], dict] = {}

    def tokens_for(shard_id: str) -> np.ndarray:
        if shard_id not in token_cache:
            token_cache[shard_id] = np.load(shards_dir / f"{shard_id}.npy")
        return token_cache[shard_id]

    def span_for(shard_id: str, document_id: str) -> dict:
        key = (shard_id, document_id)
        if key not in span_cache:
            manifest = manifest_store.get(shard_id)
            for span in manifest["document_spans"]:
                span_cache[(shard_id, span["document_id"])] = span
        return span_cache[key]

    filtered_pools: Dict[str, LanePool] = {}
    decisions: List[OpusDecision] = []

    for lane in sorted(lane_pools):
        kept: LanePool = []
        for shard_id, document_id, offset in lane_pools[lane]:
            span = span_for(shard_id, document_id)
            doc_length = span["end_token"] - span["start_token"]

            if enabled:
                doc_tokens = tokens_for(shard_id)[span["start_token"] : span["end_token"]]
                score = score_candidate(model, doc_tokens, max_sequence_length)
                if score < reject_below:
                    status, reason = "rejected", "low_proxy_utility"
                elif score > defer_above:
                    status, reason = "deferred", "anomalous_high_loss"
                else:
                    status, reason = "accepted", None
            else:
                score, status, reason = None, "accepted", None

            override = False
            if lane in protected_lanes and status != "accepted":
                status, reason, override = "accepted", None, True

            if status == "accepted":
                kept.append((shard_id, document_id, offset))

            decisions.append(
                OpusDecision(
                    candidate_id=f"opus-{shard_id}-{document_id}",
                    shard_id=shard_id,
                    document_id=document_id,
                    capability_lane=lane,
                    opus_score=score,
                    status=status,
                    rejection_reason=reason,
                    protected_floor_override=override,
                    effective_token_estimate=doc_length if status == "accepted" else 0,
                )
            )
        filtered_pools[lane] = kept

    return filtered_pools, decisions


def filtered_pools_from_decisions(
    lane_pools: Dict[str, LanePool], decisions: List[OpusDecision]
) -> Dict[str, LanePool]:
    """Reconstructs the accepted-only pools a frozen decision log implies,
    filtering the full pool down to exactly the accepted candidate_ids and
    preserving the original relative order -- Cursor's own seeded
    permutation logic then applies on top of this subset, unchanged."""
    accepted_ids = {(d.shard_id, d.document_id) for d in decisions if d.status == "accepted"}
    return {
        lane: [item for item in pool if (item[0], item[1]) in accepted_ids]
        for lane, pool in lane_pools.items()
    }


# --- Freezing: OPUS's output must be a fixed input to Packer, like the ---
# --- tokenizer and mixture schedule, never re-run against a model that ---
# --- may have moved on since selection happened.                       ---

MANIFEST_SUFFIX = "_manifest"


def _manifest_path_for(decisions_path: Path) -> Path:
    return decisions_path.with_name(decisions_path.stem + MANIFEST_SUFFIX + decisions_path.suffix)


def freeze_opus_selection(decisions: List[OpusDecision], decisions_path: str | Path) -> dict:
    decisions_path = Path(decisions_path)
    decisions_path.parent.mkdir(parents=True, exist_ok=True)

    with open(decisions_path, "w") as f:
        json.dump([asdict(d) for d in decisions], f, indent=2)

    decisions_hash = sha256_file(decisions_path)
    manifest = {
        "decisions_hash": decisions_hash,
        "candidate_count": len(decisions),
        "accepted_count": sum(1 for d in decisions if d.status == "accepted"),
        "rejected_count": sum(1 for d in decisions if d.status == "rejected"),
        "deferred_count": sum(1 for d in decisions if d.status == "deferred"),
    }
    with open(_manifest_path_for(decisions_path), "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def load_frozen_opus_selection(decisions_path: str | Path) -> Tuple[List[OpusDecision], dict]:
    decisions_path = Path(decisions_path)
    manifest_path = _manifest_path_for(decisions_path)

    if not decisions_path.exists() or not manifest_path.exists():
        raise FileNotFoundError(
            f"No frozen OPUS selection at {decisions_path}. Run scripts/run_opus_selection.py first."
        )

    with open(manifest_path) as f:
        manifest = json.load(f)

    actual_hash = sha256_file(decisions_path)
    if actual_hash != manifest["decisions_hash"]:
        raise ValueError(
            "Frozen OPUS selection has changed since it was computed: "
            f"expected {manifest['decisions_hash']}, found {actual_hash}. "
            "A changed selection is a new artifact -- rerun scripts/run_opus_selection.py "
            "to get a new decisions_hash rather than editing this one in place."
        )

    with open(decisions_path) as f:
        raw = json.load(f)
    decisions = [OpusDecision(**d) for d in raw]
    return decisions, manifest

"""Consumption Ledger: append-only record of what was actually served.

Per DATALOADER_DESIGN.md §5.8: one record per microbatch actually served --
never a live queue. This is the run's memory; resume, replay, and audit
all read from it, never from a running iterator's internal state.

The design doc's own example JSON record has singular `mixture_lane` /
`loss_mask_hash` fields, which only holds together if a microbatch never
mixes lanes or spans multiple shards per sample. Ours can do both (a real
microbatch drawn from the compiled schedule routinely spans several lanes
-- see IMPLEMENTATION_NOTES.md's Packer verification, e.g.
`['code', 'code', 'qa', 'code']` for one microbatch), so the per-sample
fields here (`packed_sample_ids`, `mixture_lane`, `shard_ids`,
`token_span_ids`, `opus_decision_id`) are parallel lists, one entry per row
of the microbatch, in the same row order as `Microbatch.samples`.
`loss_mask_hash` stays a single hash, but of the *whole stacked array*
(every row), not of the concept of "one sample's mask" -- that's what's
actually served to the model in one shot, and what a replayed step must
reproduce byte-for-byte to prove a match.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .batch_assembler import Microbatch
from .hashing import sha256_bytes
from .mixture_compiler import CompiledSchedule

DATALOADER_VERSION = "tds-v1"  # bump when the packing/consumption schema changes incompatibly


class ConsumptionLedgerError(Exception):
    """Raised when a caller tries to record a microbatch_id that's already
    on record with *different* content. Append-only means a re-append of
    identical content (a crash-recovery retry replaying the same step) is
    a harmless no-op -- but a mismatched re-append means the same
    microbatch_id was served two different ways, exactly the "repeated
    batch diverged from the original" corruption crash recovery exists to
    rule out, so it's rejected loudly rather than silently overwritten."""


def build_ledger_entry(
    run_id: str,
    branch_id: str,
    microbatch: Microbatch,
    schedule: CompiledSchedule,
    tokenizer_hash: str,
) -> dict:
    """Pure function: a Microbatch + run-level context -> one ledger
    record. No I/O -- ConsumptionLedger.append() is the only thing that
    persists it, so this can be called freely (including by a future
    replay path) without touching disk."""
    stage_name = schedule.stage_at_step(microbatch.global_step).stage.stage
    microbatch_id = f"mb-{microbatch.global_step}-{microbatch.microbatch_index}"

    packed_sample_ids: List[str] = []
    mixture_lane: List[str] = []
    shard_ids: List[List[str]] = []
    token_span_ids: List[List[str]] = []
    opus_decision_id: List[Optional[str]] = []

    for sample in microbatch.samples:
        packed_sample_ids.append(f"ps-{sample.global_step}-{sample.slot}")
        mixture_lane.append(sample.lane)

        sample_shard_ids: List[str] = []
        sample_spans: List[str] = []
        for _local_seg, shard_id, _document_id, _w_start, _w_end, s_start, s_end in sample.segment_boundaries:
            if shard_id not in sample_shard_ids:
                sample_shard_ids.append(shard_id)
            sample_spans.append(f"{shard_id}:{s_start}-{s_end}")
        shard_ids.append(sample_shard_ids)
        token_span_ids.append(sample_spans)

        # OPUS is currently an identity pass-through stub -- no real
        # decisions exist yet to attach an id to.
        opus_decision_id.append(None)

    return {
        "run_id": run_id,
        "branch_id": branch_id,
        "global_step": microbatch.global_step,
        "microbatch_index": microbatch.microbatch_index,
        "microbatch_id": microbatch_id,
        "rank": microbatch.rank,
        "curriculum_stage": stage_name,
        "packed_sample_ids": packed_sample_ids,
        "mixture_lane": mixture_lane,
        "shard_ids": shard_ids,
        "token_span_ids": token_span_ids,
        "loss_mask_hash": sha256_bytes(microbatch.loss_mask.tobytes()),
        "attention_policy": "causal+segment",
        "position_policy": "reset_per_segment",
        "tokenizer_version": tokenizer_hash,
        "dataloader_version": DATALOADER_VERSION,
        "opus_decision_id": opus_decision_id,
    }


class ConsumptionLedger:
    """Append-only registry of consumed microbatches -- mirrors
    ManifestStore's own append-only, idempotent-on-identical-reappend
    semantics, keyed by (run_id, branch_id, microbatch_id)."""

    def __init__(self, ledger_dir: str | Path):
        self.dir = Path(ledger_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.dir / "index.jsonl"

        self._by_key: Dict[Tuple[str, str, str], dict] = {}
        self._order: List[Tuple[str, str, str]] = []
        if self.index_path.exists():
            with open(self.index_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    entry = json.loads(line)
                    key = (entry["run_id"], entry["branch_id"], entry["microbatch_id"])
                    self._by_key[key] = entry
                    self._order.append(key)

    def append(self, entry: dict) -> None:
        key = (entry["run_id"], entry["branch_id"], entry["microbatch_id"])
        existing = self._by_key.get(key)
        if existing is not None:
            if existing != entry:
                raise ConsumptionLedgerError(
                    f"Refusing to record {key}: an entry already exists with different "
                    "content. The ledger is append-only, and the same microbatch_id must "
                    "never be served two different ways."
                )
            return

        with open(self.index_path, "a") as f:
            f.write(json.dumps(entry) + "\n")
        self._by_key[key] = entry
        self._order.append(key)

    def get(self, run_id: str, branch_id: str, microbatch_id: str) -> Optional[dict]:
        return self._by_key.get((run_id, branch_id, microbatch_id))

    def all(self) -> List[dict]:
        return [self._by_key[key] for key in self._order]

    def for_branch(self, run_id: str, branch_id: str) -> List[dict]:
        return [e for e in self.all() if e["run_id"] == run_id and e["branch_id"] == branch_id]

    def last_recorded_microbatch(self, run_id: str, branch_id: str) -> Optional[Tuple[int, int]]:
        """The (global_step, microbatch_index) pair with the highest such
        tuple recorded for this branch, or None if nothing's recorded yet.

        Deliberately just a fact about what's on record -- not a "what
        should run next" decision. Answering that also needs
        gradient_accumulation_steps (to know whether a step's microbatches
        are all present or a crash landed mid-step), which isn't ledger
        state; that reasoning belongs to the not-yet-built crash/resume
        component, not to this query."""
        entries = self.for_branch(run_id, branch_id)
        if not entries:
            return None
        return max((e["global_step"], e["microbatch_index"]) for e in entries)

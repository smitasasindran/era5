"""Documents -> immutable tokenized shards + manifests.

Each shard is lane-homogeneous (every token in it comes from one capability
lane) -- the mixture/cursor layer later depends on being able to treat "a
lane's shards" as a clean, orderable set. Documents are never split across
a shard boundary, so a document's tokens always live in exactly one
shard's document_spans, and `segment_id` at pack time can always be derived
from a single shard's manifest.

Two policies decide *how* a lane's documents are grouped into shards:

- "greedy": preserves corpus arrival order, filling each shard until the
  next document would overflow it. Simple and order-faithful, but
  vulnerable to wasting capacity when arrival order happens to combine
  badly with the budget (see tests for a concrete worst case).
- "best_fit": sorts documents longest-first, then places each into whichever
  open shard has the least remaining room that still fits it (opening a new
  one only when nothing fits) -- the classic best-fit-decreasing bin-packing
  heuristic. Reduces wasted shard capacity at the cost of no longer
  preserving corpus arrival order within a lane.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Tuple

import numpy as np

from .corpus import Document
from .hashing import sha256_bytes
from .manifest_store import ManifestStore

TokenizedDoc = Tuple[Document, List[int]]
Bin = List[TokenizedDoc]


@dataclass
class ShardBuilderConfig:
    shard_token_budget: int = 50_000
    shards_dir: str = "data/shards"
    manifests_dir: str = "data/manifests"
    token_dtype: str = "uint16"
    packing_policy: str = "greedy"  # "greedy" | "best_fit"


def _pack_greedy(tokenized_docs: List[TokenizedDoc], budget: int) -> List[Bin]:
    """Fill shards in arrival order; a document that alone exceeds the
    budget still becomes its own (larger) shard rather than being split."""
    bins: List[Bin] = []
    current: Bin = []
    current_len = 0

    for doc, ids in tokenized_docs:
        if current and current_len + len(ids) > budget:
            bins.append(current)
            current, current_len = [], 0
        current.append((doc, ids))
        current_len += len(ids)

    if current:
        bins.append(current)
    return bins


def _pack_best_fit(tokenized_docs: List[TokenizedDoc], budget: int) -> List[Bin]:
    """Best-fit-decreasing: sort longest-first, then place each document
    into the open shard with the least remaining room that still fits it.
    An oversized document (longer than budget on its own) still gets its
    own shard -- it just never has anything else placed alongside it, since
    no other document's length can be <= its (negative) remaining room."""
    ordered = sorted(tokenized_docs, key=lambda item: len(item[1]), reverse=True)

    bins: List[Bin] = []
    remaining: List[int] = []

    for doc, ids in ordered:
        length = len(ids)
        best_idx = None
        for idx, room in enumerate(remaining):
            if room >= length and (best_idx is None or room < remaining[best_idx]):
                best_idx = idx

        if best_idx is None:
            bins.append([(doc, ids)])
            remaining.append(budget - length)
        else:
            bins[best_idx].append((doc, ids))
            remaining[best_idx] -= length

    return bins


PACKING_POLICIES: dict[str, Callable[[List[TokenizedDoc], int], List[Bin]]] = {
    "greedy": _pack_greedy,
    "best_fit": _pack_best_fit,
}


def build_shards(
    documents: List[Document],
    tokenizer,
    tokenizer_hash: str,
    config: ShardBuilderConfig,
) -> List[dict]:
    if config.packing_policy not in PACKING_POLICIES:
        raise ValueError(
            f"unknown packing_policy={config.packing_policy!r}; "
            f"must be one of {sorted(PACKING_POLICIES)}"
        )
    pack = PACKING_POLICIES[config.packing_policy]

    vocab_size = tokenizer.get_vocab_size()
    dtype_max = np.iinfo(config.token_dtype).max
    if vocab_size > dtype_max:
        raise ValueError(
            f"tokenizer vocab_size={vocab_size} does not fit in {config.token_dtype} "
            f"(max {dtype_max}); pick a wider token_dtype."
        )

    eos_id = tokenizer.token_to_id("<eos>")
    if eos_id is None:
        raise ValueError("Frozen tokenizer has no <eos> token; cannot mark document boundaries.")

    shards_dir = Path(config.shards_dir)
    shards_dir.mkdir(parents=True, exist_ok=True)
    store = ManifestStore(config.manifests_dir)

    # Group by lane, preserving each lane's original corpus order -- shard
    # assignment is otherwise a pure function of (documents, config), so two
    # runs over the same input always produce identical shard_ids/hashes.
    lanes: dict[str, List[Document]] = {}
    for doc in documents:
        lanes.setdefault(doc.capability_lane, []).append(doc)

    manifests: List[dict] = []
    shard_index = 0

    for lane in sorted(lanes):
        tokenized_docs = [(doc, tokenizer.encode(doc.text).ids + [eos_id]) for doc in lanes[lane]]
        bins = pack(tokenized_docs, config.shard_token_budget)

        for shard_bin in bins:
            tokens: List[int] = []
            spans: List[dict] = []
            for doc, ids in shard_bin:
                start = len(tokens)
                tokens.extend(ids)
                spans.append(
                    {
                        "document_id": doc.document_id,
                        "source_document_id": doc.source_document_id,
                        "source_id": doc.source_id,
                        "language": doc.language,
                        "start_token": start,
                        "end_token": len(tokens),
                    }
                )
            manifests.append(
                _write_shard(
                    shard_index, lane, tokens, spans, tokenizer_hash, shards_dir,
                    config.token_dtype, config.packing_policy, store,
                )
            )
            shard_index += 1

    return manifests


def _write_shard(
    index: int,
    lane: str,
    tokens: List[int],
    spans: List[dict],
    tokenizer_hash: str,
    shards_dir: Path,
    dtype: str,
    packing_policy: str,
    store: ManifestStore,
) -> dict:
    shard_id = f"shard-{index:06d}"
    arr = np.array(tokens, dtype=dtype)
    content_hash = sha256_bytes(arr.tobytes())
    np.save(shards_dir / f"{shard_id}.npy", arr)

    manifest = {
        "shard_id": shard_id,
        "content_hash": content_hash,
        "tokenizer_hash": tokenizer_hash,
        "capability_lane": lane,
        "packing_policy": packing_policy,
        "token_count": int(arr.shape[0]),
        "document_count": len(spans),
        "document_spans": spans,
        # Not re-implemented here -- see DATALOADER_DESIGN.md scope assumptions.
        # An upstream admission pipeline would populate these for real.
        "license_tier": "unspecified",
        "dedup_status": "not_checked",
        "contamination_status": "not_checked",
        "eval_overlap_status": "none",
        "cleaning_pipeline_hash": None,
        "parent_shard_ids": [],
    }
    store.append(manifest)
    return manifest

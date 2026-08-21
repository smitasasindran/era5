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

This module also decides, per document, *how* it gets tokenized -- which
matters for `STRUCTURE_PRESERVING_LANES` (role-tagged prompt/response
records, e.g. "instruction"). For those, prompt and response text are
tokenized *separately* and their id lists concatenated, rather than
tokenizing the whole document as one string -- a BPE merge could otherwise
span the prompt/response text boundary, shifting where that boundary lands
in token space away from the exact split the Packer's structure-preserving
loss masking needs (see IMPLEMENTATION_NOTES.md §5). The resulting
per-document `response_start_token` (absolute position within the shard
where the response begins, or None for lanes with no such split) is
recorded in `document_spans` alongside `start_token`/`end_token`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np

from .corpus import Document
from .hashing import sha256_bytes
from .manifest_store import ManifestStore

TokenizedDoc = Tuple[Document, List[int]]
Bin = List[TokenizedDoc]

# Lanes whose documents are prompt/response records rather than plain
# prose/code -- packed with loss masked out over the prompt portion (see
# tds/packer.py's "structure_preserving" policy). A document in one of
# these lanes without a detectable marker degrades gracefully to "fully
# response" (response_start_token == its own start_token), rather than
# raising -- real-world instruction-style data doesn't always follow a
# clean template (see IMPLEMENTATION_NOTES.md §5 for how much of the real
# corpus's own "instruction" lane actually has any of these markers).
STRUCTURE_PRESERVING_LANES = {"instruction"}

# A handful of the most common prompt/response template conventions,
# matched as plain case-insensitive substrings -- not a general
# prompt/response boundary detector (no regex, no structural/template
# parsing, no model), just a slightly wider net than a single literal
# string. Order doesn't imply priority: when a document contains more
# than one, whichever occurs *earliest* in the text wins (see
# `_find_response_marker`), since that's the one actually separating
# prompt from response in that document.
RESPONSE_MARKERS = ("response:", "output:", "answer:", "assistant:")


def _find_response_marker(text: str) -> int:
    """The earliest index at which any of `RESPONSE_MARKERS` occurs in
    `text` (case-insensitive), or -1 if none do."""
    lowered = text.lower()
    positions = [pos for pos in (lowered.find(marker) for marker in RESPONSE_MARKERS) if pos != -1]
    return min(positions) if positions else -1


def _tokenize_document(doc: Document, tokenizer, eos_id: int) -> Tuple[List[int], Optional[int]]:
    """Returns (token_ids, response_start_offset). response_start_offset is
    the 0-based index *within this document's own tokens* where the
    response begins, or None for lanes that aren't structure-preserving."""
    if doc.capability_lane not in STRUCTURE_PRESERVING_LANES:
        return tokenizer.encode(doc.text).ids + [eos_id], None

    marker_pos = _find_response_marker(doc.text)
    if marker_pos == -1:
        prompt_text, response_text = "", doc.text
    else:
        prompt_text, response_text = doc.text[:marker_pos], doc.text[marker_pos:]

    prompt_ids = tokenizer.encode(prompt_text).ids if prompt_text else []
    response_ids = tokenizer.encode(response_text).ids
    return prompt_ids + response_ids + [eos_id], len(prompt_ids)


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
        tokenized = [(doc, *_tokenize_document(doc, tokenizer, eos_id)) for doc in lanes[lane]]
        tokenized_docs = [(doc, ids) for doc, ids, _ in tokenized]
        response_starts = {doc.document_id: resp_start for doc, _, resp_start in tokenized}
        bins = pack(tokenized_docs, config.shard_token_budget)

        for shard_bin in bins:
            tokens: List[int] = []
            spans: List[dict] = []
            for doc, ids in shard_bin:
                start = len(tokens)
                tokens.extend(ids)
                local_response_start = response_starts[doc.document_id]
                spans.append(
                    {
                        "document_id": doc.document_id,
                        "source_document_id": doc.source_document_id,
                        "source_id": doc.source_id,
                        "language": doc.language,
                        "start_token": start,
                        "end_token": len(tokens),
                        # Absolute position (within this shard) where the
                        # response portion begins, for structure-preserving
                        # lanes; None for lanes tokenized as a single blob.
                        "response_start_token": (
                            start + local_response_start if local_response_start is not None else None
                        ),
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

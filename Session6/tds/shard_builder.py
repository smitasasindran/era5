"""Documents -> immutable tokenized shards + manifests.

Each shard is lane-homogeneous (every token in it comes from one capability
lane) -- the mixture/cursor layer later depends on being able to treat "a
lane's shards" as a clean, orderable set. Documents are never split across
a shard boundary, so a document's tokens always live in exactly one
shard's document_spans, and `segment_id` at pack time can always be derived
from a single shard's manifest.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np

from .corpus import Document
from .manifest_store import ManifestStore
from .tokenizer_utils import sha256_bytes


@dataclass
class ShardBuilderConfig:
    shard_token_budget: int = 50_000
    shards_dir: str = "data/shards"
    manifests_dir: str = "data/manifests"
    token_dtype: str = "uint16"


def build_shards(
    documents: List[Document],
    tokenizer,
    tokenizer_hash: str,
    config: ShardBuilderConfig,
) -> List[dict]:
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
        buffer_tokens: List[int] = []
        buffer_spans: List[dict] = []

        for doc in lanes[lane]:
            ids = tokenizer.encode(doc.text).ids + [eos_id]

            if buffer_tokens and len(buffer_tokens) + len(ids) > config.shard_token_budget:
                manifests.append(
                    _write_shard(shard_index, lane, buffer_tokens, buffer_spans, tokenizer_hash, shards_dir, config.token_dtype, store)
                )
                shard_index += 1
                buffer_tokens, buffer_spans = [], []

            start = len(buffer_tokens)
            buffer_tokens.extend(ids)
            buffer_spans.append(
                {
                    "document_id": doc.document_id,
                    "source_document_id": doc.source_document_id,
                    "source_id": doc.source_id,
                    "language": doc.language,
                    "start_token": start,
                    "end_token": len(buffer_tokens),
                }
            )

        if buffer_tokens:
            manifests.append(
                _write_shard(shard_index, lane, buffer_tokens, buffer_spans, tokenizer_hash, shards_dir, config.token_dtype, store)
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

"""Blocks any corpus document whose content matches something in the
held-out (eval/validation/test) registry from ever reaching the shard
builder.

Checked at document-admission time, before any tokenization or shard
writing happens -- a blocked document never becomes part of a shard, so
there is no shard to retroactively un-admit. Matching is by content hash
of the raw text, not by document id or source: the whole point is to
catch a training document that happens to duplicate held-out content even
when its id/source has nothing to do with that benchmark.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

from .corpus import Document
from .eval_registry import EvalRegistry


@dataclass(frozen=True)
class BlockedEvent:
    document_id: str
    source_document_id: str
    content_hash: str
    benchmark_id: str
    version_tag: str


def filter_training_documents(
    documents: List[Document], registry: EvalRegistry
) -> Tuple[List[Document], List[BlockedEvent]]:
    """Split documents into (admitted, blocked). Callers must build shards
    (and train the tokenizer!) only from the returned `admitted` list --
    the original, unfiltered `documents` must never reach either."""
    admitted: List[Document] = []
    blocked: List[BlockedEvent] = []

    for doc in documents:
        record = registry.get(doc.content_hash)
        if record is None:
            admitted.append(doc)
        else:
            blocked.append(
                BlockedEvent(
                    document_id=doc.document_id,
                    source_document_id=doc.source_document_id,
                    content_hash=doc.content_hash,
                    benchmark_id=record["benchmark_id"],
                    version_tag=record["version_tag"],
                )
            )

    return admitted, blocked

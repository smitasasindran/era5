"""Corpus loading: parquet documents -> normalized Document records.

Per the design doc's scope assumptions, this module deliberately does not
re-implement cleaning, deduplication, license classification, or
contamination scanning -- those belong to an upstream admission pipeline
(Session 3/4) that this assignment does not rebuild. The source parquet
here (vendored at data/corpus/small_shard.parquet) is a real corpus sample
with only source/domain/language/id used as provenance; any richer
quality-score columns it happens to carry are intentionally ignored so this
loader matches what an actual admission pipeline's output would look like.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import pyarrow.parquet as pq

from .hashing import sha256_text

# Languages tagged "indic" regardless of domain -- mirrors the course's
# "indic" capability lane, which cuts across domain (a stackexchange
# question in Hindi is still an indic-lane document, not a qa-lane one).
INDIC_LANGUAGES = {"bn", "mr", "hi", "gu", "kn", "ml", "or", "ta", "pa", "ne", "as", "te", "sa"}

CORPUS_COLUMNS = ["id", "source", "domain", "language", "text"]


@dataclass(frozen=True)
class Document:
    document_id: str          # internal id, unique by construction (doc-000059)
    source_document_id: str   # the corpus's own "id" column -- kept for traceability only
    source_id: str            # the corpus's "source" column (e.g. "Starcoder")
    language: str
    capability_lane: str
    text: str

    @property
    def content_hash(self) -> str:
        """sha256 of the raw text -- what the eval firewall matches on.

        A property, not a stored field: it's a pure function of `text`, so
        computing it on access means it can never silently drift out of
        sync the way a separately-passed constructor argument could.
        """
        return sha256_text(self.text)


def capability_lane_for(domain: str, language: str) -> str:
    """Deterministic domain+language -> capability lane mapping.

    Language wins over domain for indic content: a stackexchange thread in
    Hindi is scarce indic-lane data, not just more qa data, and should be
    protectable as such later by the mixture compiler.
    """
    if language in INDIC_LANGUAGES:
        return "indic"
    if domain == "code":
        return "code"
    if domain in ("math", "science"):
        return "math_science"
    if domain == "qa":
        return "qa"
    if domain == "instruction":
        return "instruction"
    return "general_web"


def load_corpus(parquet_path: str | Path, limit: Optional[int] = None) -> List[Document]:
    """Load and normalize the corpus in stable, deterministic row order."""
    table = pq.read_table(str(parquet_path), columns=CORPUS_COLUMNS)
    df = table.to_pandas()
    if limit is not None:
        df = df.iloc[:limit]

    documents: List[Document] = []
    for row_idx, row in df.iterrows():
        text = (row["text"] or "").strip()
        if not text:
            continue
        documents.append(
            Document(
                document_id=f"doc-{row_idx:06d}",
                source_document_id=str(row["id"]),
                source_id=str(row["source"]),
                language=str(row["language"]),
                capability_lane=capability_lane_for(str(row["domain"]), str(row["language"])),
                text=text,
            )
        )
    return documents

"""Held-out (eval/validation/test) content registry.

Deliberately separate from the training manifest store: this tracks
content that must never become loss-bearing training data, keyed by the
exact content hash of the held-out text -- not by any document id or
source, since the whole point of the firewall this feeds is to catch
content overlap even when a training document's id/source has nothing to
do with the benchmark it happens to duplicate.

Unlike the tokenizer or shards, this registry is not tied to one corpus
build and is not cleared between pipeline runs -- a held-out benchmark
registry is meant to accumulate fingerprints across benchmarks (and
corpora) over the life of a project, matching the design doc's "when a
benchmark is updated, its hashes are added to the registry."
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

from .hashing import sha256_text


class EvalRegistryError(Exception):
    """Raised when a content hash is claimed by two different benchmarks."""


class EvalRegistry:
    def __init__(self, registry_dir: str | Path):
        self.dir = Path(registry_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.dir / "index.jsonl"

        self._by_hash: dict = {}
        if self.index_path.exists():
            with open(self.index_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    self._by_hash[record["content_hash"]] = record

    def register_text(
        self, text: str, benchmark_id: str, version_tag: str, never_train: bool = True
    ) -> dict:
        record = {
            "content_hash": sha256_text(text),
            "benchmark_id": benchmark_id,
            "version_tag": version_tag,
            "never_train": never_train,
        }
        self._append(record)
        return record

    def _append(self, record: dict) -> None:
        existing = self._by_hash.get(record["content_hash"])
        if existing is not None:
            if existing["benchmark_id"] != record["benchmark_id"]:
                raise EvalRegistryError(
                    f"content_hash {record['content_hash']} is already registered under "
                    f"benchmark {existing['benchmark_id']!r}; refusing to also register it "
                    f"under {record['benchmark_id']!r}."
                )
            return  # idempotent re-registration of the same benchmark entry

        with open(self.index_path, "a") as f:
            f.write(json.dumps(record) + "\n")
        self._by_hash[record["content_hash"]] = record

    def get(self, content_hash: str) -> Optional[dict]:
        return self._by_hash.get(content_hash)

    def all(self) -> List[dict]:
        return list(self._by_hash.values())

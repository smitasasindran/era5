"""Append-only registry of shard manifests.

`index.jsonl` is the manifest store's actual log: one line per shard,
written once, in creation order, never rewritten. Per-shard `.json` files
alongside it are just a convenience for looking up a single shard without
scanning the whole index.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class ManifestStoreError(Exception):
    """Raised when a caller tries to mutate an already-registered shard."""


class ManifestStore:
    def __init__(self, manifests_dir: str | Path):
        self.dir = Path(manifests_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.dir / "index.jsonl"

        self._by_id: dict = {}
        if self.index_path.exists():
            with open(self.index_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    manifest = json.loads(line)
                    self._by_id[manifest["shard_id"]] = manifest

    def get(self, shard_id: str) -> Optional[dict]:
        return self._by_id.get(shard_id)

    def all(self) -> List[dict]:
        return list(self._by_id.values())

    def lane_token_totals(self) -> Dict[str, int]:
        """Sum of token_count across all registered shards, grouped by
        capability_lane -- the "how much distinct supply actually exists"
        input the mixture compiler checks its planned shares against."""
        totals: Dict[str, int] = defaultdict(int)
        for manifest in self._by_id.values():
            totals[manifest["capability_lane"]] += manifest["token_count"]
        return dict(totals)

    def document_pool_by_lane(self) -> Dict[str, List[Tuple[str, str, int]]]:
        """Every document, grouped by capability_lane, as (shard_id,
        document_id, start_token) triples -- the raw material the Cursor
        draws from and permutes deterministically. Sorted by
        (shard_id, start_token) before any shuffling, so the *unshuffled*
        base order is itself reproducible rather than depending on dict or
        filesystem iteration order."""
        pools: Dict[str, List[Tuple[str, str, int]]] = defaultdict(list)
        for shard_id in sorted(self._by_id):
            manifest = self._by_id[shard_id]
            lane = manifest["capability_lane"]
            for span in sorted(manifest["document_spans"], key=lambda s: s["start_token"]):
                pools[lane].append((shard_id, span["document_id"], span["start_token"]))
        return dict(pools)

    def append(self, manifest: dict) -> None:
        """Register a shard manifest.

        Idempotent for an unchanged re-run (same shard_id + same
        content_hash is a no-op). Raises if a shard_id is reused with a
        *different* content_hash -- that would mean a shard mutated in
        place, which the design treats as impossible: a changed shard is a
        new shard with a new id and hash, never an edit.
        """
        shard_id = manifest["shard_id"]
        existing = self._by_id.get(shard_id)

        if existing is not None:
            if existing["content_hash"] != manifest["content_hash"]:
                raise ManifestStoreError(
                    f"Refusing to register {shard_id}: existing content_hash="
                    f"{existing['content_hash']} != new {manifest['content_hash']}. "
                    "Shards are immutable -- a changed shard must get a new shard_id."
                )
            return

        with open(self.index_path, "a") as f:
            f.write(json.dumps(manifest) + "\n")
        with open(self.dir / f"{shard_id}.json", "w") as f:
            json.dump(manifest, f, indent=2)

        self._by_id[shard_id] = manifest

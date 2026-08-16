"""Shared sha256 hashing helpers.

Used across the pipeline for shard content hashes, tokenizer hashes, and
(later) loss-mask hashes -- kept in one place since the design treats
"hash of X" as a recurring, load-bearing concept tying independently
recomputed data back together, not a one-off utility of any single
component.
"""

from __future__ import annotations

import hashlib
from pathlib import Path


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))

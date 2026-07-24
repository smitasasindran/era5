#!/usr/bin/env python3
"""Smoke test for pipeline.writers.open_writer().

Run directly:
    python tests/smoke_test_writers.py
"""

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.writers import open_writer  # noqa: E402

checks_run = 0
checks_failed = 0


def check(label, condition):
    global checks_run, checks_failed
    checks_run += 1
    status = "PASS" if condition else "FAIL"
    if not condition:
        checks_failed += 1
    print(f"[{status}] {label}")


RECORDS = [
    {"id": "0", "text": "hello", "raw_len": 5, "clean_len": 5, "ghost_tokens": {}},
    {"id": "1", "text": "world", "raw_len": 5, "clean_len": 5, "ghost_tokens": {"pipe_angle": 1}},
    {"id": "2", "text": "nested", "raw_len": 6, "clean_len": 6, "ghost_tokens": {"bracket_upper": 2, "angle_bare": 1}},
]

with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)

    # JSONL round trip
    jsonl_path = tmp / "out.jsonl"
    with open_writer(jsonl_path) as w:
        for r in RECORDS:
            w.write(r)
    lines = jsonl_path.read_text(encoding="utf-8").strip().split("\n")
    read_back = [json.loads(ln) for ln in lines]
    check("jsonl: correct row count", len(read_back) == len(RECORDS))
    check("jsonl: ghost_tokens stays a nested dict", read_back[2]["ghost_tokens"] == {"bracket_upper": 2, "angle_bare": 1})
    check("jsonl: content matches", read_back[0]["text"] == "hello")

    # Parquet round trip, forcing multiple small batches to exercise the
    # incremental ParquetWriter flush path (batch_size=1 -> 3 flushes).
    parquet_path = tmp / "out.parquet"
    try:
        import pyarrow.parquet as pq

        with open_writer(parquet_path, batch_size=1) as w:
            for r in RECORDS:
                w.write(r)
        table = pq.read_table(parquet_path)
        df = table.to_pandas()
        check("parquet: correct row count", len(df) == len(RECORDS))
        check("parquet: content matches", df.iloc[0]["text"] == "hello")
        check(
            "parquet: ghost_tokens flattened to a JSON string (stable scalar schema)",
            isinstance(df.iloc[2]["ghost_tokens"], str) and json.loads(df.iloc[2]["ghost_tokens"]) == {"bracket_upper": 2, "angle_bare": 1},
        )
    except ImportError:
        print("[SKIP] pyarrow not installed -- parquet checks skipped")

    # Extension with no special-casing (e.g. .txt) falls back to JSONL.
    other_path = tmp / "out.txt"
    with open_writer(other_path) as w:
        w.write(RECORDS[0])
    check("unrecognized extension falls back to JsonlWriter", json.loads(other_path.read_text().strip())["id"] == "0")

print(f"\n{checks_run - checks_failed}/{checks_run} checks passed.")
if checks_failed:
    sys.exit(1)

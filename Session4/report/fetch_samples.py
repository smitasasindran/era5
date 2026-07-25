#!/usr/bin/env python3
"""Pull a handful of RAW documents from a source, for the report's "what does
the raw data look like" panel.

Reuses pipeline.io_utils.load_documents, so this reads text exactly the same
way run_extract.py's --source/--text-field(s) do -- the only difference is
this stops after N docs and doesn't run Extraction or Normalize+Clean, so
what you get is genuinely pre-pipeline raw text.

Usage
-----
    python report/fetch_samples.py --source hf --dataset allenai/c4 \\
        --config realnewslike --split train --streaming \\
        --text-field text -n 8 --max-chars 800 \\
        --output report/profiles/c4_realnewslike.samples.json

    python report/fetch_samples.py --source local \\
        --path /path/to/raw_shard.parquet --text-field text \\
        -n 8 --max-chars 800 --output report/profiles/capstone.samples.json
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.io_utils import load_documents  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", choices=["hf", "hf_disk", "local"], required=True)
    p.add_argument("--dataset", help="HF dataset repo id")
    p.add_argument("--config", default=None)
    p.add_argument("--split", default="train")
    p.add_argument("--streaming", action="store_true")
    p.add_argument("--path", help="Local file path, or an on-disk HF dataset directory")
    p.add_argument("--txt-mode", choices=["lines", "whole"], default="lines")
    p.add_argument("--text-field", default="text")
    p.add_argument("--text-fields", default=None)
    p.add_argument("-n", "--num-samples", type=int, default=8)
    p.add_argument("--max-chars", type=int, default=800, help="Truncate each sample to this many characters")
    p.add_argument("--output", required=True)
    return p.parse_args()


def main():
    args = parse_args()
    args.limit = args.num_samples  # load_documents reads args.limit

    samples = []
    for doc in load_documents(args):
        text = doc.get("text") or ""
        truncated = len(text) > args.max_chars
        samples.append({
            "id": doc.get("id"),
            "text": text[:args.max_chars],
            "truncated": truncated,
            "full_len": len(text),
        })

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(samples, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(samples)} samples to {out_path}")


if __name__ == "__main__":
    main()

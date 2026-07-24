#!/usr/bin/env python3
"""Stage 0 of the pipeline: Extraction.

Turns raw HTML into clean prose before anything else runs: pulls the
article body and drops nav/boilerplate/cookie banners/footers around it.
Chain the output straight into run_normalize.py as its --source local input.

Output format follows the --output extension: .parquet writes Parquet,
anything else writes JSON Lines. Match it to your --source format to keep
the whole pipeline in one format end to end.

Examples
--------
Stream a raw HF dataset, then normalize the result:
    python run_extract.py --source hf --dataset some/raw-html-corpus \\
        --split train --text-field text --output out/step0_extracted.jsonl

    python run_normalize.py --source local --path out/step0_extracted.jsonl \\
        --text-field text --output out/step1_clean.jsonl

Extract a local parquet file, kept as parquet all the way through:
    python run_extract.py --source local --path ./data/raw.parquet \\
        --text-field text --output out/step0_extracted.parquet
"""

import argparse
import json
from collections import Counter
from pathlib import Path

from pipeline.extract import extract_content
from pipeline.io_utils import load_documents
from pipeline.writers import open_writer


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--source", choices=["hf", "hf_disk", "local"], required=True)

    p.add_argument("--dataset", help="HF dataset repo id")
    p.add_argument("--config", default=None, help="HF dataset config/subset name")
    p.add_argument("--split", default="train")
    p.add_argument(
        "--streaming", action="store_true", help="Stream from the Hub instead of downloading fully"
    )

    p.add_argument("--path", help="Local file path, or an on-disk HF dataset directory")
    p.add_argument(
        "--txt-mode",
        choices=["lines", "whole"],
        default="lines",
        help="For .txt local files: one doc per line, or the whole file as one doc",
    )

    p.add_argument("--text-field", default="text", help="Field/column holding the raw document text")
    p.add_argument("--limit", type=int, default=None, help="Stop after N documents (omit for all)")
    p.add_argument(
        "--output", required=True,
        help="Output path for extracted documents. Extension picks the format: .parquet or .jsonl",
    )
    p.add_argument("--report", default=None, help="Optional JSON path for the aggregate report")
    return p.parse_args()


def main():
    args = parse_args()
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    path_counts = Counter()
    n_docs = 0
    total_raw_chars = 0
    total_extracted_chars = 0

    with open_writer(out_path) as writer:
        for doc in load_documents(args):
            raw_text = doc.get("text") or ""
            extracted, meta = extract_content(raw_text)

            record = {
                "id": doc.get("id"),
                "text": extracted,
                "extraction_path": meta["path"],
                "raw_len": len(raw_text),
                "extracted_len": len(extracted),
            }
            writer.write(record)

            n_docs += 1
            path_counts[meta["path"]] += 1
            total_raw_chars += len(raw_text)
            total_extracted_chars += len(extracted)

    report = {
        "n_docs": n_docs,
        "extraction_path_distribution": dict(path_counts),
        "total_raw_chars": total_raw_chars,
        "total_extracted_chars": total_extracted_chars,
        "chars_removed": total_raw_chars - total_extracted_chars,
    }

    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    used_fallback = ("bs4" in path_counts) or ("html_fallback_stripped" in path_counts)
    if used_fallback and "trafilatura" not in path_counts:
        print(
            "\nNOTE: full HTML documents were found but `trafilatura` isn't installed, so "
            "extraction fell back to BeautifulSoup/regex stripping. `pip install trafilatura` "
            "for better main-content extraction quality on those documents."
        )


if __name__ == "__main__":
    main()

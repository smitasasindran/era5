#!/usr/bin/env python3
"""Diagnose WHY specific documents took a non-passthrough Extraction path.

Useful whenever a dataset shows surprising extraction_path_distribution
numbers (e.g. a code dataset triggering the HTML tiers) -- streams the
source, runs extract_content() on each doc, and for the first N documents
that took a path you're asking about, prints the raw/extracted texts around
their first point of divergence so you can see exactly what changed and why.

Usage
-----
    python diagnose_extraction.py --source hf \\
        --dataset hasankursun/github-code-2025-language-split \\
        --config python --split train --streaming --limit 5000 \\
        --text-field content --paths bs4,trafilatura,inline_tags_stripped \\
        --max-examples 3
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pipeline.extract import extract_content, looks_like_html_document  # noqa: E402
from pipeline.io_utils import load_documents  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", choices=["hf", "hf_disk", "local"], required=True)
    p.add_argument("--dataset")
    p.add_argument("--config", default=None)
    p.add_argument("--split", default="train")
    p.add_argument("--streaming", action="store_true")
    p.add_argument("--path")
    p.add_argument("--txt-mode", choices=["lines", "whole"], default="lines")
    p.add_argument("--text-field", default="text")
    p.add_argument("--text-fields", default=None)
    p.add_argument("--limit", type=int, default=5000, help="How many docs to scan")
    p.add_argument(
        "--paths", default="bs4,trafilatura,inline_tags_stripped,html_fallback_stripped",
        help="Comma-separated extraction_path values to look for (default: anything but passthrough)",
    )
    p.add_argument("--max-examples", type=int, default=3, help="Stop after this many matches")
    p.add_argument("--context", type=int, default=250, help="Chars of context around the first divergence")
    p.add_argument(
        "--content-hint", choices=["auto", "html", "not-html"], default="auto",
        help="Same meaning as run_extract.py's --content-hint",
    )
    return p.parse_args()


def first_divergence(a, b):
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n if len(a) != len(b) else None


def main():
    args = parse_args()
    target_paths = set(p.strip() for p in args.paths.split(",") if p.strip())
    is_html_hint = {"auto": None, "html": True, "not-html": False}[args.content_hint]

    found = 0
    scanned = 0
    for doc in load_documents(args):
        scanned += 1
        raw = doc.get("text") or ""
        extracted, meta = extract_content(raw, is_html_hint=is_html_hint)
        if meta["path"] not in target_paths:
            continue

        found += 1
        idx = first_divergence(raw, extracted)
        print(f"\n===== match {found}: id={doc.get('id')} path={meta['path']} =====")
        print(f"raw_len={len(raw)} extracted_len={len(extracted)} looks_like_html_document={looks_like_html_document(raw)}")
        if idx is None:
            print("(raw and extracted are identical over their common length -- one is a prefix of the other)")
            idx = min(len(raw), len(extracted))
        lo = max(0, idx - args.context)
        print(f"--- raw[{lo}:{idx + args.context}] ---")
        print(repr(raw[lo:idx + args.context]))
        print(f"--- extracted[{lo}:{idx + args.context}] ---")
        print(repr(extracted[lo:idx + args.context]))

        if found >= args.max_examples:
            break

    print(f"\nScanned {scanned} docs, found {found} matching path(s) in {target_paths}.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Step 1 of the pipeline: Normalize + Clean.

Reads documents from HuggingFace (streamed or fully downloaded) or from a
local file, runs pipeline.normalize.clean_text() on each one, and writes a
cleaned file alongside a report of what got flagged (ghost tokens, content
type mix, size reduction, noise-character breakdown).

Output format follows the --output extension: .parquet writes Parquet,
anything else writes JSON Lines. Point --output at a .parquet file when your
--source is parquet, to keep the whole pipeline in one format end to end.

Examples
--------
Stream directly from the HF Hub (no full download):
    python run_normalize.py --source hf --dataset ai4bharat/sangraha \\
        --config hin --split train --streaming --limit 2000 \\
        --text-field text --output out/step1_clean.jsonl

A dataset already downloaded with datasets.save_to_disk():
    python run_normalize.py --source hf_disk --path ./my_local_dataset \\
        --split train --text-field text --output out/step1_clean.jsonl

A local parquet file, kept as parquet all the way through:
    python run_normalize.py --source local --path ./data/raw.parquet \\
        --text-field text --output out/step1_clean.parquet
"""

import argparse
import json
from collections import Counter
from pathlib import Path

from pipeline.io_utils import load_documents
from pipeline.normalize import analyze_noise, clean_text, content_hash, guess_content_type
from pipeline.writers import open_writer


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--source", choices=["hf", "hf_disk", "local"], required=True)

    p.add_argument("--dataset", help="HF dataset repo id, e.g. ai4bharat/sangraha")
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
    p.add_argument("--content-type", choices=["prose", "code", "auto"], default="auto")
    p.add_argument("--limit", type=int, default=None, help="Stop after N documents (omit for all)")
    p.add_argument(
        "--output", required=True,
        help="Output path for cleaned documents. Extension picks the format: .parquet or .jsonl",
    )
    p.add_argument("--report", default=None, help="Optional JSON path for the aggregate report")
    return p.parse_args()


def main():
    args = parse_args()
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ghost_counts = Counter()
    ghost_examples = {}
    content_type_counts = Counter()
    n_docs = 0
    n_empty_after_clean = 0
    total_raw_chars = 0
    total_clean_chars = 0
    total_garbage_control_chars = 0
    total_zero_width_noise_chars = 0
    total_broken_utf_replacement_chars = 0
    total_zwnj_kept = 0
    total_zwj_kept = 0
    n_docs_with_broken_utf = 0

    with open_writer(out_path) as writer:
        for doc in load_documents(args):
            raw_text = doc.get("text") or ""
            ctype = args.content_type
            if ctype == "auto":
                ctype = guess_content_type(raw_text)

            cleaned, ghosts = clean_text(raw_text, content_type=ctype)
            noise = analyze_noise(raw_text, cleaned)

            for pattern_name, matches in ghosts.items():
                ghost_counts[pattern_name] += len(matches)
                if pattern_name not in ghost_examples:
                    ghost_examples[pattern_name] = matches[:5]

            record = {
                "id": doc.get("id"),
                "text": cleaned,
                "sha1": content_hash(cleaned),
                "content_type": ctype,
                "raw_len": len(raw_text),
                "clean_len": len(cleaned),
                "ghost_tokens": {k: len(v) for k, v in ghosts.items() if v},
                **noise,
            }
            writer.write(record)

            n_docs += 1
            content_type_counts[ctype] += 1
            total_raw_chars += len(raw_text)
            total_clean_chars += len(cleaned)
            total_garbage_control_chars += noise["garbage_control_chars"]
            total_zero_width_noise_chars += noise["zero_width_noise_chars"]
            total_broken_utf_replacement_chars += noise["broken_utf_replacement_chars"]
            total_zwnj_kept += noise["zwnj_kept"]
            total_zwj_kept += noise["zwj_kept"]
            if noise["broken_utf_replacement_chars"] > 0:
                n_docs_with_broken_utf += 1
            if not cleaned.strip():
                n_empty_after_clean += 1

    report = {
        "n_docs": n_docs,
        "content_type_distribution": dict(content_type_counts),
        "total_raw_chars": total_raw_chars,
        "total_clean_chars": total_clean_chars,
        "chars_removed": total_raw_chars - total_clean_chars,
        "n_empty_after_clean": n_empty_after_clean,
        "ghost_token_counts": dict(ghost_counts),
        "ghost_token_examples": ghost_examples,
        "noise_stats": {
            "garbage_control_chars": total_garbage_control_chars,
            "zero_width_noise_chars": total_zero_width_noise_chars,
            "broken_utf_replacement_chars": total_broken_utf_replacement_chars,
            "n_docs_with_broken_utf": n_docs_with_broken_utf,
            "zwnj_kept": total_zwnj_kept,
            "zwj_kept": total_zwj_kept,
        },
    }

    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    if ghost_counts:
        print(
            "\nNOTE: ghost special tokens were found in this corpus (see "
            "'ghost_token_examples' above). Decide which of these should become "
            "real registered special tokens BEFORE building the tokenizer."
        )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Show real context around ghost-token hits in a Normalize+Clean output file.

Useful whenever a dataset's ghost_token_counts look surprising -- e.g. a code
dataset legitimately containing many literal "<|endoftext|>"/"<output>"
strings (because the code itself manipulates tokenizer/prompt-template
tokens) versus a prose dataset where the same hit would signal contamination
or a chat-template leak. Reads a run_normalize.py output file (.parquet or
.jsonl) and prints N examples per pattern with surrounding text.

Usage
-----
    python diagnose_ghost_tokens.py --path out/step1_githubcode_py_clean.parquet \\
        --pattern pipe_angle --max-examples 3 --context 150

    # all patterns found in the file, 2 examples each:
    python diagnose_ghost_tokens.py --path out/step1_githubcode_py_clean.parquet
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pipeline.normalize import GHOST_TOKEN_PATTERNS  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--path", required=True, help="A run_normalize.py output file (.parquet or .jsonl)")
    p.add_argument("--pattern", default=None, help="Only this ghost-token pattern name (default: all found)")
    p.add_argument("--max-examples", type=int, default=2, help="Examples to show per pattern")
    p.add_argument("--context", type=int, default=150, help="Chars of context around each match")
    return p.parse_args()


def iter_records(path):
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        import pandas as pd

        df = pd.read_parquet(path)
        for _, row in df.iterrows():
            gt = row.get("ghost_tokens")
            gt = json.loads(gt) if isinstance(gt, str) else (gt or {})
            yield {"id": row.get("id"), "text": row.get("text") or "", "ghost_tokens": gt}
    else:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                yield {"id": obj.get("id"), "text": obj.get("text") or "", "ghost_tokens": obj.get("ghost_tokens") or {}}


def main():
    args = parse_args()
    target_patterns = {args.pattern} if args.pattern else None
    shown = {}

    for record in iter_records(args.path):
        for pattern_name in record["ghost_tokens"]:
            if target_patterns and pattern_name not in target_patterns:
                continue
            if pattern_name not in GHOST_TOKEN_PATTERNS:
                continue
            if shown.get(pattern_name, 0) >= args.max_examples:
                continue

            m = GHOST_TOKEN_PATTERNS[pattern_name].search(record["text"])
            if not m:
                continue

            shown[pattern_name] = shown.get(pattern_name, 0) + 1
            lo = max(0, m.start() - args.context)
            hi = m.end() + args.context
            print(f"\n===== {pattern_name} example {shown[pattern_name]}: id={record['id']} match={m.group(0)!r} =====")
            print(repr(record["text"][lo:hi]))

    if not shown:
        print("No ghost-token matches found (check --pattern spelling, or the file has none).")
    else:
        print("\nShown counts:", shown)


if __name__ == "__main__":
    main()

"""
Generates a reference file of {text, tokens, ids} using the REAL Python
`tokenizers` library for a given tokenizer.json, so compare_with_python.js
can check the JS port produces byte-for-byte identical output.

Usage:
    python test/generate_reference.py <tokenizer.json> <output.json>

Re-run this (and compare_with_python.js) whenever tokenizer.js changes, or
whenever a new tokenizer version is added to models/, to re-verify the JS
port is still faithful to the real tokenizer.
"""
import json
import sys
from tokenizers import Tokenizer

TEST_STRINGS_PATH = "test/test_strings.json"


def main(tokenizer_path, out_path):
    tok = Tokenizer.from_file(tokenizer_path)
    with open(TEST_STRINGS_PATH, encoding="utf-8") as f:
        test_strings = json.load(f)

    results = []
    for s in test_strings:
        enc = tok.encode(s)
        results.append({"text": s, "tokens": enc.tokens, "ids": enc.ids})

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Wrote {len(results)} cases to {out_path}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python test/generate_reference.py <tokenizer.json> <output.json>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])

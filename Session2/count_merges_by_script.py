"""
Classify every merge rule in a trained tokenizer by Unicode script: Latin,
Devanagari, Telugu, or Mixed-script (the two merged tokens span more than one
real script). Characters with no script (digits, punctuation) don't count
towards "mixed" on their own -- a merge is only Mixed-script if it combines
characters from two or more *actual* scripts.

Useful as a direct check on merge_tokenizers.py's core safety claim: since it
builds the final vocab from 3 script-separated groups (English/Latin,
Telugu, Hindi+Marathi/Devanagari), the resulting merge table should contain
~zero Mixed-script merges.
"""
import json
import sys
import unicodedata
from collections import Counter


def char_script(ch):
    if not ch.isalpha():
        return "Common"
    name = unicodedata.name(ch, "")
    if "DEVANAGARI" in name:
        return "Devanagari"
    if "TELUGU" in name:
        return "Telugu"
    if "LATIN" in name:
        return "Latin"
    return "Other"


def token_scripts(tok_str):
    return {s for ch in tok_str if (s := char_script(ch)) != "Common"}


def classify_merge(a, b):
    scripts = token_scripts(a) | token_scripts(b)
    if not scripts:
        return "Common/punctuation"
    if len(scripts) == 1:
        return next(iter(scripts))
    return "Mixed-script"


def main(path):
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    merges = d["model"]["merges"]

    counts = Counter()
    mixed_examples = []
    for pair in merges:
        a, b = pair
        cat = classify_merge(a, b)
        counts[cat] += 1
        if cat == "Mixed-script" and len(mixed_examples) < 20:
            mixed_examples.append((a, b))

    total = sum(counts.values())
    print(f"Tokenizer: {path}")
    print(f"Total merges: {total}\n")
    for cat in ["Latin", "Devanagari", "Telugu", "Mixed-script", "Common/punctuation", "Other"]:
        if counts.get(cat):
            pct = 100 * counts[cat] / total
            print(f"  {cat:20s} {counts[cat]:5d}  ({pct:.2f}%)")

    if mixed_examples:
        print("\nSample Mixed-script merges:")
        for a, b in mixed_examples:
            print(f"  {a!r} + {b!r} -> {(a + b)!r}")
    else:
        print("\nNo Mixed-script merges found.")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "tokenizer_final.json"
    main(path)

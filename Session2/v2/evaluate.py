"""
Evaluate a trained v2 tokenizer: faithful-unit fertility per language,
computed over each language's full markdown corpus (occurrence-level, not
per-unique-word like v1 -- see faithful_units.py's docstring for why).
"""
import json
import sys
from tokenizers import Tokenizer
from faithful_units import count_faithful_units
from train_utils import read_corpus

LANGS = ["en", "hi", "te", "mr"]
LANG_NAME = {"en": "English", "hi": "Hindi", "te": "Telugu", "mr": "Marathi"}


def evaluate(tokenizer_path):
    tok = Tokenizer.from_file(tokenizer_path)
    print(f"Tokenizer: {tokenizer_path}  vocab_size={tok.get_vocab_size()}")

    results = {}
    for l in LANGS:
        text = read_corpus(l)
        total_tokens = len(tok.encode(text).tokens)
        units = count_faithful_units(text)
        ratio = total_tokens / units
        results[l] = {"tokens": total_tokens, "faithful_units": units, "ratio": ratio}
        print(f"{l}: tokens={total_tokens:6d}  faithful_units={units:6d}  fertility={ratio:.4f}")

    sorted_langs = sorted(results.items(), key=lambda kv: kv[1]["ratio"], reverse=True)
    print("\nSorted (largest to smallest fertility):")
    for l, r in sorted_langs:
        print(f"  {LANG_NAME[l]}: {r['ratio']:.4f}")

    ratios = [r["ratio"] for r in results.values()]
    spread = max(ratios) - min(ratios)
    score = 1000 / spread if spread > 0 else None

    x1 = results["en"]["ratio"]
    x1_pass = x1 < 1.2
    print(f"\nX1 (English) = {x1:.4f} < 1.2 ? {'PASS' if x1_pass else 'FAIL'}")
    print(f"Spread = {spread:.4f}   Score = 1000/spread ≈ {score:.1f}" if score else f"Spread = {spread:.4f}")

    summary = {
        "tokenizer_path": tokenizer_path,
        "vocab_size": tok.get_vocab_size(),
        "metric": "faithful_unit fertility (tokens / faithful units, over full corpus)",
        "languages": {
            l: {
                "name": LANG_NAME[l],
                "tokens": results[l]["tokens"],
                "faithful_units": results[l]["faithful_units"],
                "ratio": results[l]["ratio"],
            }
            for l in LANGS
        },
        "sorted_largest_to_smallest": [l for l, _ in sorted_langs],
        "spread": spread,
        "score": score,
        "x1_constraint_pass": x1_pass,
    }
    return results, summary


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "tokenizer_final.json"
    out_json = sys.argv[2] if len(sys.argv) > 2 else "../tokenizer-playground/models/v2/eval_results.json"

    _, summary = evaluate(path)

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {out_json}")

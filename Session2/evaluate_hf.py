"""
Evaluate a trained HF `tokenizers` BPE tokenizer: fertility (tokens per unique
whitespace-split word) per language, matching the naive-BPE evaluation metric.
"""
import json
import sys
from tokenizers import Tokenizer

LANGS = ["en", "hi", "te", "mr"]
LANG_LABEL = {"en": "X1", "hi": "X2", "te": "X3", "mr": "X4"}
LANG_NAME = {"en": "English", "hi": "Hindi", "te": "Telugu", "mr": "Marathi"}
NAMES = {l: f"{LANG_LABEL[l]} ({LANG_NAME[l]})" for l in LANGS}


def unique_words(lang, tok):
    """Words = unique pre-tokenizer atoms (matches the tokenizer's own word/punct
    boundaries), not a raw whitespace split -- otherwise punctuation stuck to a
    word (e.g. "India,") inflates the token count for reasons unrelated to
    subword fragmentation."""
    with open(f"corpus/india_{lang}.txt", encoding="utf-8") as f:
        text = f.read()
    if tok.normalizer:
        text = tok.normalizer.normalize_str(text)
    pretoks = tok.pre_tokenizer.pre_tokenize_str(text)
    return sorted(set(p for p, _ in pretoks))


def evaluate(tokenizer_path):
    tok = Tokenizer.from_file(tokenizer_path)
    print(f"Tokenizer: {tokenizer_path}  vocab_size={tok.get_vocab_size()}")

    results = {}
    for l in LANGS:
        words = unique_words(l, tok)
        encodings = tok.encode_batch(words)
        total_tokens = sum(len(e.tokens) for e in encodings)
        n_words = len(words)
        ratio = total_tokens / n_words
        results[l] = {"vocab_words": n_words, "tokens": total_tokens, "ratio": ratio}
        print(f"{l}: unique_words={n_words:5d}  tokens={total_tokens:6d}  ratio={ratio:.4f}")

    sorted_langs = sorted(results.items(), key=lambda kv: kv[1]["ratio"], reverse=True)
    print("\nSorted (largest to smallest ratio):")
    for l, r in sorted_langs:
        print(f"  {NAMES[l]}: {r['ratio']:.4f}")

    ratios = [r["ratio"] for r in results.values()]
    spread = max(ratios) - min(ratios)
    score = 1000 / spread if spread > 0 else None

    x1 = results["en"]["ratio"]
    x1_pass = x1 <= 1.2
    print(f"\nConstraint check: X1 (English) = {x1:.4f} <= 1.2 ? {'PASS' if x1_pass else 'FAIL'}")
    print(f"Spread = {spread:.4f}   Score = 1000/spread ≈ {score:.1f}" if score else f"Spread = {spread:.4f}")

    summary = {
        "tokenizer_path": tokenizer_path,
        "vocab_size": tok.get_vocab_size(),
        "languages": {
            l: {
                "label": LANG_LABEL[l],
                "name": LANG_NAME[l],
                "unique_words": results[l]["vocab_words"],
                "tokens": results[l]["tokens"],
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
    # Writes straight into tokenizer-playground/ by default so the page's
    # "Official evaluation" section is refreshed with no manual copy step.
    out_json = sys.argv[2] if len(sys.argv) > 2 else "tokenizer-playground/eval_results.json"

    _, summary = evaluate(path)

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {out_json}")

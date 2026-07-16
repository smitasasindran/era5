"""
Evaluate a trained HF `tokenizers` BPE tokenizer: fertility (tokens per unique
whitespace-split word) per language, matching the naive-BPE evaluation metric.
"""
import sys
from tokenizers import Tokenizer

LANGS = ["en", "hi", "te", "gu"]
NAMES = {"en": "X1 (English)", "hi": "X2 (Hindi)", "te": "X3 (Telugu)", "gu": "X4 (Gujarati)"}


def unique_words(lang, tok):
    """Words = unique pre-tokenizer atoms (matches the tokenizer's own word/punct
    boundaries), not a raw whitespace split -- otherwise punctuation stuck to a
    word (e.g. "India,") inflates the token count for reasons unrelated to
    subword fragmentation."""
    with open(f"corpus/india_{lang}.txt", encoding="utf-8") as f:
        text = f.read()
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

    x1 = results["en"]["ratio"]
    print(f"\nConstraint check: X1 (English) = {x1:.4f} <= 1.2 ? {'PASS' if x1 <= 1.2 else 'FAIL'}")
    return results


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "tokenizer_hf.json"
    evaluate(path)

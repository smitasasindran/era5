import time
from collections import Counter
from naive_bpe import BPETokenizer, word_freqs_from_text

LANGS = ["en", "hi", "te", "gu"]
VOCAB_SIZE = 10000


def load_corpora():
    texts = {}
    for l in LANGS:
        with open(f"corpus/india_{l}.txt", encoding="utf-8") as f:
            texts[l] = f.read()
    return texts


def main():
    texts = load_corpora()
    per_lang_freqs = {l: word_freqs_from_text(t) for l, t in texts.items()}

    combined = Counter()
    for l in LANGS:
        combined.update(per_lang_freqs[l])

    print(f"Combined unique words across all 4 languages: {len(combined)}")

    tok = BPETokenizer()
    t0 = time.time()
    tok.train(combined, VOCAB_SIZE, verbose=True)
    print(f"Training took {time.time() - t0:.1f}s")

    print("\n=== Per-language fertility (tokens per unique word) ===")
    results = {}
    for l in LANGS:
        vocab_words = list(per_lang_freqs[l].keys())
        total_tokens = sum(len(tok.encode_word(w)) for w in vocab_words)
        n_words = len(vocab_words)
        ratio = total_tokens / n_words
        results[l] = {"vocab_words": n_words, "tokens": total_tokens, "ratio": ratio}
        print(f"{l}: unique_words={n_words:5d}  tokens={total_tokens:6d}  ratio={ratio:.4f}")

    sorted_langs = sorted(results.items(), key=lambda kv: kv[1]["ratio"], reverse=True)
    print("\nSorted (largest to smallest ratio):")
    names = {"en": "X1 (English)", "hi": "X2 (Hindi)", "te": "X3 (Telugu)", "gu": "X4 (Gujarati)"}
    for l, r in sorted_langs:
        print(f"  {names[l]}: {r['ratio']:.4f}")

    x1 = results["en"]["ratio"]
    print(f"\nConstraint check: X1 (English) = {x1:.4f} <= 1.2 ? {'PASS' if x1 <= 1.2 else 'FAIL'}")

    return tok, results


if __name__ == "__main__":
    main()

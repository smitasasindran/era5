"""
Experiment: what does v1 cost if we drop the Lowercase normalizer, to get
case-preserving round-trip fidelity (decode(encode(text)) keeping the exact
original characters, not just the same characters modulo case)?

Self-contained copy of find_min_vocab.py + optimize_allocation.py +
merge_tokenizers.py's logic with Lowercase removed -- kept separate from the
official pipeline (which keeps Lowercase) rather than mutating it, since
this is a comparison, not a replacement.
"""
import json
import sys
from tokenizers import Tokenizer, models, pre_tokenizers, trainers

EN_TARGET = 1.2
TOTAL_VOCAB = 10000
HI_W, MR_W = 4, 5


def train_single(lang, vocab_size):
    tok = Tokenizer(models.BPE(unk_token=None))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, min_frequency=1, show_progress=False, special_tokens=[])
    tok.train([f"corpus/india_{lang}.txt"], trainer)
    return tok


def unique_words(lang, tok):
    with open(f"corpus/india_{lang}.txt", encoding="utf-8") as f:
        text = f.read()
    pretoks = tok.pre_tokenizer.pre_tokenize_str(text)
    return sorted(set(p for p, _ in pretoks))


def ratio_for(lang, vocab_size):
    tok = train_single(lang, vocab_size)
    words = unique_words(lang, tok)
    total_tokens = sum(len(e.tokens) for e in tok.encode_batch(words))
    return total_tokens / len(words), tok.get_vocab_size()


def binary_search_min_vocab(lang, target_ratio, lo, hi, verbose=False):
    best = None
    while lo <= hi:
        mid = (lo + hi) // 2
        ratio, actual_vocab = ratio_for(lang, mid)
        if verbose:
            print(f"  vocab_size={mid:5d} actual={actual_vocab:5d} ratio={ratio:.4f}")
        if ratio <= target_ratio:
            best = (mid, actual_vocab, ratio)
            hi = mid - 1
        else:
            lo = mid + 1
    return best


def himr_train(vocab_size, hi_w=HI_W, mr_w=MR_W):
    tok = Tokenizer(models.BPE(unk_token=None))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, min_frequency=1, show_progress=False, special_tokens=[])
    files = ["corpus/india_hi.txt"] * hi_w + ["corpus/india_mr.txt"] * mr_w
    tok.train(files, trainer)
    return tok


def himr_ratios(vocab_size):
    tok = himr_train(vocab_size)
    r = {}
    for lang in ["hi", "mr"]:
        words = unique_words(lang, tok)
        total = sum(len(e.tokens) for e in tok.encode_batch(words))
        r[lang] = total / len(words)
    return r, tok.get_vocab_size()


def himr_min_vocab_for_rho(rho, lo=300, hi=6000):
    best = None
    while lo <= hi:
        mid = (lo + hi) // 2
        r, actual = himr_ratios(mid)
        worst = max(r.values())
        if worst <= rho:
            best = (mid, actual, r)
            hi = mid - 1
        else:
            lo = mid + 1
    return best


def te_min_vocab_for_rho(rho, lo=100, hi=6000):
    result = binary_search_min_vocab("te", rho, lo, hi)
    if result is None:
        return None
    vocab_size, actual, ratio = result
    return actual, ratio


def find_min_common_rho(budget, lo=1.0, hi=4.0, iters=25):
    best = None
    for _ in range(iters):
        mid = (lo + hi) / 2
        te_res = te_min_vocab_for_rho(mid)
        himr_res = himr_min_vocab_for_rho(mid)
        if te_res is None or himr_res is None:
            lo = mid
            continue
        te_v, te_r = te_res
        himr_v, himr_actual, himr_r = himr_res
        total = te_v + himr_v
        if total <= budget:
            best = (mid, te_v, te_r, himr_v, himr_r, total)
            hi = mid
        else:
            lo = mid
    return best


def compute_quotas():
    en_result = binary_search_min_vocab("en", EN_TARGET, 100, 9000)
    v_en, v_en_actual, r_en = en_result
    budget_remaining = TOTAL_VOCAB - v_en_actual

    result = find_min_common_rho(budget_remaining)
    rho, te_v, te_r, himr_v, himr_r, total = result

    print(f"en: vocab={v_en_actual}, ratio={r_en:.4f}")
    print(f"te: vocab={te_v}, ratio={te_r:.4f}")
    print(f"hi+mr (joint {HI_W}:{MR_W}): vocab={himr_v}, ratios={himr_r}")
    print(f"sum={v_en_actual + te_v + himr_v}")

    return {"en": v_en_actual, "te": te_v, "himr": himr_v, "hi_w": HI_W, "mr_w": MR_W}


def merge_groups(group_tokenizers, out_path):
    combined_vocab = {}
    combined_merges = []
    seen_pairs = set()

    for tok in group_tokenizers:
        d = json.loads(tok.to_str())
        for tok_str in d["model"]["vocab"].keys():
            if tok_str not in combined_vocab:
                combined_vocab[tok_str] = len(combined_vocab)
        for a, b in d["model"]["merges"]:
            key = (a, b)
            if key in seen_pairs:
                continue
            seen_pairs.add(key)
            combined_merges.append([a, b])
            merged_tok = a + b
            if merged_tok not in combined_vocab:
                combined_vocab[merged_tok] = len(combined_vocab)

    print(f"Combined vocab size: {len(combined_vocab)}")
    final = Tokenizer(models.BPE(vocab=combined_vocab, merges=[tuple(m) for m in combined_merges], unk_token=None))
    final.pre_tokenizer = pre_tokenizers.Whitespace()
    final.save(out_path)
    print(f"Saved {out_path}")
    return final


def evaluate(tok):
    results = {}
    for lang in ["en", "hi", "te", "mr"]:
        words = unique_words(lang, tok)
        total = sum(len(e.tokens) for e in tok.encode_batch(words))
        results[lang] = total / len(words)
    spread = max(results.values()) - min(results.values())
    return results, spread


if __name__ == "__main__":
    out_path = sys.argv[1] if len(sys.argv) > 1 else "tokenizer_no_lowercase.json"

    quotas = compute_quotas()
    en_tok = train_single("en", quotas["en"])
    te_tok = train_single("te", quotas["te"])
    himr_tok = himr_train(quotas["himr"], quotas["hi_w"], quotas["mr_w"])

    final = merge_groups([en_tok, te_tok, himr_tok], out_path)

    results, spread = evaluate(final)
    print("\nFertility (no Lowercase):")
    for lang, r in results.items():
        print(f"  {lang}: {r:.4f}")
    print(f"Spread = {spread:.4f}   Score = 1000/spread ≈ {1000/spread:.1f}")

    print("\nRound-trip check:")
    text = "India, officially the Republic of India"
    decoded = final.decode(final.encode(text).ids)
    print(f"  original: {text!r}")
    print(f"  decoded:  {decoded!r}")

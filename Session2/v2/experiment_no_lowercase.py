"""
Experiment: v2 with NFKC alone (no Lowercase), for case-preserving round-trip
fidelity. Self-contained variant of train_utils.py/optimize_allocation.py/
merge_tokenizers.py's logic -- kept separate from the official v2 pipeline
(which keeps NFKC+Lowercase) rather than mutating it.
"""
import json
import sys
from tokenizers import Tokenizer, models, pre_tokenizers, trainers, normalizers, decoders
from faithful_units import count_faithful_units

UNK_TOKEN = "[UNK]"
TOTAL_VOCAB = 10000
HI_W, MR_W = 2, 3


def corpus_path(lang):
    return f"corpus/india_{lang}.txt"


def read_corpus(lang):
    with open(corpus_path(lang), encoding="utf-8") as f:
        return f.read()


def make_normalizer():
    return normalizers.NFKC()  # no Lowercase this time


def train_single(lang, vocab_size):
    tok = Tokenizer(models.BPE(unk_token=UNK_TOKEN))
    tok.normalizer = make_normalizer()
    tok.pre_tokenizer = pre_tokenizers.Metaspace()
    tok.decoder = decoders.Metaspace()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size, min_frequency=1, show_progress=False, special_tokens=[UNK_TOKEN]
    )
    tok.train([corpus_path(lang)], trainer)
    return tok


def himr_train(vocab_size, hi_w, mr_w):
    tok = Tokenizer(models.BPE(unk_token=UNK_TOKEN))
    tok.normalizer = make_normalizer()
    tok.pre_tokenizer = pre_tokenizers.Metaspace()
    tok.decoder = decoders.Metaspace()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size, min_frequency=1, show_progress=False, special_tokens=[UNK_TOKEN]
    )
    files = [corpus_path("hi")] * hi_w + [corpus_path("mr")] * mr_w
    tok.train(files, trainer)
    return tok


def fertility_for(tok, lang):
    text = read_corpus(lang)
    total_tokens = len(tok.encode(text).tokens)
    units = count_faithful_units(text)
    return total_tokens / units


def ratio_for(lang, vocab_size):
    tok = train_single(lang, vocab_size)
    return fertility_for(tok, lang), tok.get_vocab_size()


def binary_search_min_vocab(lang, target_ratio, lo, hi):
    best = None
    while lo <= hi:
        mid = (lo + hi) // 2
        ratio, actual_vocab = ratio_for(lang, mid)
        if ratio <= target_ratio:
            best = (mid, actual_vocab, ratio)
            hi = mid - 1
        else:
            lo = mid + 1
    return best


def himr_ratios(vocab_size, hi_w=HI_W, mr_w=MR_W):
    tok = himr_train(vocab_size, hi_w, mr_w)
    return {"hi": fertility_for(tok, "hi"), "mr": fertility_for(tok, "mr")}, tok.get_vocab_size()


def himr_min_vocab_for_rho(rho, lo=300, hi=8000):
    best = None
    while lo <= hi:
        mid = (lo + hi) // 2
        r, actual = himr_ratios(mid)
        if max(r.values()) <= rho:
            best = (mid, actual, r)
            hi = mid - 1
        else:
            lo = mid + 1
    return best


def min_vocab_for_rho(lang, rho, lo=100, hi=8000):
    result = binary_search_min_vocab(lang, rho, lo, hi)
    if result is None:
        return None
    vocab_size, actual, ratio = result
    return actual, ratio


def find_min_common_rho(budget, lo=0.3, hi=2.0, iters=25):
    best = None
    for _ in range(iters):
        mid = (lo + hi) / 2
        en_res = min_vocab_for_rho("en", mid)
        te_res = min_vocab_for_rho("te", mid)
        himr_res = himr_min_vocab_for_rho(mid)
        if en_res is None or te_res is None or himr_res is None:
            lo = mid
            continue
        en_v, en_r = en_res
        te_v, te_r = te_res
        himr_v, himr_actual, himr_r = himr_res
        total = en_v + te_v + himr_v
        if total <= budget:
            best = (mid, en_v, en_r, te_v, te_r, himr_v, himr_r, total)
            hi = mid
        else:
            lo = mid
    return best


def compute_quotas():
    result = find_min_common_rho(TOTAL_VOCAB)
    rho, en_v, en_r, te_v, te_r, himr_v, himr_r, total = result
    print(f"Equalized rho = {rho:.4f}")
    print(f"en: vocab={en_v}, ratio={en_r:.4f}")
    print(f"te: vocab={te_v}, ratio={te_r:.4f}")
    print(f"hi+mr (joint {HI_W}:{MR_W}): vocab={himr_v}, ratios={himr_r}")
    print(f"sum={total}")
    return {"en": en_v, "te": te_v, "himr": himr_v, "hi_w": HI_W, "mr_w": MR_W}


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
    final = Tokenizer(
        models.BPE(vocab=combined_vocab, merges=[tuple(m) for m in combined_merges], unk_token=UNK_TOKEN)
    )
    final.normalizer = make_normalizer()
    final.pre_tokenizer = pre_tokenizers.Metaspace()
    final.decoder = decoders.Metaspace()
    final.save(out_path)
    print(f"Saved {out_path}")
    return final


def evaluate(tok):
    results = {}
    for lang in ["en", "hi", "te", "mr"]:
        results[lang] = fertility_for(tok, lang)
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
    print("\nFertility (NFKC only, no Lowercase):")
    for lang, r in results.items():
        print(f"  {lang}: {r:.4f}")
    print(f"Spread = {spread:.4f}   Score = 1000/spread ≈ {1000/spread:.1f}")

    print("\nRound-trip check:")
    text = "India, officially the Republic of India"
    decoded = final.decode(final.encode(text).ids)
    print(f"  original: {text!r}")
    print(f"  decoded:  {decoded!r}")

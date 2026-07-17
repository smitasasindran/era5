"""
Script-aware allocation: Hindi and Marathi share Devanagari, so training them
independently and then concatenating merge lists causes cross-language
priority interference (a Hindi-only merge sitting earlier in the combined
list can hijack how a Marathi word merges, since encoding always applies the
earliest-listed matching rule). Verified empirically: naively merging 4
independently-trained tokenizers pushed Marathi's ratio from 2.31 -> 2.66.

Fix: train Hindi+Marathi as ONE joint pool (their own internal order is then
self-consistent, no interference) with a 4:5 Hindi:Marathi file-weight
(swept several ratios at the pair's actual quota and picked the one that
minimizes max(r_hi, r_mr)) to counteract Hindi's larger corpus dominating the
shared frequency race. English and Telugu remain single-language groups since
Latin/Devanagari/Telugu script never overlap -- concatenating those three
groups' merges is safe.

Same allocation idea as before: give English the minimal vocab for ratio<=1.2,
then split what's left between Telugu and the Hindi+Marathi pool to equalize
the worse of {r_hi, r_mr} with r_te, using the full remaining budget.
"""
from tokenizers import Tokenizer, models, pre_tokenizers, trainers, normalizers
from find_min_vocab import train_single, ratio_for, binary_search_min_vocab, unique_words

EN_TARGET = 1.2
TOTAL_VOCAB = 10000
HI_W, MR_W = 4, 5


def himr_train(vocab_size, hi_w=HI_W, mr_w=MR_W):
    tok = Tokenizer(models.BPE(unk_token=None))
    tok.normalizer = normalizers.Lowercase()
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
    result = binary_search_min_vocab("te", rho, lo, hi, verbose=False)
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


def compute_quotas(en_target=EN_TARGET, total_vocab=TOTAL_VOCAB, verbose=True):
    """Run the full allocation search and return {'en':.., 'te':.., 'himr':..}
    quotas, plus the hi:mr weight to train the himr group with. This is the
    single source of truth for quotas -- merge_tokenizers.py calls this
    instead of hardcoding numbers that would drift out of sync."""
    if verbose:
        print(f"Step 1: minimal English vocab for ratio <= {en_target}")
    en_result = binary_search_min_vocab("en", en_target, 100, 9000, verbose=False)
    v_en, v_en_actual, r_en = en_result
    if verbose:
        print(f"-> v_en = {v_en_actual}, ratio = {r_en:.4f}\n")

    budget_remaining = total_vocab - v_en_actual
    if verbose:
        print(f"Step 2: remaining budget for Telugu + (Hindi+Marathi pool) = {budget_remaining}")

    result = find_min_common_rho(budget_remaining)
    rho, te_v, te_r, himr_v, himr_r, total = result
    if verbose:
        print(f"-> equalized target rho = {rho:.4f}")
        print(f"-> Telugu: vocab={te_v}, ratio={te_r:.4f}")
        print(f"-> Hindi+Marathi pool: vocab={himr_v}, ratios={himr_r}")
        print(f"-> sum = {total}, budget = {budget_remaining}, leftover = {budget_remaining - total}")
        print("\nFinal quotas:")
        print(f"  en: {v_en_actual}")
        print(f"  te: {te_v}")
        print(f"  hi+mr (joint, weight {HI_W}:{MR_W}): {himr_v}")
        print(f"  TOTAL: {v_en_actual + te_v + himr_v}")

    return {
        "en": v_en_actual,
        "te": te_v,
        "himr": himr_v,
        "hi_w": HI_W,
        "mr_w": MR_W,
        "rho": rho,
        "r_en": r_en,
        "r_te": te_r,
        "r_himr": himr_r,
    }


if __name__ == "__main__":
    compute_quotas()

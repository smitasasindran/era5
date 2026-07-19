"""
v3 quota search: same symmetric 3-way equalization strategy as v2 (English,
Telugu, Hindi+Maithili-joint), just applied to the instructor's richer
corpus with Maithili in place of Marathi as the 4th language. Purpose: an
apples-to-apples test of our allocation-search strategy against their
corpus, holding the training strategy as the only variable (see report.md).

Hindi+Maithili are trained jointly (share Devanagari -- see v1's
optimize_allocation.py docstring for why independently-trained same-script
tokenizers can't just be concatenated).
"""
from train_utils import binary_search_min_vocab, himai_train, fertility_for

TOTAL_VOCAB = 10000
EN_MAX = 1.2  # still must hold, but should never bind in this regime
HI_W, MAI_W = 1, 2


def himai_ratios(vocab_size, hi_w=HI_W, mai_w=MAI_W):
    tok = himai_train(vocab_size, hi_w, mai_w)
    r_hi, _, _ = fertility_for(tok, "hi")
    r_mai, _, _ = fertility_for(tok, "mai")
    return {"hi": r_hi, "mai": r_mai}, tok.get_vocab_size()


def himai_min_vocab_for_rho(rho, lo=300, hi=8000):
    best = None
    while lo <= hi:
        mid = (lo + hi) // 2
        r, actual = himai_ratios(mid)
        worst = max(r.values())
        if worst <= rho:
            best = (mid, actual, r)
            hi = mid - 1
        else:
            lo = mid + 1
    return best


def min_vocab_for_rho(lang, rho, lo=100, hi=8000):
    result = binary_search_min_vocab(lang, rho, lo, hi, verbose=False)
    if result is None:
        return None
    vocab_size, actual, ratio = result
    return actual, ratio


def find_min_common_rho(budget, lo=0.1, hi=2.0, iters=25):
    best = None
    for _ in range(iters):
        mid = (lo + hi) / 2
        en_res = min_vocab_for_rho("en", mid)
        te_res = min_vocab_for_rho("te", mid)
        himai_res = himai_min_vocab_for_rho(mid)
        if en_res is None or te_res is None or himai_res is None:
            lo = mid
            continue
        en_v, en_r = en_res
        te_v, te_r = te_res
        himai_v, himai_actual, himai_r = himai_res
        total = en_v + te_v + himai_v
        if total <= budget:
            best = (mid, en_v, en_r, te_v, te_r, himai_v, himai_r, total)
            hi = mid
        else:
            lo = mid
    return best


def compute_quotas(total_vocab=TOTAL_VOCAB, verbose=True):
    result = find_min_common_rho(total_vocab)
    rho, en_v, en_r, te_v, te_r, himai_v, himai_r, total = result

    if verbose:
        print(f"Equalized target rho = {rho:.4f}")
        print(f"  English: vocab={en_v}, ratio={en_r:.4f}")
        print(f"  Telugu:  vocab={te_v}, ratio={te_r:.4f}")
        print(f"  Hindi+Maithili (joint {HI_W}:{MAI_W}): vocab={himai_v}, ratios={himai_r}")
        print(f"  sum={total}, budget={total_vocab}, leftover={total_vocab - total}")
        print(f"  X1 (English) < {EN_MAX}? {'PASS' if en_r < EN_MAX else 'FAIL'}")

    return {
        "en": en_v,
        "te": te_v,
        "himai": himai_v,
        "hi_w": HI_W,
        "mai_w": MAI_W,
        "rho": rho,
        "r_en": en_r,
        "r_te": te_r,
        "r_himai": himai_r,
    }


if __name__ == "__main__":
    compute_quotas()

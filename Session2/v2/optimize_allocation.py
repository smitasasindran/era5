"""
v2 quota search: unlike v1, English no longer needs special protection --
Metaspace + markdown corpus + NFKC/Lowercase already gets every language's
solo fertility close to ~1.0 at a modest vocab share, comfortably clearing
the X1 < 1.2 requirement without a dedicated carve-out. So instead of
"give English the bare minimum, then equalize the rest" (v1's strategy),
v2 does a fully symmetric 3-way equalization across all three groups
(English, Telugu, Hindi+Marathi-joint) using the whole 10,000 budget --
binary search for the smallest common fertility target rho such that the
three groups' minimum-vocab-for-rho sums to <= budget.

Hindi+Marathi are still trained jointly (they share Devanagari -- see v1's
optimize_allocation.py docstring for why independently-trained same-script
tokenizers can't just be concatenated).
"""
from train_utils import binary_search_min_vocab, himr_train, fertility_for

TOTAL_VOCAB = 10000
EN_MAX = 1.2  # still must hold, but should never bind in this regime
HI_W, MR_W = 2, 3


def himr_ratios(vocab_size, hi_w=HI_W, mr_w=MR_W):
    tok = himr_train(vocab_size, hi_w, mr_w)
    r_hi, _, _ = fertility_for(tok, "hi")
    r_mr, _, _ = fertility_for(tok, "mr")
    return {"hi": r_hi, "mr": r_mr}, tok.get_vocab_size()


def himr_min_vocab_for_rho(rho, lo=300, hi=8000):
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


def min_vocab_for_rho(lang, rho, lo=100, hi=8000):
    result = binary_search_min_vocab(lang, rho, lo, hi, verbose=False)
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


def compute_quotas(total_vocab=TOTAL_VOCAB, verbose=True):
    result = find_min_common_rho(total_vocab)
    rho, en_v, en_r, te_v, te_r, himr_v, himr_r, total = result

    if verbose:
        print(f"Equalized target rho = {rho:.4f}")
        print(f"  English: vocab={en_v}, ratio={en_r:.4f}")
        print(f"  Telugu:  vocab={te_v}, ratio={te_r:.4f}")
        print(f"  Hindi+Marathi (joint {HI_W}:{MR_W}): vocab={himr_v}, ratios={himr_r}")
        print(f"  sum={total}, budget={total_vocab}, leftover={total_vocab - total}")
        print(f"  X1 (English) < {EN_MAX}? {'PASS' if en_r < EN_MAX else 'FAIL'}")

    return {
        "en": en_v,
        "te": te_v,
        "himr": himr_v,
        "hi_w": HI_W,
        "mr_w": MR_W,
        "rho": rho,
        "r_en": en_r,
        "r_te": te_r,
        "r_himr": himr_r,
    }


if __name__ == "__main__":
    compute_quotas()

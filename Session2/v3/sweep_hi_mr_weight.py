"""
Sweep Hindi:Marathi joint-training weights for v3's corpus. The corpus size
ratio between them changed from v2 (14,808 vs 8,284 faithful units, ~1.8x)
to v3 (88,359 vs 29,766, ~3.0x) since the richer HTML-based fetch grew
Hindi's corpus proportionally more than Marathi's -- so v2's 2:3 weight
likely isn't optimal here anymore; re-tuning from scratch.

Usage: python sweep_hi_mr_weight.py [vocab_size]
"""
import sys
from train_utils import himr_train, fertility_for

VOCAB_SIZE = 4000
WEIGHTS_TO_TRY = [
    (2, 5), (3, 7), (1, 2), (3, 8), (4, 9),
]


def sweep(vocab_size):
    results = []
    for hi_w, mr_w in WEIGHTS_TO_TRY:
        tok = himr_train(vocab_size, hi_w, mr_w)
        r_hi, _, _ = fertility_for(tok, "hi")
        r_mr, _, _ = fertility_for(tok, "mr")
        worst = max(r_hi, r_mr)
        results.append((hi_w, mr_w, r_hi, r_mr, worst))
        print(f"{hi_w}:{mr_w} -> hi={r_hi:.4f} mr={r_mr:.4f} max={worst:.4f}")

    best = min(results, key=lambda r: r[4])
    print(f"\nBest: {best[0]}:{best[1]} (max={best[4]:.4f})")
    return best


if __name__ == "__main__":
    vocab_size = int(sys.argv[1]) if len(sys.argv) > 1 else VOCAB_SIZE
    sweep(vocab_size)

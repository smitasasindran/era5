"""
Sweep Hindi:Maithili joint-training weights to find the one that minimizes
max(r_hi, r_mai) at a representative vocab size, the same way v1's 4:5 and
v2's 2:3 Hindi:Marathi weights were found. Hindi's corpus here is far larger
than Maithili's (88,359 vs 5,808 faithful units, ~15x) -- a bigger imbalance
than v1/v2 had with Marathi -- so Maithili likely needs heavy counter-weight.

Usage: python sweep_hi_mai_weight.py [vocab_size]
"""
import sys
from train_utils import himai_train, fertility_for

VOCAB_SIZE = 4000
WEIGHTS_TO_TRY = [
    (1, 1), (1, 2), (1, 3), (1, 4), (1, 5),
]


def sweep(vocab_size):
    results = []
    for hi_w, mai_w in WEIGHTS_TO_TRY:
        tok = himai_train(vocab_size, hi_w, mai_w)
        r_hi, _, _ = fertility_for(tok, "hi")
        r_mai, _, _ = fertility_for(tok, "mai")
        worst = max(r_hi, r_mai)
        results.append((hi_w, mai_w, r_hi, r_mai, worst))
        print(f"{hi_w}:{mai_w:3d} -> hi={r_hi:.4f} mai={r_mai:.4f} max={worst:.4f}")

    best = min(results, key=lambda r: r[4])
    print(f"\nBest: {best[0]}:{best[1]} (max={best[4]:.4f})")
    return best


if __name__ == "__main__":
    vocab_size = int(sys.argv[1]) if len(sys.argv) > 1 else VOCAB_SIZE
    sweep(vocab_size)

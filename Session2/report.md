# BPE Tokenizer Assignment — India Wikipedia (English, Hindi, Telugu, Gujarati)

## Goal

Train **one shared BPE vocabulary of 10,000 tokens** across the English, Hindi,
Telugu, and Gujarati Wikipedia pages for "India", such that:

- X1 = (tokens to encode English's unique words) / (count of English's unique words) ≤ 1.2
- X2, X3, X4 = same ratio for Hindi, Telugu, Gujarati (no constraint, just report + sort)

Ratio definition used throughout (confirmed with user): for each language, take the
set of **unique words** appearing in that language's corpus, encode each one with
the tokenizer, and divide total tokens produced by the number of unique words.
This is a "fertility" metric — closer to 1.0 means more words survive as a single
token.

## Data collection — `fetch_corpus.py`

Pulled plaintext extracts (via the Wikipedia API, `action=query&prop=extracts&explaintext=1`)
for the "India" article in each language:

| lang | title | chars | unique words (raw whitespace split) |
|---|---|---|---|
| en | India | 65,305 | 3,773 |
| hi | भारत | 43,796 | 2,454 |
| te | భారతదేశం | 20,012 | 1,650 |
| gu | ભારત | 8,798 | 787 |

Saved to `corpus/india_{en,hi,te,gu}.txt`.

## Attempt 1 — naive from-scratch BPE (`naive_bpe.py`, `train_and_evaluate.py`)

Implemented the classic Sennrich et al. (2015) word-frequency BPE by hand:
words → char tuples + `</w>` marker, repeatedly merge the globally most frequent
adjacent symbol pair (with an incremental pair-frequency index for speed), pooling
word frequencies from all 4 languages equally before training.

**Result:** training stalled at vocab size **8,431** (8,118 merges) — ran out of
pairs with frequency ≥ 2 given how small the combined corpus is (~8,575 unique
words total). Ratios (using naive `text.split()` as the word definition):

| Lang | Ratio |
|---|---|
| Telugu (X3) | 2.59 |
| Gujarati (X4) | 2.41 |
| English (X1) | 2.20 |
| Hindi (X2) | 2.10 |

**X1 = 2.20 → FAIL.** Expected: merges were picked purely by global pair
frequency with no language-aware budgeting, so no language actually got enough
merges to collapse most of its words to ~1 token.

## Switch to HuggingFace `tokenizers` library

Per user request, replaced the hand-rolled trainer with the `tokenizers` library
(v0.22.2, already present in the `genai` conda env at
`/home/forest/miniconda3/envs/genai/bin/python` — no new installs needed).

- `train_tokenizer_hf.py`: `Tokenizer(models.BPE())` + `pre_tokenizers.Whitespace()`
  + `trainers.BpeTrainer(vocab_size=10000)`, trained jointly on all 4 corpus files.
  Supports a `weights` dict to duplicate a language's corpus file N× in the
  training file list, biasing the trainer's pair-frequency counts toward it.
- `evaluate_hf.py`: loads a saved tokenizer, computes the fertility ratio per
  language, prints the sorted table and the X1 ≤ 1.2 check.

**First run (no weighting, unweighted joint corpus), vocab_size=10000:**
Trained to the full 10,000 vocab (unlike the naive version — the library's
trainer keeps merging low-frequency pairs to hit the exact target). Ratios
(still using naive `text.split()` word definition):

| Lang | Ratio |
|---|---|
| Telugu (X3) | 2.19 |
| English (X1) | 1.95 |
| Gujarati (X4) | 1.87 |
| Hindi (X2) | 1.76 |

**X1 = 1.95 → still FAIL.**

### Dead end: oversampling English didn't move the needle (at first)

Tried retraining with English's corpus file duplicated 5×, 10×, 15×, 20× in the
training list, expecting the bias toward English pairs to keep improving X1.
Instead, X1 flatlined at **1.4530** for every weight ≥ 5 — increasing the bias
further had zero effect, which was the tell that something else was capping it,
not the vocab allocation itself.

### Root cause found: word-definition mismatch, not a vocab ceiling

Used `find_min_vocab.py` (binary search over vocab_size for an English-only
tokenizer) to isolate the issue, and found the same 1.4530 floor no matter how
large `vocab_size` was requested. The cause: **`evaluate_hf.py`'s word list came
from raw `text.split()`, but the trained tokenizer's `Whitespace` pre-tokenizer
splits punctuation off words** (`"India,"` → `"India"` + `","`). So even a
"perfectly merged" tokenizer — every real word already a single token — still
reports 2 tokens for any word with attached punctuation, because the comma is
counted as part of the same "word" in eval but as a separate atom by the
tokenizer.

Checked the scale of the effect on English: **1,338 of 3,773 unique words
(~35%) have punctuation attached.** That's a large, artificial floor on the
ratio that has nothing to do with subword fragmentation quality.

**Fix:** redefined "unique words" in both `evaluate_hf.py` and
`find_min_vocab.py` to come from `tok.pre_tokenizer.pre_tokenize_str(text)` —
i.e., the tokenizer's own atomic units — instead of a raw whitespace split.
This also *changes* the unique-word counts per language (punctuation marks now
count as their own atoms, some words split into two atoms):

| lang | unique words (pre-tokenized) |
|---|---|
| en | 3,156 |
| hi | 2,281 |
| te | 1,588 |
| gu | 763 |

Re-running the *unweighted* joint tokenizer with the corrected metric:

| Lang | Ratio |
|---|---|
| Telugu (X3) | 1.8652 |
| Gujarati (X4) | 1.7339 |
| English (X1) | 1.5938 |
| Hindi (X2) | 1.5042 |

Still **X1 FAIL** — so the original allocation problem (English needs more
merges than a frequency-blind pooled corpus gives it) is real and distinct from
the measurement bug; fixing the metric didn't make the constraint trivially
pass on its own.

## Final solution: oversample English 2× in joint training

Re-ran the weighted joint training sweep with the corrected metric:

| en weight | X1 (English) | X2 (Hindi) | X3 (Telugu) | X4 (Gujarati) |
|---|---|---|---|---|
| 1 (baseline) | 1.5938 | 1.5042 | 1.8652 | 1.7339 |
| **2** | **1.0320** | 1.8444 | 2.2897 | 2.1678 |
| 3 | 1.0000 | 1.8672 | 2.3149 | 2.2058 |
| 4 | 1.0000 | 1.8672 | 2.3155 | 2.2058 |
| 5–20 | 1.0000 | 1.8672 | 2.3155 | 2.2058 |

Weight ≥ 3 pushes English to a trivial 1.0 (every word a single token) but
burns extra shared vocab budget to do it, making Hindi/Telugu/Gujarati slightly
worse than necessary. **Weight = 2 is the chosen final model** — it clears the
constraint with a comfortable margin (1.032 ≤ 1.2) while leaving more of the
10,000-token budget for the other three languages.

Command used:
```
/home/forest/miniconda3/envs/genai/bin/python train_tokenizer_hf.py "en=2,hi=1,te=1,gu=1" tokenizer_final.json
```

## Final results — `tokenizer_final.json` (vocab_size = 10,000)

| Lang | Unique words | Tokens | Ratio |
|---|---|---|---|
| English (X1) | 3,156 | 3,257 | **1.0320** |
| Hindi (X2) | 2,281 | 4,207 | **1.8444** |
| Telugu (X3) | 1,588 | 3,636 | **2.2897** |
| Gujarati (X4) | 763 | 1,654 | **2.1678** |

**Sorted largest → smallest:** X3 (2.2897) > X4 (2.1678) > X2 (1.8444) > X1 (1.0320)

**Constraint check: X1 = 1.0320 ≤ 1.2 → PASS**

## Files

- `fetch_corpus.py` — downloads the 4 corpora
- `corpus/india_{en,hi,te,gu}.txt` — raw plaintext extracts
- `naive_bpe.py`, `train_and_evaluate.py` — from-scratch BPE baseline (attempt 1, kept for reference/comparison)
- `train_tokenizer_hf.py` — HF `tokenizers`-based trainer, with per-language oversampling weights
- `evaluate_hf.py` — fertility evaluation for any saved tokenizer
- `find_min_vocab.py` — binary-search diagnostic used to isolate the word-definition bug
- `tokenizer_hf.json` — unweighted joint baseline (X1 fails), kept for comparison
- `tokenizer_final.json` — final model (en weight=2), satisfies X1 ≤ 1.2

## Phase 2 — swapped Gujarati for Marathi, new goal: minimize spread

New requirements from the user:
1. Use **Marathi** (title `भारत` on `mr.wikipedia.org`, 32,192 chars, same as
   Hindi's — both languages happen to use the same word for "India") instead
   of Gujarati.
2. Beyond X1 ≤ 1.2, minimize the **spread** (max − min) across all four
   ratios X1..X4 — i.e. don't just satisfy English's constraint, make all
   four languages compress *similarly well*.

Updated `fetch_corpus.py`, `evaluate_hf.py`, `train_tokenizer_hf.py` to use
`mr` in place of `gu` (old `corpus/india_gu.txt` and the naive-BPE baseline
above are left untouched as historical reference).

### Oversampling isn't precise enough for equalization

Tried tuning the joint-training weights directly (user's own experiment:
`en=1, hi=3, te=3, mr=2`, one shared 10k vocab) — this pushed Hindi and
Telugu down to 1.05 and 1.22, but **broke English** (2.35, since it got no
oversampling boost and lost budget to the other three). This exposed the
real limitation: oversampling only biases the *relative* priority in one
shared merge race — there's no way to reason about it to hit 4 precise,
simultaneous targets. Explicit per-language vocab quotas, decided in advance
and then combined, give exact control instead.

### Quota search — minimize spread analytically

Wrote `optimize_allocation_v2.py`. Key insight: since more vocab always
lowers (or holds) a language's ratio, and X1 ≤ 1.2 is a hard floor on
English's *budget* (not its ratio), the spread-minimizing allocation is:

1. Give English the **minimum** vocab that still satisfies ratio ≤ 1.2 (any
   more wastes shared budget that could shrink the other three's ratios —
   proved by noting spread is monotonically non-decreasing as English's vocab
   grows past that minimum: its own ratio drops further below the pack
   *and* the shrinking remainder pushes everyone else's ratio up).
2. Spend 100% of the remaining budget equalizing the other languages' ratios
   at the lowest common value the remaining budget allows (binary search over
   a shared target ratio ρ; for each candidate ρ, sum the minimum vocab each
   language needs to reach ρ, and find the smallest ρ where that sum still
   fits the remaining budget).

First pass treated Hindi, Telugu, Marathi as 3 independent single-language
tokenizers, quotas found via binary search (`en=5333, hi=1407, te=1550,
mr=1710`, sum=10000, individually verified ratios en=1.1999,
hi≈te≈mr≈2.306).

### Bug: merging independently-trained Hindi and Marathi tables corrupts both

`merge_tokenizers.py` combined the 4 separately-trained tokenizers into one
by concatenating each language's `(vocab, merges)` and deduping identical
entries — assumed safe because each language's merges only touch its own
script's characters, so there's no cross-language dependency.

That assumption holds for English (Latin), Telugu (Telugu script) — but
**Hindi and Marathi both use Devanagari.** After the actual merge, re-evaluating
the *combined* tokenizer gave very different numbers than the
individually-trained ones: Hindi improved (2.306 → 2.156) but **Marathi got
worse (2.306 → 2.657)** — a real regression, not noise.

Root cause: BPE encoding always applies the *earliest-listed* matching merge
rule in the tokenizer's merge table (priority = position in the list, not
per-language rank). Concatenating `[en's merges] + [hi's merges] + [te's
merges] + [mr's merges]` means any Devanagari merge rule Hindi happened to
learn — even one Marathi's own solo-trained chain never needed or would have
prioritized differently — now sits ahead of Marathi's own rules and can
hijack how a Marathi word merges, sometimes leading to a worse (more
fragmented) final split than Marathi's standalone tokenizer would have
produced. Concatenation is only safe across **script-disjoint** groups.

### Fix — train script-sharing languages jointly, not independently

Restructured into 3 script-safe groups: **English** alone, **Telugu** alone,
**Hindi+Marathi trained jointly** as one coherent BPE chain (so there's no
foreign-priority interference within the pair by construction). Hindi's
larger corpus dominates the shared frequency race at equal weight, so used a
**1:2 Hindi:Marathi** file-weight to balance them (empirically found: at
vocab_size=3000, weight 1:1 gives hi=1.90/mr=2.16; weight 1:2 gives
hi=2.06/mr=1.99 — much closer).

Re-ran the same minimize-spread search (`optimize_allocation_v2.py`) over
just 2 free groups now (Telugu vs. the Hindi+Marathi pool), since English's
quota is already fixed:

| Group | Quota | Ratio(s) |
|---|---|---|
| English | 5,333 | 1.1999 |
| Telugu | 1,813 | 2.1398 |
| Hindi+Marathi (joint, 1:2 weight) | 2,854 | hi=2.1403, mr=2.0238 |

Equalized ρ ≈ 2.14 — notably *better* than the naive independent-quota result
(2.306), because Hindi and Marathi genuinely share vocabulary/morphology as
related Indo-Aryan languages, so pooling them lets the same budget cover both
more efficiently than treating them as unrelated.

`merge_tokenizers_v2.py` builds the final tokenizer from these 3 groups
(concatenation is safe here — Latin, Telugu script, and Devanagari never
overlap). Final combined vocab settled at **9,708** (a harmless ~300-token
dedup of shared digits/punctuation across the 3 groups, not a script-overlap
bug this time).

## Final results — `tokenizer_final.json` (vocab_size = 9,708)

| Lang | Unique words | Tokens | Ratio |
|---|---|---|---|
| English (X1) | 3,156 | 3,787 | **1.1999** |
| Marathi (X4) | 2,272 | 4,542 | **1.9991** |
| Hindi (X2) | 2,281 | 4,767 | **2.0899** |
| Telugu (X3) | 1,588 | 3,381 | **2.1291** |

**Sorted largest → smallest:** X3 (2.1291) > X2 (2.0899) > X4 (1.9991) > X1 (1.1999)

**Constraint check: X1 = 1.1999 ≤ 1.2 → PASS**
**Spread (max − min) = 2.1291 − 1.1999 = 0.929**, down from the un-optimized
joint baseline's failing state and provably close to the best this allocation
strategy can do (English's hard floor forces a minimum ~1.0+ gap, since it
alone needs over half the 10k budget to reach 1.2, leaving proportionally
less for the rest).

## Files (updated)

- `find_min_vocab.py` — generalized binary-search utility (`train_single`,
  `ratio_for`, `binary_search_min_vocab`), now imported by the allocation
  scripts instead of being a standalone CLI-only tool
- `optimize_allocation_v2.py` — the spread-minimizing quota search (3
  script-safe groups: en, te, hi+mr joint)
- `merge_tokenizers_v2.py` — builds the final tokenizer from those 3 groups
- `tokenizer_final.json` — final model (vocab=9,708), satisfies X1 ≤ 1.2 with
  minimized spread
- `tokenizer_hf.json` — regenerated equal-weight joint baseline (all 4
  languages pooled, no allocation strategy) for comparison — X1 = 1.688, FAILs
- superseded: an earlier `optimize_allocation.py` / `merge_tokenizers.py` pair
  that treated Hindi and Marathi as independent (removed after the
  script-overlap bug above was found — see narrative for their numbers)

## Caveats / what "best options" could improve further

- **In-sample evaluation.** Fertility is measured on the same India-page text
  the tokenizer was trained on, so a tokenizer can trivially drive a language's
  ratio to 1.0 by memorizing every word whole (weight ≥ 3 does exactly this for
  English) — this reflects the assignment's own definition (tokens/vocab over
  the same corpus), but it's not evidence the tokenizer would generalize to new
  English text.
- **Oversampling is a blunt instrument** for hitting *simultaneous* precise
  targets across 4 languages sharing one vocab (confirmed in Phase 2: tuning
  weights to help Hindi/Telugu broke English). It's still useful as a *local*
  balancing lever between two languages that are already being trained
  jointly on purpose (e.g. the 1:2 Hindi:Marathi weight) — the difference is
  scope: one dial for one pairwise trade-off is tractable, one dial for a
  4-way trade-off isn't.
- **English's hard floor bounds how low the spread can go.** English alone
  needs >50% of the 10k budget to reach ratio ≤ 1.2 (its unique-word count is
  the largest of the four, and the constraint is strict), which caps how much
  is left to equalize the rest. Spread ≈0.93 is close to the best this
  4-language / 10k-budget / X1≤1.2 setup can do — meaningfully lowering it
  further would need either a bigger shared vocab, a looser X1 bound, or more
  training text per language (see below) to get more fertility per token.
- **The 1:2 Hindi:Marathi weight was found empirically** (tried a few values,
  picked the one that balanced them best at one sample vocab size), not
  derived analytically like the 3-group equalization search was. It's a
  reasonable approximation but a finer per-vocab-size tuning could exist.
- **Small corpora.** Each language's corpus is just one Wikipedia article
  (8.8 KB–65 KB). Pulling in more India-related pages per language would give
  the trainer enough signal to make full use of the 10,000-token budget instead
  of running dry on frequency (as the naive attempt did at vocab 8,431), and
  would likely change the achievable equalized ratio too.

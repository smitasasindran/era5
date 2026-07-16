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

## Caveats / what "best options" could improve further

- **In-sample evaluation.** Fertility is measured on the same India-page text
  the tokenizer was trained on, so a tokenizer can trivially drive a language's
  ratio to 1.0 by memorizing every word whole (weight ≥ 3 does exactly this for
  English) — this reflects the assignment's own definition (tokens/vocab over
  the same corpus), but it's not evidence the tokenizer would generalize to new
  English text.
- **Oversampling is a blunt instrument.** It works here only because it happens
  to bias the greedy merge order in the right direction; it doesn't give
  precise control over exactly how much vocab each language gets. A cleaner
  "best options" approach would explicitly reserve a per-language token quota
  (e.g. train each language's BPE separately to a target vocab size, then
  union the merge tables into one final vocab), which is more controllable and
  auditable than tuning a duplication factor by trial and error.
- **Small corpora.** Each language's corpus is just one Wikipedia article
  (8.8 KB–65 KB). Pulling in more India-related pages per language would give
  the trainer enough signal to make full use of the 10,000-token budget instead
  of running dry on frequency (as the naive attempt did at vocab 8,431).

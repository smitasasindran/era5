# Experiment: our allocation-search method on the instructor's corpus

**This is not "Version 3" of the project tokenizer.** It's a validation
experiment: same allocation-search training strategy as v2 (script-safe
3-group merge, symmetric equalization), but trained on the instructor's own
corpus (`../../solution2/corpus/*.faithful.txt`, fetched via rendered-HTML +
`markdownify`) with **Maithili** instead of Marathi, to isolate "is their
corpus just bigger" from "is their training strategy better" — see the main
project's discussion for context.

The real Version 3 (in `../v3/`) keeps the project's actual 4 languages
(English/Hindi/Telugu/Marathi) and adopts the corpus-building *technique*
learned here (HTML + markdownify, much richer than v2's wikitext parsing),
not this exact corpus or language set.

## Result

| Lang | Fertility |
|---|---|
| Hindi | 0.7735 |
| Telugu | 0.7722 |
| English | 0.6706 |
| Maithili | 0.6701 |

Spread = 0.1033, **Score ≈ 9,676** — beats the instructor's own reported
score (6,503) on their exact corpus, using our training strategy instead of
theirs. Vocab dedup during the 3-group merge was larger than expected here
(8,469 final vocab vs 10,000 allocated — this corpus's 3 groups apparently
share more vocabulary, e.g. common URLs/digits/Latin tokens, than v1/v2's
did), which likely cost some of the equalization precision; the pre-merge,
per-group numbers (`optimize_allocation.py`'s own search) suggested a much
tighter spread (~0.04) before that dedup loss. Re-tuning quotas to account
for it would likely close the gap further, but wasn't pursued since this
folder's job (validating the method) is already done.

## Files

Same structure as `v2/`: `train_utils.py`, `optimize_allocation.py`,
`merge_tokenizers.py`, `evaluate.py`, `faithful_units.py`, `sanity_check.py`,
plus `sweep_hi_mai_weight.py` (found the 1:2 Hindi:Maithili joint-training
weight — interesting that unlike v1/v2's Hindi:Marathi tuning, near-equal
weight was already close to optimal here, likely because Maithili's much
smaller corpus needed proportionally less vocab to reach good coverage
despite Hindi's ~15x larger corpus dominating the raw frequency race).

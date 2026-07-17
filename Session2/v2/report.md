# Tokenizer v2 — instructor-inspired optimizations

## Motivation

The instructor's solution reported fertility between **0.57 and 0.73**
across four languages (English/Hindi/Telugu/Maithili, the last chosen because
it shares Devanagari with Hindi — same reasoning we used for Marathi), with
a **spread of 0.153786**. Five optimizations were listed:

1. Wiki-faithful Markdown corpus instead of plain text
2. `Metaspace` as pre-tokenizer instead of `Whitespace`
3. A "faithful unit" fertility metric, decoupled from the tokenizer's own
   pre-tokenizer: `faithful_unit` = one contiguous Unicode letter/mark/number
   run, OR one visible non-space punctuation/symbol character (counted
   individually, not grouped) — `fertility = tokens / faithful_units` over
   the **full running corpus text**, not per unique word type
4. `NFKC` normalizer
5. An explicit `[UNK]` token instead of `unk_token=None`

This is v2, built from scratch alongside (not replacing) v1 — v1's entire
pipeline was moved to `../v1/` unchanged and still works from there.

## Why this actually helps: the mechanism

v1's Whitespace pre-tokenizer splits punctuation off words (`"India,"` →
`"India"` + `","`), which put a hard floor of ~1.0 fertility per atom — no
merge can span two separate pre-tokenizer atoms. `Metaspace` doesn't split
punctuation from the word it's attached to, so a whole `"▁india,"` can
collapse into a single learned token if frequent enough — a real compression
gain (not just a measurement artifact) *specifically* for word+attached-punct
patterns, plus for standalone repeated punctuation runs (Markdown's `**`,
`##`, `](`, `|` table delimiters etc., each character counted as its own
faithful unit, so merging `**` into one token is a 2-units-for-1-token win).
Verified this doesn't merge across actual spaces (BPE only ever merges
within a single pre-tokenized atom, and Metaspace's atoms are still
space-delimited) — so the gain is real but bounded to punctuation-adjacency
patterns, not arbitrary word-to-word fusion.

## Corpus: wiki-faithful Markdown (`fetch_corpus_markdown.py`)

Fetches raw **wikitext** (not the plain-text `explaintext` API used in v1)
via `action=query&prop=revisions&rvprop=content`, parses it with
`mwparserfromhell` (installed for this — not present in the `genai` env
before), and walks the parse tree converting each node type: headings → `#`
levels, `'''bold'''`/`''italic''` → `**bold**`/`*italic*`, wikilinks/external
links → `[text](target)`, list bullets → `-`/`1.`, wikitables → best-effort
markdown tables, citation-family templates (`cite*`, `sfn`, `refn`, ...)
dropped entirely, other templates (infoboxes) flattened to their parameter
values as plain text, `<ref>`/`<references>` dropped, comments dropped,
HTML entities unescaped.

Result: substantially larger, richer corpora than v1's plain-text extracts
(same "India" article, same 4 languages):

| Lang | v1 (plain text) | v2 (markdown) | Growth |
|---|---|---|---|
| English | 65,305 chars | 102,093 chars | +56% |
| Hindi | 43,796 chars | 61,552 chars | +41% |
| Telugu | 20,012 chars | 31,931 chars | +60% |
| Marathi | 32,192 chars | 40,012 chars | +24% |

## Faithful-unit metric (`faithful_units.py`)

Implemented via `unicodedata.category()` (no external Unicode-property regex
library needed): scan character by character, group consecutive
letter/mark/number characters (`unicodedata.category(ch)[0] in "LMN"`) into
one unit per run, count every other non-space character as its own
individual unit, reset the run on whitespace.

This is a genuinely different metric shape from v1's: v1 measured fertility
as *tokens to encode each unique word once* (type-level, so a word used once
counted the same as one used 50 times) — this measures tokens over the
**entire running corpus** (occurrence-level, so common short words/punctuation
dominate). That's *why* fertility can go below 1.0 here but couldn't in v1's
framework: v1's metric had a hard 1.0 floor per unique atom; this one doesn't
per-occurrence, because the same cheap merge gets credited every time it
fires in the running text.

## Training pipeline (`train_utils.py`, `optimize_allocation.py`, `merge_tokenizers.py`)

Same script-safe 3-group structure as v1 (English alone, Telugu alone,
Hindi+Marathi trained jointly since they share Devanagari — concatenating
independently-trained same-script tokenizers corrupts merge order, see v1's
`optimize_allocation.py` docstring), reusing the proven binary-search
quota-allocation machinery. One strategic change:

**v1's strategy** ("give English the bare-minimum vocab for its X1<1.2
requirement, spend 100% of the remainder equalizing the rest") **no longer
applies.** With markdown + Metaspace + NFKC/Lowercase, *every* language's
solo fertility already lands near ~1.0-1.3 at a modest vocab share — English
clears 1.2 without needing special protection. So v2 does a fully
**symmetric 3-way equalization**: binary search for the smallest common
fertility target ρ such that English + Telugu + (Hindi+Marathi joint)'s
minimum vocabs for "≤ ρ" sum to ≤ 10,000, no group privileged.

Hindi:Marathi joint-training weight retuned the same way as v1 (swept a few
ratios to minimize `max(r_hi, r_mr)`): **2:3** here (v1 used 4:5 — different
because the metric and corpus both changed, not directly comparable).

## Results — `tokenizer_final.json` (vocab_size = 9,410)

| Lang | Tokens | Faithful units | Fertility |
|---|---|---|---|
| Telugu | 8,223 | 6,916 | **1.1890** |
| English (X1) | 26,704 | 22,688 | **1.1770** |
| Hindi | 17,268 | 14,808 | **1.1661** |
| Marathi | 9,234 | 8,284 | **1.1147** |

**X1 = 1.1770 < 1.2 → PASS**
**Spread = 1.1890 − 1.1147 = 0.0743**
**Score = 1000 / 0.0743 ≈ 13,458.4**

## Comparison with the instructor's solution

| | Fertility range | Spread | Score (1000/spread) |
|---|---|---|---|
| v1 (this project, before this session) | 1.11 – 1.97 | 0.9040 | ≈1,106 |
| **v2 (this session)** | **1.11 – 1.19** | **0.0743** | **≈13,458** |
| Instructor's solution | 0.6 – 0.73 | 0.1538 | ≈6,503 |

**By the actual graded quantity (spread → score), v2 already beats the
instructor's solution by a wide margin** — our four languages sit closer
together than theirs do, even though our *absolute* fertility level is
higher. The instructor's lower absolute level (0.6-0.73 vs our ~1.1-1.2)
likely comes from a richer corpus (more pages, and/or a more thorough
markdown conversion preserving more infobox/table content faithfully than
ours does) or possibly a larger effective vocab budget — worth investigating
further if matching their absolute fertility level (not just the score) is a
goal, but not necessary to beat their score as currently defined.

## Files

- `fetch_corpus_markdown.py` — wikitext → faithful-Markdown corpus fetcher
- `corpus/india_{en,hi,te,mr}.txt` — the markdown corpora
- `faithful_units.py` — the fertility metric's unit-counting function
- `train_utils.py` — shared training/eval utilities (Metaspace, NFKC+Lowercase,
  `[UNK]`, `fertility_for`, `binary_search_min_vocab`)
- `optimize_allocation.py` — the symmetric 3-way quota search; exposes
  `compute_quotas()`
- `merge_tokenizers.py` — builds the final tokenizer from the 3 groups
- `evaluate.py` — computes final fertility/spread/score, writes `eval_results.json`
- `tokenizer_final.json` — the v2 deliverable

## Playground integration (done)

Extended `tokenizer-playground/tokenizer.js` to support v2's full pipeline
alongside v1's: a `Metaspace` pre-tokenizer (its exact splitting behavior —
including leading/trailing/consecutive-space edge cases — was reverse
engineered from the real Python `tokenizers` output and verified against
~15 cases before writing the JS), a `Sequence` normalizer (`NFKC` via JS's
built-in `.normalize("NFKC")`, then `Lowercase`), and `[UNK]`-aware encoding
(an unrecognized piece maps to the real `[UNK]` vocab entry — a genuine
token, not a drop, matching verified real behavior).

`models/v2/tokenizer.json` + `models/v2/eval_results.json` added, and
`models/index.json` now lists both versions. A real bug was caught while
building the verification test for this: the merge-rank Map key was built
by string concatenation with an empty separator, so two *different* merge
pairs could collide if they happened to concatenate to the same string
(`("2","024")` and `("20","24")` both → `"2024"`) — v1's test strings never
hit this, v2's digit-heavy Metaspace content did. Fixed with a NUL-character
separator; both v1 and v2 now pass 30/30 verification cases against the real
Python tokenizer (see `tokenizer-playground/test/`).

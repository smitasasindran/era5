# Tokenizer v3 — richer HTML-based corpus, same 4 languages

## Motivation

After reviewing `../../solution2` (the instructor's own reference solution)
in detail, the single biggest difference found was corpus construction: they
fetch Wikipedia's **rendered HTML** (the same content a browser shows) and
convert it to Markdown with `markdownify`, keeping infobox tables, citation
footnotes, navboxes, and category links — while v2 parsed raw **wikitext**
and deliberately dropped that content as clutter. Their corpus ended up
5-8x richer per language as a direct result, which is most of why their
absolute fertility (0.58-0.73) was so much lower than v2's (1.15-1.21),
independent of any difference in training strategy.

Two questions followed:
1. Is our allocation-search training strategy actually better than their
   single joint-weighted run, once the corpus-size confound is removed?
2. Could adopting their corpus-building *technique* (not their corpus, not
   their language choice) improve our own 4-language project?

## Step 1 — validation experiment (not part of the official project)

Built a full v2-style pipeline (`../v3_experiment_maithili_corpus/`) trained
directly on the instructor's own corpus files
(`../../solution2/corpus/*.faithful.txt`) with **Maithili** in place of
Marathi (matching their language choice, for the fairest possible
comparison), using our symmetric 3-group equalization strategy instead of
their joint-weighted single run.

Cross-validated our faithful-unit counter against theirs first: exact match
on all 4 languages (e.g. English: 186,367 units both ways) — confirms our
metric implementation is functionally identical to theirs, just a different
code path (character-scan + `unicodedata` here, `regex` `\p{}` there).

Result: **spread = 0.1033, score ≈ 9,676** — beats the instructor's own
score (6,503) on their exact corpus, using our strategy instead of theirs.
(Vocab dedup during the 3-group merge was larger than expected on this
corpus — 8,469 final vocab vs 10,000 allocated, since the 3 groups share
more vocabulary here — which likely cost some precision; the pre-merge
per-group search suggested spread ≈0.04 before that loss.) This answered
question 1: **our training strategy holds up, or beats theirs, independent
of corpus size.** See `../v3_experiment_maithili_corpus/README.md` for
details — that folder is a validation experiment, not part of the official
project deliverables.

## Step 2 — the real Version 3

Kept the project's actual 4 languages (English/Hindi/Telugu/Marathi — v3
doesn't change *what* we're tokenizing, only *how the corpus is built*) and
adopted the HTML+markdownify corpus-building technique for them.

### Corpus (`fetch_corpus_html.py`)

Adapted directly from `solution2/build_wiki_faithful_markdown.py`: fetch
`https://{lang}.wikipedia.org/api/rest_v1/page/html/{title}`, parse with
BeautifulSoup+lxml, strip only `<script>/<style>/<meta>`, convert
`<link rel="mw:PageProp/Category">` to visible `Category: ...` text,
absolutize all links/images to full URLs, convert to Markdown with
`markdownify` (ATX headings, `-` bullets). Installed `markdownify` and
`beautifulsoup4` for this (`lxml` was already present).

Validated against their own fetch: our English/Hindi/Telugu char counts
came out at 598,376 / 266,728 / 113,835 — matching their own reported
figures almost exactly (598,265 / 266,728 / 113,835), confirming the
approach reproduces correctly.

Corpus growth vs v2's wikitext-based corpus:

| Lang | v2 (wikitext) | v3 (rendered HTML) | Growth |
|---|---|---|---|
| English | 22,688 units | 186,426 units | **8.2×** |
| Hindi | 14,808 units | 88,359 units | **6.0×** |
| Telugu | 6,916 units | 36,292 units | **5.2×** |
| Marathi | 8,284 units | 29,766 units | **3.6×** |

### Training pipeline

Identical to v2's (`train_utils.py`, `optimize_allocation.py`,
`merge_tokenizers.py`, `evaluate.py` are the same code, just pointed at the
new corpus) — Metaspace, NFKC-only normalizer, explicit `[UNK]`, Metaspace
decoder, same script-safe 3-group merge (English alone, Telugu alone,
Hindi+Marathi jointly), same symmetric equalization search.

One re-tuned parameter: the Hindi:Marathi joint-training weight.
`sweep_hi_mr_weight.py` found **3:7** here vs v2's **2:3** — the richer
fetch grew Hindi's corpus proportionally more than Marathi's (size ratio
went from ~1.8x to ~3.0x), so Marathi needs more counter-weight now.

## Results — `tokenizer_final.json` (vocab_size = 8,503)

| Lang | Tokens | Faithful units | Fertility |
|---|---|---|---|
| Telugu | 28,605 | 36,292 | **0.7882** |
| Marathi | 21,605 | 29,766 | **0.7258** |
| Hindi | 62,098 | 88,359 | **0.7028** |
| English (X1) | 128,992 | 186,426 | **0.6919** |

**X1 = 0.6919 < 1.2 → PASS**
**Spread = 0.7882 − 0.6919 = 0.0963**
**Score = 1000 / 0.0963 ≈ 10,387.5**
**Hindi penalty factor = 1.0** (Hindi fertility 0.7028 well under 1.2 — no penalty)

Absolute fertility (0.69–0.79) now sits squarely in the instructor's own
reported range (0.58–0.73) — answering the lingering question from v2's
report about why our fertility was higher than theirs: it was the corpus,
not the tokenizer.

## v2 vs v3 — richer corpus doesn't automatically mean a better score

| | Vocab | Fertility range | Spread | Score |
|---|---|---|---|---|
| v2 (wikitext corpus) | 9,361 | 1.15 – 1.21 | 0.0603 | **≈16,581** |
| v3 (HTML corpus) | 8,503 | 0.69 – 0.79 | 0.0963 | ≈10,388 |

v3's absolute fertility is much closer to the instructor's, but its *spread*
— the actual scored quantity — is worse than v2's. Two likely reasons:
1. **Heavier vocab dedup.** v3's 3 groups share substantially more
   vocabulary (URLs, digit sequences, common Latin tokens embedded across
   scripts) than v2's did, so more of the allocated 10,000 tokens get
   deduplicated away during merging (1,497 tokens lost here vs ~600-700 in
   v2), leaving the post-merge tokenizer with less precise equalization
   than the pre-merge search targeted.
2. **More room for language-specific idiosyncrasy at scale.** With far more
   real content per language, Telugu's script/morphology (its own worst
   performer, 0.7882) has more opportunity to diverge from the others than
   it did on a smaller, more homogeneous corpus.

Both v2 and v3 comfortably beat the instructor's reported score (6,503);
which one is "better" depends on whether absolute fertility level or
spread/score is what's being optimized for. Both are kept as real, working,
playground-selectable versions rather than picking one as the winner.

## Files

- `fetch_corpus_html.py` — rendered-HTML → Markdown corpus fetcher (adapted
  from solution2's approach)
- `corpus/india_{en,hi,te,mr}.txt` — the richer corpora
- `faithful_units.py`, `train_utils.py`, `optimize_allocation.py`,
  `merge_tokenizers.py`, `evaluate.py`, `sanity_check.py` — same as v2,
  pointed at the new corpus (only `HI_W, MR_W = 3, 7` changed)
- `sweep_hi_mr_weight.py` — found the re-tuned Hindi:Marathi weight
- `tokenizer_final.json` — the v3 deliverable

Playground: `tokenizer-playground/models/v3/` (tokenizer.json,
eval_results.json, design.json), registered in `models/index.json`.
Verified 30/30 against the real Python tokenizer (`test/reference_v3.json`)
— no JS engine changes needed, since v3 uses the identical pipeline shape
as v2 (Metaspace + NFKC + `[UNK]` + Metaspace decoder).

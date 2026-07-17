# India Multilingual Tokenizer Playground

A static, dependency-free playground (like [tiktokenizer](https://tiktokenizer.vercel.app)
or HF's [tokenizer playground](https://huggingface.co/spaces/Xenova/the-tokenizer-playground))
for the custom BPE tokenizer built for the ERAv5 tokenizer assignment
(English/Hindi/Telugu/Marathi, 10,000-token shared vocab).

It loads a `tokenizer.json` (a real HuggingFace `tokenizers` fast-tokenizer
file, copied straight from the training pipeline in `../v1` or `../v2`) and
re-implements BPE encode in plain JS in `tokenizer.js` -- no server, no
bundler, no external tokenizer library. The JS port is verified byte-for-byte
against the real Python `tokenizers` output; see `test/`.

Two tokenizer pipelines are supported: v1's `Whitespace` pre-tokenizer +
`Lowercase`(-or-none) normalizer + `unk_token=None`, and v2's `Metaspace`
pre-tokenizer + `NFKC`+`Lowercase` normalizer + an explicit `[UNK]` token.
Anything outside those two throws a clear, specific error on load rather
than silently mis-tokenizing (see `tokenizer.js`'s `validate()`).

The page supports multiple tokenizer *versions* via a dropdown -- see
"Adding a new tokenizer version" below.

## Files

- `index.html`, `style.css`, `app.js` -- the UI
- `tokenizer.js` -- the BPE engine (normalizer + pre-tokenizer + merge logic),
  usable from both the browser (`<script>` tag) and Node (`require`)
- `models/index.json` -- the version registry: `[{id, label, dir}, ...]`.
  The dropdown is populated from this file; it's the only thing you need to
  edit to add/rename/reorder versions.
- `models/v1/tokenizer.json` -- the "Version 1" trained tokenizer.
  **Replace this file** if you retrain/update it (`cp ../v1/tokenizer_final.json models/v1/tokenizer.json`)
- `models/v1/eval_results.json` -- Version 1's official per-language fertility
  numbers and spread/score, shown in the page's "Official evaluation"
  section. Static snapshot, not computed live in the browser --
  `../v1/evaluate_hf.py` writes straight into this file by default
  (`python evaluate_hf.py tokenizer_final.json`, run from `../v1`), no manual
  copy step. Re-run it whenever `tokenizer_final.json` changes, or the
  numbers shown here will silently go stale.
- `models/v2/tokenizer.json`, `models/v2/eval_results.json` -- same idea for
  Version 2, produced by `../v2/merge_tokenizers.py` and `../v2/evaluate.py`
  respectively.
- `test/test_strings.json` -- shared list of test strings (plain text,
  multilingual mixes, punctuation, digits, empty/leading/trailing/multiple
  spaces, unrecognized characters) used to verify the JS port
- `test/generate_reference.py` -- runs those strings through the *real*
  Python `tokenizers` library for a given tokenizer.json, writing
  `{text, tokens, ids}` reference cases
- `test/compare_with_python.js` -- Node script that checks the JS port's
  output against those reference cases

## Run locally

No build step. Any static file server works, e.g.:

```bash
cd tokenizer-playground
python3 -m http.server 8080
# open http://localhost:8080
```

or `npx serve .`.

## Re-verify the JS port against Python

If you change `tokenizer.js`, add test cases, or add a new tokenizer version,
re-check correctness against the real tokenizer for each version:

```bash
# 1. Regenerate the Python reference cases (edit test/test_strings.json to add more)
python3 test/generate_reference.py models/v1/tokenizer.json test/reference_v1.json
python3 test/generate_reference.py models/v2/tokenizer.json test/reference_v2.json

# 2. Compare
node test/compare_with_python.js models/v1/tokenizer.json test/reference_v1.json
node test/compare_with_python.js models/v2/tokenizer.json test/reference_v2.json
```

This is how a real bug was caught during v2 development: the merge-rank
lookup used string concatenation with an empty separator, so two *different*
merge pairs that happened to concatenate to the same string (e.g. `("2",
"024")` and `("20","24")`, both → `"2024"`) collided in the same Map key.
v1's test strings never happened to trigger it; v2's digit-heavy Metaspace
content did. Fixed by using a NUL-character separator instead (see
`tokenizer.js`'s `SEP` constant).

## Adding a new tokenizer version

The dropdown reads `models/index.json`, a plain list of versions:

```json
[
  { "id": "v1", "label": "Version 1", "dir": "models/v1" }
]
```

To add a new version (say "Version 3"), using either supported pipeline:

```bash
mkdir tokenizer-playground/models/v3
cp your_tokenizer.json tokenizer-playground/models/v3/tokenizer.json
python evaluate.py your_tokenizer.json ../tokenizer-playground/models/v3/eval_results.json  # or evaluate_hf.py for v1's pipeline/metric
```

Then add one entry to `models/index.json`:

```json
[
  { "id": "v1", "label": "Version 1", "dir": "models/v1" },
  { "id": "v2", "label": "Version 2", "dir": "models/v2" },
  { "id": "v3", "label": "Version 3", "dir": "models/v3" }
]
```

No changes to `app.js`/`index.html` needed -- the dropdown, vocab badge,
download link, live tokenization, and the "Official evaluation" table all
switch together when the user picks a version, driven entirely by which
`dir` that version's entry points at.

This assumes the new tokenizer uses one of the two supported pipelines --
see the next section for what happens if it doesn't, and re-run the tests
in `test/` for the new version either way (new pipeline or not) before
trusting it.

## Swapping in a new tokenizer.json

**Short answer: it depends on what changed.**

`tokenizer.js` doesn't generically interpret every possible tokenizer.json --
it hand-implements exactly two pipelines:
1. A `BPE` model, `Lowercase`-or-none normalizer, `Whitespace` pre-tokenizer,
   `unk_token=None` (v1's `merge_tokenizers.py`)
2. A `BPE` model, a `Sequence` of `NFKC`+`Lowercase` (or either alone),
   `Metaspace` pre-tokenizer, an explicit `unk_token` (v2's `merge_tokenizers.py`)

It checks the JSON's declared types against an allowlist on load and
**throws a clear error instead of silently mis-tokenizing** if something
outside those two pipelines shows up.

| You changed... | Drop-in replacement? |
|---|---|
| Vocab size, corpus, quotas, language weights -- anything that reruns either `merge_tokenizers.py` as-is | **Yes.** Only the `vocab`/`merges` data changes; the pipeline stays the same. Just copy the new `tokenizer_final.json` into the right `models/<id>/` folder (or add it as a new version -- see above). |
| Pre-tokenizer other than `Whitespace`/`Metaspace` (e.g. `ByteLevel`, a `Sequence` of several) | **No.** Throws on load naming the exact unsupported type. You'd need to add a branch in `BPETokenizer.preTokenize()`. |
| Merge strategy / model type (e.g. `Unigram`, `WordPiece` instead of `BPE`) | **No.** These use fundamentally different encode algorithms (Unigram: Viterbi over a probability-weighted trie; WordPiece: greedy longest-prefix-match) and different JSON fields (Unigram's `model.vocab` is a list of `[piece, score]` pairs, not a dict, and it has no `merges` field at all). Throws on load rather than crashing confusingly on `model.merges.forEach`. |
| Normalizer step other than `NFKC`/`Lowercase` (e.g. `NFC`, `Strip`) | **No.** Throws on load naming the unsupported step, even inside a `Sequence` that also has supported steps. |

If you hit one of the "No" cases, the playground will fail to load and print
exactly which field wasn't recognized (check the badge / token view, or the
browser console) -- extend `BPETokenizer.normalize()` / `.preTokenize()` /
the constructor's model handling, and the allowlists near the top of
`tokenizer.js`, to add support.

## Deploy to Netlify

This is a plain static site -- `netlify.toml` sets the publish directory to
`.` and there's no build command needed.

- **Drag-and-drop**: zip/drag this folder into the Netlify dashboard.
- **Git-connected**: point a Netlify site at this repo with base directory
  `Session2/tokenizer-playground`, no build command, publish directory `.`.

## Notes on behavior

- Both versions' normalizers include `Lowercase` -- the displayed/tokenized
  text is always lowercase, even if you type mixed case (matches what the
  real tokenizer actually encodes to). v2 additionally applies `NFKC` first
  (Unicode compatibility normalization -- e.g. `ﬁ` → `fi`, fullwidth forms →
  standard forms -- before lowercasing).
- **v1 has `unk_token=None`.** Verified against Python: any character with no
  vocab entry (e.g. `!`, which never appeared in the India-Wikipedia plain-text
  training corpus) is **silently dropped** from the real token/id output --
  not replaced with a placeholder. The UI still renders such characters with
  a dashed outline so they don't just invisibly disappear, but they're
  excluded from the token count, ids, and fertility calculation, matching
  the real tokenizer exactly.
- **v2 has an explicit `unk_token="[UNK]"`.** Verified against Python: an
  unrecognized character becomes a real, visible, counted `[UNK]` token
  (e.g. encoding "中文" produces two literal `[UNK]` tokens, one per
  character) -- a genuinely different behavior from v1's silent drop, not
  just a different vocab. The UI reflects this automatically: `encodeAtom()`
  returns a real `id` (the `[UNK]` vocab entry's id) rather than `null`, so
  it renders and counts like any other token.
- v2's `Metaspace` pre-tokenizer keeps a leading space (`▁`) attached to each
  word rather than splitting punctuation off separately (v1's `Whitespace`
  does the latter) -- this is *why* v2 can push fertility below 1.0 (a whole
  `"word,"` can become one token) where v1 had a hard ~1.0 floor per atom.
  See `../v2/report.md` for the full mechanism.
- The "Fertility" stat above the token view is computed live from whatever
  text you type in the box. The "Official evaluation" table further down the
  page is a *different* thing -- a static snapshot of fertility measured
  against the full training corpora (see `eval_results.json` above), not
  live-recomputed from your input.
- The light/dark toggle (top right) persists your choice in `localStorage`
  and overrides the system `prefers-color-scheme` once you click it; before
  that, it follows your system setting automatically.

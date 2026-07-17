# India Multilingual Tokenizer Playground

A static, dependency-free playground (like [tiktokenizer](https://tiktokenizer.vercel.app)
or HF's [tokenizer playground](https://huggingface.co/spaces/Xenova/the-tokenizer-playground))
for the custom BPE tokenizer built for the ERAv5 tokenizer assignment
(English/Hindi/Telugu/Marathi, 10,000-token shared vocab).

It loads `tokenizer.json` (a real HuggingFace `tokenizers` fast-tokenizer
file, copied straight from the training pipeline in `..`) and re-implements
BPE encode in plain JS in `tokenizer.js` -- no server, no bundler, no
external tokenizer library. The JS port is verified byte-for-byte against
the real Python `tokenizers` output; see `test/`.

## Files

- `index.html`, `style.css`, `app.js` -- the UI
- `tokenizer.js` -- the BPE engine (normalizer + pre-tokenizer + merge logic),
  usable from both the browser (`<script>` tag) and Node (`require`)
- `tokenizer.json` -- the trained tokenizer. **Replace this file** if you
  retrain/update the tokenizer in `..` (`cp ../tokenizer_final.json tokenizer.json`)
- `eval_results.json` -- the official per-language fertility numbers and
  spread/score, shown in the page's "Official evaluation" section. This is a
  static snapshot, not computed live in the browser. `evaluate_hf.py` (in
  `..`) writes straight into this file by default --
  `python evaluate_hf.py tokenizer_final.json`, run from `..`, no manual copy
  step. Re-run it whenever `tokenizer_final.json` changes, or the numbers
  shown here will silently go stale (the page has no way to detect a
  mismatch on its own).
- `test/compare_with_python.js` -- Node script that checks the JS port's
  output against the real Python tokenizer for a set of test strings

## Run locally

No build step. Any static file server works, e.g.:

```bash
cd tokenizer-playground
python3 -m http.server 8080
# open http://localhost:8080
```

or `npx serve .`.

## Re-verify the JS port against Python

If you change `tokenizer.js` or swap in a new `tokenizer.json`, re-check
correctness against the real tokenizer:

```bash
# 1. Regenerate the Python reference cases (edit the test_strings list as needed)
python3 -c "
import json
from tokenizers import Tokenizer
tok = Tokenizer.from_file('tokenizer.json')
test_strings = ['This is a new day', 'भारत एक विशाल देश है।']  # add more
results = [{'text': s, 'tokens': tok.encode(s).tokens, 'ids': tok.encode(s).ids} for s in test_strings]
json.dump(results, open('/tmp/py_reference.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
"

# 2. Compare
node test/compare_with_python.js
```

## Swapping in a new tokenizer.json

**Short answer: it depends on what changed.**

`tokenizer.js` doesn't generically interpret every possible tokenizer.json --
it hand-implements exactly one pipeline: a `BPE` model, an optional
`Lowercase` normalizer, and a `Whitespace`-style pre-tokenizer (the pipeline
`merge_tokenizers.py` produces). It checks the JSON's declared types against
an allowlist on load and **throws a clear error instead of silently
mis-tokenizing** if something outside that pipeline shows up.

| You changed... | Drop-in replacement? |
|---|---|
| Vocab size, corpus, per-language quotas, hi:mr weight -- anything that reruns `merge_tokenizers.py` as-is | **Yes.** Only the `vocab`/`merges` data changes; the pipeline (BPE + Lowercase + Whitespace) stays the same. Just `cp ../tokenizer_final.json tokenizer.json`. |
| Pre-tokenizer (e.g. `Metaspace`, `ByteLevel`, a `Sequence` of several) | **No.** `tokenizer.js` always applies one hardcoded Whitespace-style regex regardless of what `pre_tokenizer.type` actually says. Before this validation was added it would have *silently* produced wrong atom boundaries; now it throws on load naming the exact unsupported type. You'd need to add a branch in `BPETokenizer.preTokenize()` for the new type. |
| Merge strategy / model type (e.g. `Unigram`, `WordPiece` instead of `BPE`) | **No.** These use fundamentally different encode algorithms (Unigram: Viterbi over a probability-weighted trie; WordPiece: greedy longest-prefix-match) and different JSON fields (Unigram's `model.vocab` is a list of `[piece, score]` pairs, not a dict, and it has no `merges` field at all). Throws on load rather than crashing confusingly on `model.merges.forEach`. |
| Normalizer (e.g. `NFC`, `Strip`, a `Sequence`) | **No.** Only `Lowercase` (or no normalizer) is implemented; anything else throws on load rather than being silently skipped. |

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

- The tokenizer's normalizer is `Lowercase` -- the displayed/tokenized text
  is always lowercase, even if you type mixed case (matches what the real
  tokenizer actually encodes to).
- This tokenizer has `unk_token=None`. Verified against Python: any
  character with no vocab entry (e.g. `!`, which never appeared in the
  India-Wikipedia training corpora) is **silently dropped** from the real
  token/id output -- not replaced with a placeholder. The UI still renders
  such characters with a dashed outline so they don't just invisibly
  disappear, but they are excluded from the token count, ids, and fertility
  calculation, matching the real tokenizer exactly.
- The "Fertility" stat above the token view is computed live from whatever
  text you type in the box. The "Official evaluation" table further down the
  page is a *different* thing -- a static snapshot of fertility measured
  against the full training corpora (see `eval_results.json` above), not
  live-recomputed from your input.
- The light/dark toggle (top right) persists your choice in `localStorage`
  and overrides the system `prefers-color-scheme` once you click it; before
  that, it follows your system setting automatically.

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

# Implementation Notes

Living log of concrete choices made while building the system described in
`DATALOADER_DESIGN.md`. Updated as each component lands. At the end, this
gets folded together with the design doc into the final submission README
— this file is the "what we actually did and why" half; the design doc is
the "what we planned" half.

## Status

| Component | Status | Key files |
|---|---|---|
| Shard Builder & Manifest Store | done | `tds/corpus.py`, `tds/tokenizer_utils.py`, `tds/manifest_store.py`, `tds/shard_builder.py`, `scripts/run_pipeline.py` |
| Eval / Test Firewall | not started | |
| Curriculum & Mixture Compiler | not started | |
| Cursor | not started | |
| OPUS Selector (stub) | not started | |
| Packer | not started | |
| Batch Assembler | not started | |
| Consumption / Learning Ledgers | not started | |
| Checkpoint / Crash / Resume | not started | |
| Replay / Fork | not started | |
| Audit / Evidence Bundle | not started | |

## 1. Shard Builder & Manifest Store

### Corpus

Two corpus sources, both under `data/corpus/`:

- **`small_shard.parquet`** — the real corpus, vendored in from
  `/work/courses/capstone_era4/Data-benchmark/small_shard.parquet` (852
  documents). Only `id`, `source`, `domain`, `language`, `text` are used;
  the file's other columns (`difficulty`, `has_code`, `agentic_score`, ...)
  are quality-scoring metadata from an external benchmarking process, not
  something a real admission pipeline would hand us, so `load_corpus()`
  reads exactly those 5 columns and nothing else — a file with extra
  columns just has them ignored.
- **`toy_corpus.parquet`** — 11 hand-authored documents (`tds/toy_corpus.py`),
  covering all 6 lanes with text short enough to read directly in that
  file. Exists purely so the shard builder / manifest pipeline can be run
  and its output verified by eye in under a second before trusting the
  same code against the larger real file.

One script runs the whole pipeline for either corpus:

```
python scripts/run_pipeline.py                    # real corpus (default)
python scripts/run_pipeline.py --corpus toy         # tiny fixture
python scripts/run_pipeline.py --corpus path/to/other.parquet
```

It always writes to the same `data/tokenizer/`, `data/shards/`,
`data/manifests/` — there is only ever one "current" build on disk, not a
parallel tree per corpus. Since the tokenizer is trained *from* whichever
corpus is passed in, switching corpora necessarily produces a new frozen
tokenizer (new `tokenizer_hash`), which would make any shards left over
from the *previous* corpus invalid alongside the new ones — and the
manifest store would (correctly) refuse to mix them in, since e.g.
`shard-000000` from a toy build and `shard-000000` from a real build have
different `content_hash` values, and that's exactly the kind of mutation
`ManifestStore.append()` exists to reject. So every run of
`run_pipeline.py` clears those three directories first, then rebuilds all
three fully from whichever corpus was specified. That's a
development/validation convenience for iterating on one corpus at a time —
not a relaxation of the immutability guarantee itself, which still holds
within any single build.

**Swapping in a different corpus file later:** drop a new parquet under
`data/corpus/` and pass `--corpus <path>`. Two things to know before doing
that:

- The file must have `id`, `source`, `domain`, `language`, `text` columns
  by those exact names (see `CORPUS_COLUMNS` in `tds/corpus.py`); anything
  else is ignored automatically.
- Duplicate `id` values in the source file are handled already and don't
  need fixing upstream: `document_id` (what everything downstream actually
  uses) is derived from row position, not from the raw `id` column. The raw
  `id` is kept separately as `source_document_id`, purely for provenance —
  fixing collisions there is a data-quality nicety, not a functional
  requirement of this pipeline.

### Capability lane mapping

`capability_lane_for(domain, language)` in `tds/corpus.py`:

- `language` wins over `domain` for Indic languages (`bn`, `mr`, `hi`, `gu`,
  `kn`, `ml`, `or`, `ta`, `pa`, `ne`, `as`, `te`, `sa`) — a Hindi
  stackexchange thread becomes `indic`, not `qa`, since indic is the
  scarce/protectable lane the mixture compiler will care about later, not
  its `qa` domain.
- Otherwise: `code` → `code`; `math`/`science` → `math_science`; `qa` →
  `qa`; `instruction` → `instruction`; everything else (`web`, `news`,
  `encyclopedia`, `social`, ...) → `general_web`.

On the real corpus this produces 6 lanes with a useful scarcity spread —
`qa` (343 docs) and `general_web` (297) dominate, `indic` (49) and
`instruction` (17) are minorities. That's deliberately left as-is rather
than rebalanced, since demonstrating protected floors later requires an
actually scarce lane.

### Tokenizer

A fresh byte-level BPE tokenizer trained specifically for this assignment
(not reused from Session 2's tokenizer work) — see
`tds/tokenizer_utils.py`:

- `vocab_size=8000`, special tokens `<pad>`, `<eos>`, `<unk>`.
- **Bug caught by tests, now fixed**: the trainer must be given
  `initial_alphabet=pre_tokenizers.ByteLevel.alphabet()` explicitly.
  Without it, the tokenizer only learns the 256-byte alphabet symbols that
  actually appear in the training corpus — so a byte sequence absent from
  training text (e.g. a script the corpus happens not to contain much of)
  falls back to `<unk>`, defeating the entire point of byte-level
  tokenization (every input should always be representable). Verified with
  a held-out Hindi sentence not present anywhere in training data.
- "Frozen" is enforced, not just documented: `load_frozen_tokenizer()`
  recomputes the sha256 of `tokenizer.json` on every load and raises if it
  doesn't match the hash recorded at training time. `train_tokenizer()` is
  only ever called from `scripts/build_tokenizer.py` (a one-time setup
  step) — nothing else in the pipeline can retrain it.

### Shard storage

- One shard = one capability lane's worth of documents up to
  `shard_token_budget` tokens (default 50,000); documents are never split
  across a shard boundary — a document that alone exceeds the budget still
  becomes its own (larger) shard.
- Each document's tokens end with `<eos>`; `document_spans` in the manifest
  record exactly which token range within the shard belongs to which
  document (this is what a later packer will use to derive `segment_id`
  without re-tokenizing anything).
- Stored as `.npy` (`uint16` — plenty for an 8000-entry vocab); asserted at
  build time that vocab size fits the chosen dtype.
- `content_hash` is `sha256` over the raw token bytes (`arr.tobytes()`),
  independent of the `.npy` container format, so the hash only reflects
  actual content.

### Manifest store

Append-only: `data/manifests/index.jsonl` (one line per shard, in creation
order) plus a per-shard `data/manifests/{shard_id}.json` for convenient
lookup. `ManifestStore.append()`:

- is a no-op if the exact same `(shard_id, content_hash)` is re-appended
  (unchanged re-runs of `build_shards.py` don't duplicate anything);
- raises `ManifestStoreError` if a `shard_id` is reused with a *different*
  `content_hash` — the design treats shards as immutable, so this would
  mean something mutated a shard in place, which should never happen.

Fields not populated from real data (yet): `license_tier`,
`dedup_status`, `contamination_status`, `eval_overlap_status` all get
honest placeholder values (`"unspecified"` / `"not_checked"` / `"none"`)
rather than fabricated ones, since neither corpus file carries that
information and this component deliberately doesn't re-implement an
admission pipeline (see design doc §8, scope assumptions).

### Real run

```
$ python scripts/run_pipeline.py
Cleared data/tokenizer, data/shards, data/manifests
Loaded 852 documents from data/corpus/small_shard.parquet
Trained tokenizer -> data/tokenizer/tokenizer.json
tokenizer_hash: sha256:ab94d8076f693546bd6fc30ec9b043f6e506749b6562694737743ccb68c2ca71
vocab_size: 8000

Built 47 shards, 2090730 tokens total
Lanes: ['code', 'general_web', 'indic', 'instruction', 'math_science', 'qa']
  code: 7 shards, 318974 tokens
  general_web: 11 shards, 491593 tokens
  indic: 5 shards, 220821 tokens
  instruction: 1 shards, 21881 tokens
  math_science: 3 shards, 128852 tokens
  qa: 20 shards, 908609 tokens
```

Confirmed stable across corpus switches: running with `--corpus toy`, then
plain (real corpus), then `--corpus toy` again each reproduces exactly the
same shard_ids/hashes/token counts as the first time that corpus was built
— the "clear, then rebuild" step doesn't introduce any nondeterminism.

### Tests (`tests/`, 29 total, `python -m unittest discover -s tests`)

- `test_corpus.py` — lane-mapping rules (pure function); a smoke test
  against the real vendored corpus (document count, no empty text, unique
  ids, expected lanes present, indic is actually scarce).
- `test_tokenizer_utils.py` — train/load roundtrip; byte-level really does
  avoid `<unk>` on unseen script; loading with no trained tokenizer fails
  clearly; loading a tampered `tokenizer.json` is detected and rejected.
- `test_manifest_store.py` — append/get, index + per-shard file both
  written, idempotent re-append, mutation rejected, reload-from-disk
  recovers prior state.
- `test_shard_builder.py` — hermetic synthetic corpus (own tokenizer
  trained in a temp dir, not the real one): multiple shards produced, every
  shard lane-homogeneous, every document in exactly one shard, spans
  contiguously partition each shard's tokens, every span ends in `<eos>`,
  on-disk shard hash matches manifest, an oversized single document still
  gets its own unsplit shard, and rebuilding from the same inputs is
  byte-for-byte deterministic.

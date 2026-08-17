# Implementation Notes

Living log of concrete choices made while building the system described in
`DATALOADER_DESIGN.md`. Updated as each component lands. At the end, this
gets folded together with the design doc into the final submission README
— this file is the "what we actually did and why" half; the design doc is
the "what we planned" half.

## Status

| Component | Status | Key files |
|---|---|---|
| Shard Builder & Manifest Store | done | `tds/corpus.py`, `tds/tokenizer_utils.py`, `tds/manifest_store.py`, `tds/shard_builder.py`, `tds/config.py`, `configs/*.yaml`, `scripts/run_pipeline.py` |
| Eval / Test Firewall | done | `tds/hashing.py`, `tds/eval_registry.py`, `tds/eval_firewall.py`, `configs/eval_registry*.yaml`, `scripts/build_eval_registry.py` |
| Curriculum & Mixture Compiler | done | `tds/mixture_compiler.py`, `tds/manifest_store.py` (`lane_token_totals`), `configs/curriculum*.yaml`, `scripts/compile_mixture.py` |
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

One script runs the whole pipeline, driven entirely by a config file
(`tds/config.py`, `PipelineConfig`) rather than a long CLI flag list:

```
python scripts/run_pipeline.py                                    # configs/pipeline.yaml (real corpus)
python scripts/run_pipeline.py --config configs/pipeline_toy.yaml   # tiny fixture
```

`--config` is the only remaining CLI flag. Every tunable — `corpus`,
`tokenizer_dir`, `shards_dir`, `manifests_dir`, `vocab_size`,
`shard_token_budget`, `packing_policy` — lives in the YAML file, so the
full parameter set for a given run is readable at a glance in one place
instead of reconstructed from a command line. `PipelineConfig.from_yaml()`
rejects unknown keys (catches typos) and fills in defaults for anything
omitted; `.resolved()` makes relative directory paths absolute against the
repo root regardless of the caller's current working directory. Two
ready-made profiles ship in `configs/`: `pipeline.yaml` (real corpus,
defaults) and `pipeline_toy.yaml` (the tiny fixture, small vocab/budget). A
one-off variation (e.g. trying `best_fit` packing) is a matter of copying
one of those files and pointing `--config` at the copy — not adding a new
flag to the script.

Regardless of which profile runs, it always writes to the config's
`tokenizer_dir`/`shards_dir`/`manifests_dir` — by default the same
`data/tokenizer/`, `data/shards/`, `data/manifests/` for every profile, so
there is only ever one "current" build on disk unless a config explicitly
names different directories. Since the tokenizer is trained *from*
whichever corpus the config names, switching corpora necessarily produces
a new frozen tokenizer (new `tokenizer_hash`), which would make any shards
left over from the *previous* corpus invalid alongside the new ones — and
the manifest store would (correctly) refuse to mix them in, since e.g.
`shard-000000` from a toy build and `shard-000000` from a real build have
different `content_hash` values, and that's exactly the kind of mutation
`ManifestStore.append()` exists to reject. So every run of
`run_pipeline.py` clears those three directories first, then rebuilds all
three fully from whichever corpus was specified. That's a
development/validation convenience for iterating on one profile at a time —
not a relaxation of the immutability guarantee itself, which still holds
within any single build.

**Swapping in a different corpus file later:** drop a new parquet under
`data/corpus/` and point a config file's `corpus` field at it (or add a new
`configs/*.yaml` profile for it). Two things to know before doing that:

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

### Packing policy (document -> shard grouping)

Two policies for how a lane's documents are grouped into shards, chosen via
`--packing-policy {greedy,best_fit}` on `run_pipeline.py` (default
`greedy`, matching original behavior) and recorded per-shard in the
manifest's `packing_policy` field:

- **`greedy`** (`tds.shard_builder._pack_greedy`) — preserves corpus
  arrival order, filling each shard until the next document would overflow
  it. Simple, order-faithful, but arrival order can interact badly with
  the budget: a run of items each just over half the budget forces a new
  shard per item even though pairs of them would fit together if reordered.
- **`best_fit`** (`tds.shard_builder._pack_best_fit`) — best-fit-decreasing:
  sort documents longest-first, then place each into whichever open shard
  has the least remaining room that still fits it, opening a new one only
  when nothing does. An oversized document (bigger than the whole budget)
  still gets an unshared shard under both policies — nothing else can fit
  into its negative remaining room.

Both policies share one invariant regardless of choice: no document is
ever split across a shard boundary, and a document appears in exactly one
shard's `document_spans`. One consequence worth knowing: the reported
utilization (`tokens / (shard_count * budget)`) can exceed 100% for a lane
whose only shard holds a single oversized document — the "capacity"
denominator assumes `budget`-sized shards, but an unsplit oversized
document is, correctly, bigger than that. Seen on the toy profile's
`instruction` lane (one 140-token document against a 90-token budget →
155.6%); not a bug, just a reminder that the metric assumes typically-sized
documents.

Real corpus, same tokenizer/corpus/budget, only the policy changed:

```
greedy:     47 shards -- e.g. general_web 89.4%, qa 90.9% of allocated shard capacity
best_fit:   45 shards -- general_web 98.3%, qa 95.6% of allocated shard capacity
```

Tests (`tests/test_packing_policies.py`) split into two levels:

- Pure algorithm tests on `_pack_greedy`/`_pack_best_fit` directly, with
  item *sizes* controlled exactly (not routed through a real tokenizer) so
  the combinatorial claims are exact: every item placed exactly once, no
  bin exceeds budget (unless a single oversized item forces it), an
  oversized item is always isolated under both policies, best-fit is
  deterministic, and a concrete adversarial-arrival-order case (alternating
  51/50-sized items against a 100 budget) where greedy is forced into 6
  isolated shards while best-fit packs them into fewer.
- Integration tests through `build_shards()` with a real (small, temp)
  tokenizer confirming the policy is correctly wired end to end: rejects an
  unknown policy name, manifests record the policy used, no lost/duplicated
  documents, spans still partition contiguously, on-disk hash still matches,
  and — on a realistic mixed-length synthetic corpus — best-fit never uses
  *more* shards than greedy.

### Real run

```
$ python scripts/run_pipeline.py
Config: configs/pipeline.yaml
  corpus=data/corpus/small_shard.parquet vocab_size=8000 shard_token_budget=50000 packing_policy=greedy
Cleared data/tokenizer, data/shards, data/manifests
Loaded 852 documents from data/corpus/small_shard.parquet
[PASS] eval_shard_blocked: 2 document(s) blocked
    doc-000000 (source_document_id=5e5a8b86a3ab7c221c7da1329fc9d0b0) matches benchmark=session6-real-holdout-v1 version=v1
    doc-000500 (source_document_id=4bddb69b053e4a43849a403e0de21d443be01202) matches benchmark=session6-real-holdout-v1 version=v1
Admitted 850/852 documents to the training pool
Trained tokenizer -> data/tokenizer/tokenizer.json
tokenizer_hash: sha256:32f4aec77e728a6613c014374d6b3e5efa81b301640aab4a9157783ebeb1df83

Built 46 shards (greedy packing), 2086631 tokens total
Lanes: ['code', 'general_web', 'indic', 'instruction', 'math_science', 'qa']
  code: 7 shards, 318903 tokens, 91.1% of allocated shard capacity
  general_web: 10 shards, 489429 tokens, 97.9% of allocated shard capacity
  indic: 5 shards, 219112 tokens, 87.6% of allocated shard capacity
  instruction: 1 shards, 21898 tokens, 43.8% of allocated shard capacity
  math_science: 3 shards, 128818 tokens, 85.9% of allocated shard capacity
  qa: 20 shards, 908471 tokens, 90.8% of allocated shard capacity
```

Note the shard/token counts here are naturally slightly lower than the
figures earlier in this section (e.g. 46 shards vs. 47, 2086631 tokens vs.
2090730) now that the eval firewall runs first and 2 real documents are
held out of the training pool -- see §2 below. The tokenizer_hash also
changed for the same reason: it's trained only on admitted documents, so
excluding held-out content changes what the tokenizer learns, as it
should.

Confirmed stable across corpus/profile switches: running with
`--config configs/pipeline_toy.yaml`, then plain (real corpus), then the
toy config again each reproduces exactly the same shard_ids/hashes/token
counts as the first time that profile was built — the "clear, then
rebuild" step doesn't introduce any nondeterminism.

### Tests (`tests/`, `python -m unittest discover -s tests`)

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
- `test_config.py` — `PipelineConfig` and `EvalRegistryConfig`: defaults,
  partial overrides, unknown-key rejection, missing-file handling, path
  resolution, and that the mutable-default `held_out_document_ids` list
  doesn't leak between separate config instances.

## 2. Eval / Test Firewall

### Why a separate registry, not a manifest field

The shard manifest already has an `eval_overlap_status` field (currently a
placeholder, `"none"`, per §1). That field alone can't *do* anything --
it's just a label someone would have to set correctly by hand. The actual
enforcement needs a second, independent object: a registry of held-out
content hashes that gets checked *before* a document is ever tokenized,
so a document never becomes a shard in the first place if it overlaps
something held out. `eval_overlap_status` remains as a manifest-level
summary field for now (still `"none"`, since anything that would have set
it otherwise was already filtered out upstream and never reached the
shard builder at all) -- the registry is where the real decision happens.

### Matching by content, not identity

`EvalRegistry` (`tds/eval_registry.py`) keys held-out entries by
`sha256(raw text)`, computed via the same `tds/hashing.py` helper the
shard builder already used for shard `content_hash` (factored out of
`tokenizer_utils.py` into its own module for exactly this kind of reuse).
`Document` (`tds/corpus.py`) now exposes `content_hash` as a computed
property, not a stored field -- it's a pure function of `text`, so there's
no way for it to drift out of sync with the document it describes.

Matching on content hash rather than `document_id`/`source_id` is the
point, not an implementation detail: a training corpus can accidentally
contain a benchmark's questions verbatim under a completely unrelated
source, and the firewall needs to catch that regardless of what id or
lane the duplicate happens to carry. Proven directly: `tds/toy_corpus.py`
now has a 12th document (`toy-0012`, source `toy_web_mirror`, lane
`general_web`) that is an exact-text duplicate of `toy-0005` (source
`toy_qa`, lane `qa`, document_id `doc-000004`, held out by
`configs/eval_registry_toy.yaml`). Both get blocked, even though only
`doc-000004` is named in the registry config:

```
[PASS] eval_shard_blocked: 2 document(s) blocked
    doc-000004 (source_document_id=toy-0005) matches benchmark=session6-toy-holdout-v1 version=v1
    doc-000011 (source_document_id=toy-0012) matches benchmark=session6-toy-holdout-v1 version=v1
```

### Where the firewall runs

`filter_training_documents()` (`tds/eval_firewall.py`) runs immediately
after `load_corpus()` in `scripts/run_pipeline.py`, before anything else
touches the document list. It returns `(admitted, blocked)`; the original,
unfiltered `documents` list is never referenced again in the script --
**both** the tokenizer trainer and the shard builder are called with
`admitted` only. This matters for the tokenizer specifically: if it were
trained on the full document list, held-out text would still leak into
the tokenizer's learned vocabulary even though no shard ever contained it
-- a subtler form of the same contamination the firewall exists to
prevent. Training the tokenizer only on admitted documents closes that
gap.

### Registry persistence vs. pipeline artifacts

`data/eval_registry/` is deliberately **not** cleared by `run_pipeline.py`
the way `data/tokenizer/`, `data/shards/`, and `data/manifests/` are.
Those three are tightly coupled to one specific corpus build (a shard is
only valid under the exact tokenizer that produced it), so switching
corpora invalidates all three together. The eval registry has the opposite
lifecycle: it's meant to accumulate held-out fingerprints across
benchmarks -- and across corpora -- over time, via repeated runs of
`scripts/build_eval_registry.py` (one config per benchmark; see
`configs/eval_registry.yaml` for the real corpus, `configs/eval_registry_toy.yaml`
for the toy fixture). Running both against the same `data/eval_registry/`
accumulates 3 entries total, and each corpus's pipeline run only ever
matches its own content -- registering the toy benchmark doesn't cause any
false blocks when later running the real-corpus profile, since the hashes
simply don't collide.

`EvalRegistry.register_text()` is idempotent for an unchanged
re-registration (same text, same benchmark_id) and raises
`EvalRegistryError` if a content hash already claimed by one benchmark is
registered again under a *different* benchmark_id -- deliberately strict,
since a silent overwrite there would be exactly the kind of un-auditable
mutation the manifest store's own mutation guard (§1) exists to prevent
for shards.

One consequence worth flagging: because the registry isn't rebuilt by
`run_pipeline.py`, a fresh clone of this repo has to run
`build_eval_registry.py` for each profile at least once before the
firewall has anything to block (an empty registry is a valid state --
"nothing has been flagged yet" -- so the pipeline still runs, it just
blocks nothing). The eventual `run_demo.py` needs to call
`build_eval_registry.py` for every profile it demonstrates before calling
`run_pipeline.py`, or the demo's evidence for this requirement would be
empty. Noted here so it isn't forgotten when that script is assembled.

### Real corpus run

Held out `doc-000000` (`C4`, `general_web`) and `doc-000500`
(`sangraha_mr`, `indic`) -- deliberately two different lanes, so the demo
doesn't look like it only works for one capability lane:

```
$ python scripts/build_eval_registry.py --config configs/eval_registry.yaml
Registered doc-000000 -> sha256:debdc15b60197cac6d87a0f83b4b74a527fa11b4804bd48038a034e3f6928eda under benchmark 'session6-real-holdout-v1'
Registered doc-000500 -> sha256:f3784bc05c9cfa34bb8ed888b9d2dd05a4fd0aaa638e5fe960386240ca0c6ce4 under benchmark 'session6-real-holdout-v1'

Eval registry at data/eval_registry now holds 2 held-out hash(es) total.
```

Then `run_pipeline.py` (see updated §1 "Real run" above): 850/852
documents admitted, 2 blocked, neither blocked `document_id` appears in
any shard's `document_spans` in the resulting `data/manifests/index.jsonl`
(checked directly, not just asserted).

### Tests

- `test_eval_registry.py` — register/get, idempotent re-registration,
  registering the same hash under a *different* benchmark is rejected,
  reload-from-disk recovers prior entries, accumulates correctly across
  multiple distinct benchmarks.
- `test_eval_firewall.py` — empty registry admits everything; a document
  matching the registry is blocked and excluded from `admitted`; **the
  concrete content-vs-identity case**: two documents with identical text
  but different id/source/lane are both blocked while an unrelated third
  document is admitted; admitted-list order is preserved (only blocked
  entries are removed, nothing is reordered).
- `test_eval_firewall_integration.py` — full path: register a held-out
  text, filter a small document set, then actually run the *real*
  `build_shards()` on the admitted set and assert neither blocked
  `document_id` appears in any shard's `document_spans` -- not just that
  the firewall's return value looks right in isolation.

## 3. Curriculum & Mixture Compiler

### What "compiling" actually means here

The design doc's stage schema (`stage`, `token_start`/`token_end`,
`sequence_length`, `mixture`, `protected_floors`, `warmup_tokens`) is
human-authored intent. The compiler's job is to check that intent against
reality -- how many tokens each capability lane's shards actually contain
(`ManifestStore.lane_token_totals()`, a small new aggregate method) -- and
produce a `CompiledSchedule` that can answer "what mixture applies at step
N" without re-deriving anything at query time. Three things make this more
than bookkeeping:

1. **Scarcity is tracked cumulatively across stages, not per-stage in
   isolation.** A lane's shard supply is one shared resource spent across
   the whole curriculum. A stage can look perfectly satisfiable checked
   alone and still be scarce once you add in what earlier stages already
   drew from the same lane -- see `test_second_stage_accounts_for_first_stages_consumption`
   for the concrete case (700 + 500 tokens requested cumulatively from a
   lane with only 1000 available; each stage alone looks fine, together
   they don't).
2. **Protected floors are honored via repetition, not just accounting.**
   Under `reduce_share`, a lane's share above its floor gets cut when
   supply runs out, but the floor itself is still met even if that requires
   dipping back into already-used tokens (`repeat_factor > 1` on that
   lane's `LanePlan`). The floor is a commitment about the training
   stream's composition, not a promise of never repeating.
3. **Shortfalls are reported, never silently redistributed.** When a
   stage's total effective mixture doesn't reach 100% (`unallocated_share`
   on `CompiledStage`), the compiler does not proportionally reallocate the
   gap to other lanes. The design doc frames the scarcity response
   (repeat / synthesize / reduce / defer) as an *operator* decision; the
   compiler applies whichever policy is configured and reports the
   remainder honestly rather than papering over it with an opaque
   redistribution that could itself violate some other lane's floor.

### A real bug the real numbers caught

Step ranges were originally computed per-stage as
`token_position // (sequence_length * global_batch_size)` -- fine as long
as every stage shares the same `sequence_length`, but `configs/curriculum.yaml`'s
`anneal` stage deliberately uses a longer one (256 vs. 128, matching the
design's "long-context packing is handled separately"). Compiling it
produced `anneal: steps [732, 927)` immediately following
`capability-expansion: steps [781, 1464)` -- overlapping, and running
backwards. The fix: `step_start` for a stage is a running counter carried
forward from the previous stage's `step_end` (`compile_curriculum`'s
`step_cursor`), never re-derived from absolute token position. Token
ranges tile contiguously by construction (`_validate_stage_ordering`);
step ranges have to be *derived* to tile contiguously too, since the
token-to-step ratio can change between stages. Caught immediately by
actually running the compiler against real shard counts rather than only
against small hand-picked test numbers — `test_steps_stay_monotonic_when_sequence_length_changes_between_stages`
now guards it directly.

### A second bug: a tautological scarcity check

The first cut of the protected-floor-unmeetable check compared
`effective_tokens < floor_tokens`, but `effective_tokens` is defined as
`max(remaining_capacity, floor_tokens)` -- which by construction can never
be less than `floor_tokens`. The check could never fire. The real
"impossible" condition isn't about `effective_tokens` at all: a floor can
always be met via repetition as long as *any* tokens exist for that lane,
however large the resulting `repeat_factor`; it's genuinely impossible
only when `available == 0` -- there is nothing at all to repeat. Fixed to
check that directly; `test_floor_unmeetable_even_by_itself_raises` and
`test_floor_is_honored_even_beyond_remaining_capacity` cover both sides of
the line.

### Shared `data/manifests/`, same gotcha as run_pipeline.py

`configs/curriculum.yaml` and `configs/curriculum_toy.yaml` both default
`manifests_dir` to `data/manifests` -- the same shared directory
`run_pipeline.py` builds into. `compile_mixture.py` reads whatever corpus's
shards are *currently* built there; it does not build anything itself. So
demoing the toy curriculum requires running
`run_pipeline.py --config configs/pipeline_toy.yaml` first, or
`compile_mixture.py` will silently check the toy stages' tiny token
budgets against the real corpus's (much larger) supply and report no
scarcity at all -- not wrong, just not the demonstration intended. Same
consideration as `data/eval_registry/` in §2, noted here so the eventual
`run_demo.py` sequences these correctly per profile.

### Real corpus run

`configs/curriculum.yaml` deliberately plans for more tokens than some
lanes can supply -- instruction, code, math_science, and indic all become
scarce at some point across 3 stages:

```
$ python scripts/compile_mixture.py
Available supply: {'code': 318903, 'general_web': 489429, 'indic': 219112, 'instruction': 21898, 'math_science': 128818, 'qa': 908471}

Stage 'foundation': steps [0, 781), span 800000 tokens, sequence_length=128
    instruction    target=3.00% effective=2.74% [scarce_reduced]
    (all other lanes ok)
    [WARN] unallocated_share=0.26% of this stage is unmet

Stage 'capability-expansion': steps [781, 1464), span 700000 tokens, sequence_length=128
    code           target=25.00% effective=22.70% [scarce_reduced]
    instruction    target=5.00% effective=2.00% [scarce_reduced], repeat_factor=1.64x
    math_science   target=20.00% effective=6.97% [scarce_reduced]
    (general_web, indic, qa ok)
    [WARN] unallocated_share=18.33% of this stage is unmet

Stage 'anneal': steps [1464, 1659), span 400000 tokens, sequence_length=256
    indic          target=35.00% effective=20.00% [scarce_reduced], repeat_factor=1.26x
    math_science   target=30.00% effective=0.00% [scarce_reduced]
    (general_web, qa ok)
    [WARN] unallocated_share=45.00% of this stage is unmet
```

`math_science` reaching effective share **0%** in `anneal` (it has no
floor there, and its supply was already fully claimed by
`capability-expansion`) next to `indic` still getting its full floor of
20% (protected) in the same stage is the clearest single side-by-side
illustration of what a protected floor actually buys a lane.

### Tests (`tests/test_mixture_compiler.py`, plus `ManifestStore`/`CurriculumConfig` additions)

- Stage validation: rejects no stages, weights not summing to 1.0, a floor
  exceeding its lane's weight, a floor for a lane absent from `mixture`,
  floors summing over 100%, non-contiguous stages, overlapping stages,
  out-of-order stages, zero/negative span.
- Ample supply: every lane comes back `"ok"` with `effective_weight ==
  target_weight`; step range is a plain floor-division sanity check.
- All three scarcity policies (`repeat`, `reduce_share`, `defer`) each
  get their own behavior verified, including both `reduce_share` floor
  edge cases described above.
- Cumulative scarcity across stages (the case in point 1 above), including
  under `repeat` specifically (a stage that looks fine alone still comes
  back `scarce_repeat` once prior stages' demand is added in).
- The step-numbering regression case described above, as a standing test.
- `mixture_at_step`/`stage_at_step`: out-of-range steps raise; the first
  stage never ramps even with `warmup_tokens` set; mixture at a stage's
  first step equals the *previous* stage's mixture; mixture once past the
  warmup window equals the *current* stage's; the midpoint of a warmup
  window is an exact linear blend.
- `ManifestStore.lane_token_totals()`: sums correctly across shards sharing
  a lane, empty for a fresh store.
- `CurriculumConfig` (in `test_config.py`): YAML `stages` parses into real
  `MixtureStage` instances (not raw dicts), unknown top-level *and*
  per-stage keys are both rejected, the mutable-default `stages` list
  doesn't leak between instances, `.resolved()` makes `manifests_dir`/
  `output_path` absolute while leaving `stages` itself untouched.

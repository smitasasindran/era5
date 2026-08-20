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
| Cursor | done, revised (see §5) | `tds/cursor.py`, `tds/manifest_store.py` (`document_pool_by_lane`) |
| OPUS Selector (stub) | not started | |
| Packer | done | `tds/packer.py`, `tds/shard_builder.py` (`response_start_token`) |
| Batch Assembler | done | `tds/batch_assembler.py` |
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
`run_pipeline.py` builds into. Run standalone, `compile_mixture.py` reads
whatever corpus's shards are *currently* built there; it does not build
anything itself, so demoing the toy curriculum this way still requires
running `run_pipeline.py --config configs/pipeline_toy.yaml` first, or it
will silently check the toy stages' tiny token budgets against the real
corpus's (much larger) supply and report no scarcity at all -- not wrong,
just not the demonstration intended. Same consideration as
`data/eval_registry/` in §2.

**This gotcha is closed for the common case.** `PipelineConfig` gained a
`curriculum` field (a path to a `CurriculumConfig` YAML, blank to skip);
both `configs/pipeline.yaml` and `configs/pipeline_toy.yaml` set it to
their matching curriculum file. `run_pipeline.py` now compiles and freezes
the mixture schedule as its last step, and does so against
`config.manifests_dir` -- the manifests *that same run just built* --
overriding whatever `manifests_dir` the curriculum YAML itself names.
`compile_mixture.py` still exists standalone (for recompiling a curriculum
against shards already on disk, without rebuilding them), sharing the
report-printing logic (`print_schedule_report`) with `run_pipeline.py` so
the two never drift into differently-formatted output. `python scripts/run_pipeline.py [--config ...]` is now the single command that runs the whole pipeline end to end.

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

### Freezing the compiled schedule (hash-pinned, like the tokenizer)

`compile_mixture.py` originally wrote the schedule with a bare
`json.dump(dataclasses.asdict(schedule))`. That's fine for a human to read,
but it's the wrong shape for what this artifact is actually for: per
`DATALOADER_DESIGN.md`, the training stream must be a pure function of
*frozen* manifests + mixture config + seed + step, for resume/replay to
hold. If a downstream consumer (the not-yet-built Cursor) read the mixture
config by re-running `compile_curriculum()` against `data/manifests/` *at
resume time*, that's not actually frozen -- if shards were rebuilt or added
between the original run and a later resume, `lane_token_totals()` would
return different numbers and scarcity resolution could come out
differently, silently changing which lane a given step draws from. That's
exactly the kind of drift the frozen-tokenizer pattern
(`tokenizer_utils.load_frozen_tokenizer`) already exists to prevent for the
vocabulary; the schedule needed the same treatment.

Added to `tds/mixture_compiler.py`:

- `freeze_schedule(schedule, schedule_path)` -- writes the schedule JSON,
  then writes a sibling manifest (`mixture_schedule_manifest.json`, next to
  `mixture_schedule.json`) containing its `schedule_hash` (sha256 of the
  schedule file's bytes) plus `global_batch_size`/`scarcity_policy`/stage
  names for quick inspection without loading the full schedule.
- `load_frozen_schedule(schedule_path)` -- recomputes the hash and compares
  against the manifest before doing anything else; raises `ValueError` on
  mismatch (tampering, or a schedule file edited by hand), `FileNotFoundError`
  if either file is missing. Only on a hash match does it reconstruct the
  nested dataclasses (`MixtureStage` / `LanePlan` / `CompiledStage` /
  `CompiledSchedule`) from the parsed JSON and hand back a fully
  functional `CompiledSchedule` -- `mixture_at_step`/`stage_at_step` work
  identically on a loaded schedule as on one still in memory from
  `compile_curriculum()`.

`compile_mixture.py` now calls `freeze_schedule()` instead of dumping JSON
directly, and prints the resulting `schedule_hash`. The manifest file sits
alongside the schedule under `data/`, so `.gitignore`'s
`data/mixture_schedule.json` / `data/mixture_schedule_toy.json` entries
were collapsed into one glob (`data/mixture_schedule*.json`) to cover both
new manifest files too.

This doesn't change anything about *how* the schedule is compiled -- it's
purely about how it's handed off. The actual Cursor, when built, should
call `load_frozen_schedule()` rather than `compile_curriculum()` directly.

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
- `freeze_schedule`/`load_frozen_schedule`: round-trip through disk
  reproduces the same `mixture_at_step`/`stage_at_step` behavior as the
  in-memory schedule and the same `schedule_hash`; loading with no frozen
  schedule present raises `FileNotFoundError`; a single appended byte in
  the schedule file after freezing is detected and raises `ValueError`.
- `PipelineConfig.curriculum` (in `test_config.py`): a relative path is
  resolved absolute the same way the existing dir fields are; a blank
  (default) value is left untouched rather than resolving to `root` itself.

## 4. Cursor

### What a "candidate" actually is

`DATALOADER_DESIGN.md` §5.4 gives the cursor as pseudocode:

```python
def cursor(run_config, global_step):
    stage = lookup_stage(run_config.curriculum, global_step)
    lane = pick_lane(run_config.seed, stage, global_step)
    k = picks_so_far_in_lane(run_config.seed, stage, lane, global_step)
    shard_id, offset = lane_sequence(run_config.seed, lane)[k]
    return CandidateItem(lane, shard_id, offset)
```

Two things needed pinning down before this could become real code:

1. **What granularity does one candidate point at?** §5.6 (Packer) says the
   packer "fills fixed-length sequences *from* accepted candidates" -- the
   packer produces fixed-length windows, candidates are its raw material,
   not already-cut windows themselves. So `(shard_id, offset)` is a
   **document's** location: `offset` is that document's `start_token`
   within the shard, taken straight from the shard's own manifest
   `document_spans` (already built by the Shard Builder). This also fully
   decouples the cursor from `sequence_length`, which varies by stage
   (128 vs. 256 in `configs/curriculum.yaml`) -- windowing is entirely the
   (not-yet-built) Packer's job.
2. **What does `global_step` alone leave out?** `mixture_compiler.py`
   already fixed `tokens_per_step = sequence_length * global_batch_size`,
   i.e. one step = one full batch of `global_batch_size` sequences -- so a
   single call can't return one `CandidateItem` per step, it needs a
   **slot** within the step's batch too. `tds/cursor.py` flattens
   `(global_step, slot)` into `slot_index = global_step * global_batch_size
   + slot` and works in terms of that flat index internally.

### Lane pools

Added `ManifestStore.document_pool_by_lane()`: every document across every
shard, grouped by `capability_lane`, as `(shard_id, document_id,
start_token)` triples, sorted by `(shard_id, start_token)` before any
shuffling -- so the *unshuffled* base order is itself reproducible rather
than depending on dict or filesystem iteration order. This is the raw pool
`lane_sequence` permutes.

### How a lane's turn is picked

The pseudocode's `pick_lane` is a "deterministic weighted pick" without
saying how. Two real options were considered:

- **Live weighted random sampling** (a single `random.Random` stream,
  advanced one call at a time) -- rejected: it's stateful, so answering
  "what would slot 47,000 be" means replaying every draw before it in
  order, which is exactly the kind of hidden iterator state the whole
  design exists to avoid.
- **Exact proportional round-robin** (give each lane its precise target
  share every N draws) -- rejected: `mixture_at_step` changes *every step*
  during a warmup ramp, so there's no simple closed form for "how many of
  the last N draws should this lane have gotten" once the target itself is
  a moving function of step.

What's implemented instead: `pick_lane(seed, weights, global_step, slot)`
computes a uniform value in `[0, 1)` via SHA-256 of `(seed, global_step,
slot)` -- a counter-based hash, not a stream -- and compares it against the
mixture's cumulative weight thresholds. Any `(step, slot)` can be evaluated
in complete isolation, in any order, with no history. The cost: realized
lane shares only *converge* to the target mixture over many draws rather
than exactly matching it immediately -- an honest, reproducible trade-off,
and one that gives the eventual "planned vs. actual shares" audit
something real to report rather than a trivial exact match.

**Renormalizing under scarcity:** if a stage's effective mixture doesn't
sum to 1.0 (`unallocated_share > 0`, e.g. `toy-expansion` at 25.67%
unmet -- see §3), `pick_lane` renormalizes across only the lanes with
positive weight. A real batch slot can't literally be "left empty" for the
unmet fraction; the compiler's `unallocated_share` stays a legitimate
pre-flight warning for the operator, not something the cursor tries to
paper over silently -- it just still has to hand back *some* real document
for every slot.

### `picks_so_far_in_lane`: O(n) replay, not a closed form

Unlike proportional round-robin, counting "how many of the hash-based
picks so far landed on lane L" has no shortcut -- it genuinely requires
enumerating them. `iter_candidates(seed, schedule, lane_pools)` does this
the efficient way: a single forward generator maintaining a running
per-lane counter (O(1) amortized per slot), rather than replaying from
slot 0 at *every single slot* (which would be O(n) per slot, O(n^2)
overall for consuming a whole run). `cursor(seed, schedule, lane_pools,
global_step, slot)` is a thin point-query convenience built from the same
generator (one `itertools.islice`) -- used for one-off lookups and tests,
not for bulk consumption.

Resume and replay both reduce to the same operation here: start
`iter_candidates` fresh from slot 0 and advance it to the point of
interest. That's a real O(n) computation, not O(1) -- but at this
project's scale (the real corpus's compiled schedule tops out at 1,659
steps x 8 batch = 13,272 slots) it's a sub-second replay, not the kind of
cost the "pure function, no persisted state" principle was trying to avoid
(replaying a live shuffle buffer's mutation history, or restoring
serialized iterator internals).

### Verification against the real and toy manifests

Ran against the actual frozen `data/manifests/` + `data/mixture_schedule*.json`
for both profiles (toy corpus for human-checkable output, then the real
corpus restored as the resting state afterward, per the usual protocol --
see §3's shared-`data/manifests/` note):

```
$ python -c "... iter_candidates('demo-seed-1', schedule, lane_pools) ..."
lanes with pools: {'code': 2, 'general_web': 2, 'indic': 2, 'instruction': 1, 'math_science': 2, 'qa': 1}
CandidateItem(global_step=0, slot=0, lane='code', shard_id='shard-000000', document_id='doc-000003', token_offset=23, pick_index=0, epoch=0)
CandidateItem(global_step=0, slot=1, lane='code', shard_id='shard-000000', document_id='doc-000002', token_offset=0, pick_index=1, epoch=0)
CandidateItem(global_step=1, slot=1, lane='qa', shard_id='shard-000008', document_id='doc-000005', token_offset=0, pick_index=0, epoch=0)
CandidateItem(global_step=2, slot=0, lane='qa', shard_id='shard-000008', document_id='doc-000005', token_offset=0, pick_index=1, epoch=1)
...
point query step=5 slot=1 matches bulk iteration: OK
replay of steps slice [10:20) matches original: OK
```

`qa`'s toy-corpus pool has exactly one document -- every draw from `qa`
after the first is a repeat with an incrementing `epoch`, visibly matching
the `scarce_reduced`/`repeat_factor` story the mixture compiler already
reported for that lane. The point-query-vs-bulk and replay-slice checks
are exactly the crash/resume and replay guarantees the assignment asks the
final demo to prove, exercised here in miniature before the Batch
Assembler/Checkpoint/Resume machinery that will do it for real exists.

### Tests (`tests/test_cursor.py`, plus `ManifestStore` additions)

- `pick_lane`: deterministic for identical inputs; never returns a
  zero-weight lane across 200 draws; raises when no lane has positive
  weight; renormalizes correctly when weights sum below 1.0 (a
  single-lane mixture at weight 0.3 is still picked every time); different
  seeds diverge.
- `lane_sequence`: deterministic for identical inputs; is a true
  permutation of the pool (same multiset, different order); different
  epochs reshuffle; raises on an empty pool.
- `iter_candidates`: the first `len(pool)` items for a single-lane
  schedule cover the pool exactly once (`epoch=0`, `pick_index` 0..n-1);
  the next `len(pool)` items are the same documents reshuffled (`epoch=1`);
  stops exactly at `schedule.total_steps * global_batch_size` items.
- `cursor`: a point query matches the corresponding item from bulk
  iteration exactly; an out-of-range `slot` is rejected.
- Replay determinism (the property this component exists to prove): two
  independent fresh calls to `iter_candidates` with the same
  `(seed, schedule, lane_pools)` produce an identical stream; slicing out
  a historical interval from a fresh generator matches that same interval
  from the original run; a different seed diverges somewhere in the
  stream.
- Scarcity awareness: under `scarcity_policy="defer"`, the deferred lane
  (`effective_weight == 0`) is never drawn.
- `ManifestStore.document_pool_by_lane()`: groups and flattens spans
  correctly by lane; sorted by `(shard_id, start_token)` regardless of
  append order; empty for a fresh store.

## 5. Packer (and a correction to §4's Cursor)

### Building the Packer exposed a real mismatch in the Cursor's API

§4's `iter_candidates`/`CandidateItem`/`picks_so_far_in_lane` assumed
exactly one document per (global_step, slot) -- i.e. one document = one
packed sequence. That assumption doesn't survive contact with the Packer:
DATALOADER_DESIGN.md §5.6 says the packer "fills fixed-length sequences
*from* candidates" -- plural candidates per sequence. A short document
doesn't fill a whole window by itself (several get concatenated); a long
document can overflow one window and spill into the next. Neither case fits
"one candidate maps to one output sequence."

**Fix:** split what was one concept into two genuinely separate ones,
both still in `tds/cursor.py`:

- `pick_lane` / `lane_for_slot` / `iter_lane_assignments` -- *which lane*
  does output sequence `(global_step, slot)` draw from. This part was
  always correct: it never depended on document lengths, only on
  `(seed, global_step, slot)` and the compiled mixture. Simplified as a
  result -- no cumulative pick-count bookkeeping needed at all, since the
  decision has no history dependency. `iter_candidates`'s O(n)
  `picks_so_far_in_lane` replay is gone; `lane_for_slot` is now a direct
  O(1) point query.
- `lane_document_stream` -- an *infinite* per-lane document generator
  (permuted, reshuffled each full cycle), advanced by the Packer at
  *its own pace* (one step per document actually consumed while filling a
  window), not once per slot.

`CandidateItem` is gone -- nothing needed a standalone "one document
assigned to one sequence" record once the Packer owns document consumption
directly. `tests/test_cursor.py` was rewritten against the new,
smaller surface; the replay-determinism and scarcity-awareness properties
it proved before still hold (see its test list below), now proved against
the simpler API.

### response_start_token: why prompt/response must be tokenized separately

Structure-preserving packing (see below) needs to know exactly which
tokens are "prompt" and which are "response" for a role-tagged document.
The tempting shortcut -- tokenize the whole document as one string, find
the prompt/response split in the *text*, and assume that maps to some
clean index in the *token* array -- is wrong in general: a BPE merge can
span the prompt/response text boundary, so the token adjacent to the split
might not correspond to a clean cut in text space at all. This is a known
issue in real SFT tokenization pipelines (it's why e.g. TRL's
`SFTTrainer` warns about exactly this when computing response-only loss
masks).

Fixed at the source: `tds/shard_builder.py`'s new `_tokenize_document()`
tokenizes the prompt and response text *separately* for lanes in
`STRUCTURE_PRESERVING_LANES = {"instruction"}`, then concatenates the two
id lists (+ eos) -- guaranteeing an exact, unambiguous token boundary,
at the (negligible, at this scale) cost of losing whatever single BPE merge
might otherwise have spanned that boundary. The resulting boundary is
recorded as `response_start_token` (absolute position within the shard) on
the document's span in the manifest, alongside `start_token`/`end_token`;
it's `None` for every other lane, which keeps their tokenization (and
shard content) completely unchanged.

For lanes in `STRUCTURE_PRESERVING_LANES`, the split point is found by a
case-insensitive search for the literal marker `"output:"` in the
document's raw text; a document without that marker degrades gracefully
to "fully response" (`response_start_token` == its own `start_token`,
i.e. no extra masking -- identical to `concatenate_and_chop` for that one
document) rather than raising. This matters in practice: the toy corpus's
one `instruction` document uses a clean `Instruction:/Input:/Output:`
template and gets a real split (`response_start_token=98` out of 140
tokens, verified below), but the **real corpus's `instruction` lane is
much messier** -- of its 17 documents, only **1** contains a detectable
`"output:"` marker at all (checked directly against
`data/corpus/small_shard.parquet`). The other 16 fall back to "fully
response," which is an honest reflection of the data, not a bug: this
project's real corpus doesn't actually have clean SFT-style records for
most of its `instruction` lane, so structure-preserving packing is properly
demonstrated on the toy corpus, and degrades safely rather than
mis-splitting on the real one. Rebuilding shards after this change did not
change either corpus's per-lane token totals (`schedule_hash` for both
profiles came out byte-identical to before) -- the one real-corpus document
with a marker happened to tokenize to the same total length either way.

### The Packer itself

`tds/packer.py`'s `Packer` builds one packed window at a time
(`pack_step(global_step) -> List[PackedSample]`, one `PackedSample` per
batch slot) by pulling from a lane's `lane_document_stream`, concatenating
document tokens until the window is full, and **carrying over** whatever
of a document didn't fit into the *next* window for that same lane (no
document is ever dropped, duplicated, or re-read from its own start
prematurely). Per packed sequence:

- `segment_id`: local index (0, 1, 2, ...), reset every window -- never
  compared across windows, only needs to be locally unique.
- `position_id`: resets to 0 at each segment boundary.
- `loss_mask`: 1 by default; 0 at the window's true final position (no
  next-token target); additionally 0 over a segment's prompt portion under
  `"structure_preserving"`. Padding never arises in this design --
  `lane_document_stream` cycles forever, so a window is always fully
  fillable; there's no "ran out of data" case to pad for.
- `segment_boundaries`: per segment, both the window-local range *and* the
  absolute shard-token range it was read from -- the latter is what lets
  the loss-mask computation compare against `response_start_token`.

Two policies, selected per lane via the same `STRUCTURE_PRESERVING_LANES`
set the shard builder uses (so the two can never disagree about which
lanes get which treatment):

- `concatenate_and_chop` (default, everything except `instruction`): every
  position loss-visible except the window's final one -- matches the
  design doc's own worked example exactly (reproduced verbatim as
  `tests/test_packer.py::TestConcatenateAndChopWorkedExample`, including
  the doc-B-overflows-into-the-next-window case).
- `structure_preserving` (`instruction`): additionally masks the prompt
  portion of any segment with a real `response_start_token`.

`attention_bias_from_segments(segment_id) -> np.ndarray` materializes the
boolean causal+same-segment matrix -- purely a verification/test utility.
Per the design doc, real attention computation should bias directly from
the O(L) `segment_id` array, never build this O(L^2) matrix; nothing in
`PackedSample` stores it.

### Verified against the real toy shards

```
$ python -c "... Packer(...).pack_step(0) on the plain 'code' lane ..."
token_ids   : (223, 13, 272, 1, 285, 72, 273, 299)
segment_id  : (0, 0, 0, 0, 1, 1, 1, 1)
loss_mask   : (1, 1, 1, 1, 1, 1, 1, 0)
boundaries  : ((0, shard-000000, doc-000002, 0,4, 19,23), (1, shard-000000, doc-000003, 4,8, 23,27))
attention bias (8x8): causal-and-same-segment exactly as the two 4-token blocks predict

$ python -c "... Packer(...).pack_step(0) on the real 'instruction' document (140 tokens, response_start_token=98) ..."
policy: structure_preserving
prompt-masked positions: 98/98
response loss-visible positions: 41/41 (position 139 is the window's final position -> masked)
loss_mask[95:105]: (0, 0, 0, 1, 1, 1, 1, 1, 1, 1)   <- exact 98-token boundary
```

(The toy corpus's own compiled curriculum doesn't happen to include the
`instruction` lane in its mixture, so the second check above used a small
one-off single-lane schedule against the same real, already-built toy
shards/manifests -- not a separate corpus.)

### `best_fit` sequence packing (a second, orthogonal axis)

Mirrors the shard builder's own `greedy`/`best_fit` choice, but at a
different layer: the shard builder's `packing_policy` decides how
*documents get grouped into shards*; this new `sequence_packing_policy`
decides *which document gets pulled next to fill a window's remaining
room*, once a lane has already been chosen. It's a Packer-wide setting
(`Packer(..., sequence_packing_policy="best_fit")`, default `"greedy"`),
completely independent of the existing loss-masking policy
(`concatenate_and_chop`/`structure_preserving`, unchanged, still driven by
lane) -- a lane can be packed `best_fit` *and* `structure_preserving` at
once.

**Why "best fit" can't just reuse the shard builder's `_pack_best_fit`.**
That function does classic best-fit-decreasing bin-packing, where a bin is
allowed to end up under-full when nothing remaining fits it -- fine for a
shard (any size is valid), fatal for a fixed-`sequence_length` window
(there's no padding mechanism in this design; every window must come out
exactly full). So sequence-level best-fit needed its own algorithm, one
that guarantees an exact fill by construction:

1. If a document is already partially consumed (mid-carry from the
   previous window), finish it first -- carry-over behaves identically
   under both policies; `best_fit` only changes how a *new* pick is made,
   never how an overflowing document gets sliced.
2. Otherwise, look across the *current epoch's* not-yet-consumed documents
   (materialized via a new `tds/cursor.py` primitive, `lane_epoch_stream`
   -- the per-epoch counterpart to `lane_document_stream`, yielding one
   whole shuffled epoch at a time instead of flattening every epoch into a
   single per-document stream) and pick whichever document fits the
   remaining room most tightly (largest that's `<= room`).
3. If none fits at all, fall back to the *smallest* remaining document --
   it will still overflow and carry over, but minimizes how much of it
   spills into the next window, converging back to exact fits sooner than
   picking a big one would.

Both cases still route through the exact same `_fill_window` carry-over
loop as `greedy` -- the two policies only differ in the one-line
"how do I get the next document" step, injected as a small callable
(`_next_document_picker`), so the carry/slicing logic that was already
tested for `greedy` can't silently diverge between the two.

**A real correctness lesson from writing this section's tests:** the first
draft of a "does best_fit reduce fragmentation" test assumed it would find
a multi-document *combination* that exactly fills a window (e.g. picking a
6-length and a 4-length document together to exactly fill a 10-token
room). That's not what this algorithm does, and isn't what "best fit"
means in the classic bin-packing sense either -- it's a single best choice
*at each decision point*, not a combinatorial subset-sum search (which
would be a much more expensive, different algorithm). The corrected test
uses a case where the single best choice is unambiguous (an exact-length
match), and a separate test checks the *set* of tokens consumed per epoch
(not their order, which best_fit deliberately doesn't preserve) to confirm
no document is ever lost or duplicated.

**Checked against the real corpus and toy corpus, honestly:** on both of
this project's current configs, most documents are considerably *longer*
than a window (`sequence_length` of 8-256 vs. documents often in the
hundreds or thousands of tokens), so the "nothing fits, pick smallest
overflowing" fallback dominates in practice, and the two policies produce
nearly identical fragmentation (measured as average `segment_boundaries`
length per packed sequence over a full schedule):

```
Real corpus (sequence_length=128/256):  greedy 1.05/sequence vs. best_fit 1.15/sequence (first 50 steps)
Toy corpus  (sequence_length=8/16):     greedy 1.13/sequence vs. best_fit 1.11/sequence (all 27 steps)
```

This is an honest, useful finding rather than a disappointing one:
`best_fit`'s benefit is real and is proven correct in the unit tests
(constructed with document lengths deliberately comparable to the window
size, where it visibly avoids splits `greedy` would force), but neither
corpus currently on disk happens to have many documents *smaller* than its
configured `sequence_length` -- the case where this policy actually
changes the outcome. Worth keeping in mind when picking a real training
corpus later: `best_fit`'s value shows up on short-document lanes (chat
turns, code snippets, Q&A pairs) packed at a modest `sequence_length`, not
on lanes made of long-form documents already bigger than the window.

### Tests (`tests/test_packer.py`, plus rewritten `tests/test_cursor.py` and additions to `tests/test_shard_builder.py`)

- Worked-example fidelity: a synthetic doc-A/doc-B shard, packed with a
  seed chosen so epoch 0's shuffle happens to land in the same order as
  the design doc's own example, reproduces its `token_ids`/`segment_id`/
  `position_id`/`loss_mask` exactly -- including that the EOS-to-next-doc
  transition stays loss-visible.
- Carry-over correctness: the *next* window picks up exactly where the
  previous one left off (verified against the design's own numbers), and
  a document 2.5x longer than one window reconstructs byte-for-byte across
  3 consecutive windows before the (size-1) pool correctly cycles back to
  re-read the same document.
- Structure-preserving masking: prompt tokens masked, response tokens
  visible, final position still masked regardless of policy; a
  structure-preserving-lane document with no marker (`response_start_token
  is None`) stays fully loss-visible except the final position -- the
  graceful-degradation path.
- `attention_bias_from_segments`: causal+same-segment truth table checked
  by hand against a known `segment_id` array.
- Determinism: two independent fresh `Packer` instances given the same
  `(seed, schedule, lane_pools)` produce an identical `PackedSample`
  stream across 10 steps.
- `best_fit` sequence packing: rejects an unknown `sequence_packing_policy`;
  picks the exact-length match over documents that would need splitting,
  regardless of their order in the pool; falls back to the smallest
  remaining document when nothing fits the current room; no document is
  lost or duplicated across many windows/epochs (checked as a token-content
  set, since `best_fit` deliberately doesn't preserve stream order); the
  policy name is recorded on `PackedSample` independently of the
  loss-masking policy; defaults to `"greedy"` when unspecified; two fresh
  `best_fit` `Packer` instances given the same seed produce an identical
  stream.
- `lane_epoch_stream` (`tests/test_cursor.py`): its first item is a full
  permutation of the pool; matches `lane_sequence` epoch-for-epoch;
  successive epochs reshuffle; deterministic across independent fresh
  generators.
- `shard_builder._tokenize_document`: non-structure-preserving lanes get
  `response_start_token=None` and unchanged tokenization; a document with
  the marker splits at exactly the boundary separately-tokenizing
  prompt/response would produce; a document without the marker degrades to
  "fully response" (`response_start_token=0`); the manifest records the
  *absolute* (shard-relative) response start correctly.
- `tests/test_cursor.py` (rewritten): `pick_lane`/`lane_for_slot`/
  `iter_lane_assignments` cover the same determinism, zero-weight
  exclusion, renormalization, divergent-seed, and scarcity-awareness
  properties as before; `lane_document_stream` covers full-pool coverage,
  reshuffle-on-cycle, cross-generator determinism, and empty-pool
  rejection.

## 6. Batch Assembler

Small and mechanical, exactly as §5.7 frames it for a single-GPU scope:
`Packer.pack_step(global_step)` already returns one full global batch's
worth of `PackedSample`s, in slot order; `tds/batch_assembler.py` just
slices that list into consecutive, equal-sized microbatches
(`global_batch_size = microbatch_size * gradient_accumulation_steps`, no
rank/worker partitioning to do) and stacks each microbatch's parallel
arrays into the shape a model actually consumes: `(microbatch_size,
sequence_length)`.

### Design choices

- **Order preservation is the load-bearing property, not an incidental
  one.** Row `i` of microbatch `k` is always `packed_samples[k *
  microbatch_size + i]` — never reordered, never reshuffled. This is what
  makes a later replay (recompute `pack_step` + `assemble_step` from
  scratch for a historical step) directly comparable, microbatch-for-
  microbatch and row-for-row, against what a live run's consumption
  ledger recorded at the time.
- **`microbatch_size` is a plain constructor argument, not a YAML config
  field yet.** There's no training script consuming it yet (that's the
  next component down the line) — plumbing it into a config file now
  would be a config surface with nothing real behind it. It becomes a
  config field once an actual training loop exists to read one.
- **dtypes chosen for direct model consumption**: `token_ids`/`segment_id`/
  `position_id` as `int64` (the dtype PyTorch embedding/index lookups
  expect), `loss_mask` as `float32` (multiplies elementwise against
  per-token loss — the standard `(per_token_loss * loss_mask).sum() /
  loss_mask.sum()` idiom).
- **`rank` is always `0`**, kept on `Microbatch` only because the
  consumption ledger schema (course notes §8) lists it as a required
  per-microbatch field — a forward-compatible placeholder for a scope this
  design deliberately doesn't have (single-GPU, no cross-rank concerns).
- **Attention still isn't materialized here**, for the same reason as in
  `tds/packer.py`: `segment_id` is stacked and handed to the model as an
  `O(microbatch_size * sequence_length)` array; nothing builds an explicit
  per-sample L×L mask at this layer either.
- **`Microbatch` needed a custom `__eq__`** (`@dataclass(frozen=True,
  eq=False)` plus a hand-written one) since it holds `numpy` arrays —
  the dataclass-generated `__eq__` would compare them with `==` and get
  an array back instead of a bool, which is exactly the kind of subtle
  breakage the test suite is there to catch before it reaches real code.
- **`samples: Tuple[PackedSample, ...]` rides along on every `Microbatch`**
  alongside the stacked arrays — full provenance (lane, document IDs,
  segment boundaries) for whatever eventually logs the consumption ledger,
  without needing to recompute anything.

### Verified against the real corpus

```
global_batch_size=8, microbatch_size=4
produced 2 microbatches for step 0
  microbatch_index=0 token_ids.shape=(4, 128) dtype=int64 lanes=['code', 'code', 'qa', 'code']
  microbatch_index=1 token_ids.shape=(4, 128) dtype=int64 lanes=['math_science', 'math_science', 'general_web', 'general_web']
```

### Tests (`tests/test_batch_assembler.py`)

- Slicing/stacking: consecutive, order-preserving chunks of the correct
  shape; `microbatch_index`/`global_step`/`rank` recorded correctly;
  `microbatch_size == global_batch_size` degenerates to one microbatch;
  provenance `samples` preserved in the same order as the stacked rows;
  `loss_mask`/`position_id` values stacked correctly (not just shapes).
- Validation: rejects non-positive `microbatch_size`, uneven division,
  and mixed-`global_step` input; empty input returns an empty list.
- `Microbatch.__eq__`: two independently-assembled microbatches from
  identical inputs compare equal (exercises the custom array-aware
  equality).
- End-to-end with a real `Packer`: two independently-constructed
  `Packer`/`BatchAssembler` pairs, advanced in lockstep from scratch,
  produce identical results — `assemble_step`'s decomposition is lossless
  and exactly reconstructs what a direct `pack_step` call would have
  returned, for every step checked. Microbatch shapes match the compiled
  schedule's stage `sequence_length`.

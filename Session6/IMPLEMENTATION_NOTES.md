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
| OPUS Selector | done | `tds/opus.py`, `configs/opus*.yaml`, `scripts/run_opus_selection.py` |
| Audit / Evidence Bundle | done | `tds/audit.py`, `tds/evidence.py`, `scripts/run_demo.py` |
| Packer | done | `tds/packer.py`, `tds/shard_builder.py` (`response_start_token`) |
| Batch Assembler | done | `tds/batch_assembler.py` |
| Consumption Ledger | done | `tds/consumption_ledger.py` |
| Toy Model & Training Step | done | `tds/model.py`, `tds/training_step.py` |
| Learning Ledger | done | `tds/learning_ledger.py` |
| Checkpoint / Crash / Resume | done | `tds/checkpoint.py`, `tds/resume.py` |
| Replay / Fork | done | `tds/replay.py`, `tds/fork.py` |

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
- **`microbatch_size` is a plain constructor argument here** — `BatchAssembler`
  itself stays config-agnostic. It became a real `CurriculumConfig` YAML
  field (`microbatch_size`, alongside `global_batch_size`) once
  `scripts/run_demo.py` existed as a training loop to read one; see the
  note below.

#### `microbatch_size` as a config field (extension, closed)

`CurriculumConfig.microbatch_size` defaults to `0`, meaning "unset" —
`CurriculumConfig.resolved()` maps that to `global_batch_size` (the old
implicit behavior: one microbatch per step, `accum_steps=1`), so any
config that predates this field, or simply omits it, behaves exactly as
before. Setting it explicitly to a proper divisor of `global_batch_size`
turns on real gradient accumulation: `scripts/run_demo.py` passes
`curriculum_config.microbatch_size` (not `global_batch_size`) to
`BatchAssembler`/`recompute_step`/`verify_resume`/`replay_range` — the
only place `global_batch_size` itself is still needed is compiling the
schedule (`compile_curriculum`), since `tokens_per_step` is defined in
terms of the *global* batch, not the microbatch.

Both shipped profiles now demonstrate `accum_steps=2`:
`configs/curriculum.yaml` sets `global_batch_size=8, microbatch_size=4`;
`configs/curriculum_toy.yaml` sets `global_batch_size=2,
microbatch_size=1`. This is invisible to packing/mixture/consumption —
the Packer still produces `global_batch_size` samples per step regardless
of how they're later sliced into microbatches — confirmed by re-running
`scripts/run_demo.py` on the real corpus and diffing `tokenizer_hash`
and `schedule_hash` against the pre-change resting state: unchanged.
What *does* change is the evidence bundle's crash-recovery line, which
now genuinely shows more than one microbatch id per step (e.g.
`batch_ids=['mb-26-0', 'mb-26-1']`).
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

## 7. Consumption Ledger

### The design doc's example schema doesn't quite hold together as written

DATALOADER_DESIGN.md §5.8 gives one illustrative JSON record with
*singular* `mixture_lane` and `loss_mask_hash` fields, alongside a
*plural* `packed_sample_ids` list. That only holds together if a
microbatch never mixes lanes and never spans more than one shard per
sample. Ours routinely does both — the real-corpus `BatchAssembler` check
in §6 already showed one microbatch with lanes
`['code', 'code', 'qa', 'code']`, and a single packed sample can span
multiple shards whenever its window carries over a document that started
in a different shard than the one it finishes in.

**Fix, matching the same reasoning as every other design-doc correction
this session:** promote the per-sample fields (`packed_sample_ids`,
`mixture_lane`, `shard_ids`, `token_span_ids`, `opus_decision_id`) to
parallel lists, one entry per row of the microbatch, in the same order as
`Microbatch.samples`. `loss_mask_hash` stays a single hash — but of the
*whole stacked* `(microbatch_size, sequence_length)` array via
`.tobytes()`, not of any one sample's mask. That's the object actually
served to the model in one shot, and the thing a replayed step has to
reproduce byte-for-byte. Fields that are genuinely microbatch-wide
constants (`curriculum_stage`, `attention_policy`, `position_policy`,
`tokenizer_version`, `dataloader_version`, `rank`) stayed singular, since
they don't vary per sample within one step.

### `build_ledger_entry` is a pure function; `ConsumptionLedger` just persists it

Splitting these apart (rather than one class doing both) means the entry
a hypothetical replay path recomputes can be diffed directly against what
`ConsumptionLedger.append()` already wrote, with no I/O on the
recomputation side. `curriculum_stage` is derived from
`schedule.stage_at_step(microbatch.global_step)` rather than passed in
separately — one less thing a caller could get out of sync with the
schedule itself.

`ConsumptionLedger` mirrors `ManifestStore`'s append-only,
idempotent-on-identical-reappend design exactly, keyed by
`(run_id, branch_id, microbatch_id)`: re-appending the *same* entry (a
crash-recovery retry replaying a step that was already fully recorded) is
a harmless no-op, but appending *different* content under an already-used
key raises `ConsumptionLedgerError`. That rejection is precisely the
mechanism that would catch a "repeated batch diverged from the original"
bug — one of the assignment's explicit crash-recovery requirements — the
moment it happened, rather than letting it silently corrupt the ledger.

`last_recorded_microbatch(run_id, branch_id)` returns the highest
`(global_step, microbatch_index)` tuple on record, or `None` for a fresh
branch. Deliberately just a fact query, not a "what should run next"
decision — answering that also needs `gradient_accumulation_steps` (to
know whether a step's microbatches are all present, or a crash landed
mid-step with only some of them recorded), which isn't ledger state. That
reasoning belongs to the not-yet-built crash/resume component.

### Verified against the real corpus

```
total entries: 6
last recorded: (2, 1)

{
  "microbatch_id": "mb-0-0",
  "curriculum_stage": "foundation",
  "packed_sample_ids": ["ps-0-0", "ps-0-1", "ps-0-2", "ps-0-3"],
  "mixture_lane": ["code", "code", "qa", "code"],
  "shard_ids": [["shard-000001"], ["shard-000001"], ["shard-000033"], ["shard-000001"]],
  "token_span_ids": ["shard-000001:42204-42332", ...],
  "loss_mask_hash": "sha256:c39caba...",
  "opus_decision_id": [null, null, null, null]
}
```

### Tests (`tests/test_consumption_ledger.py`)

- `build_ledger_entry`: correct `microbatch_id`/`packed_sample_id`
  formatting; `curriculum_stage` correctly derived from the schedule; all
  microbatch-wide constant fields present and correct; per-sample list
  fields have one entry per row, in row order; `shard_ids` deduplicated
  per sample (a sample packing two documents from the same shard lists it
  once, not twice); `token_span_ids` format matches
  `{shard_id}:{start}-{end}` built from each segment's absolute
  shard-token range; `loss_mask_hash` matches an independently-computed
  hash of the whole stacked array; two independently-constructed
  `Packer`/`BatchAssembler` pairs produce byte-for-byte identical ledger
  entries for the same step — the property resume/replay ultimately rests
  on.
- `ConsumptionLedger`: append+get roundtrip; identical re-append is
  idempotent (one line on disk); a mismatched re-append under the same key
  raises `ConsumptionLedgerError`; reloading from disk recovers prior
  entries; `for_branch` correctly isolates entries across different
  `(run_id, branch_id)` combinations, including a fork sharing a `run_id`
  with `main`; `last_recorded_microbatch` returns the correct max tuple,
  `None` for a fresh branch, and ignores other branches' entries.

## 8. Toy Model & Training Step

Per DATALOADER_DESIGN.md §8's scope assumption: "Model is a small toy
transformer, only large enough to produce real loss/perplexity numbers
for the learning ledger — not a scale target." Built now because the
Learning Ledger's numbers (`avg_token_loss`, `loss_delta_before_after`)
have to come from *somewhere real* — recording a fabricated or
placeholder loss value would violate the same "prove it, don't assert it"
principle this entire project has followed for everything else.

### `tds/model.py`: deliberately manual, not library-attention

`ToyTransformer` is a small decoder-only transformer (`d_model`,
`n_layers`, `n_heads`, `d_ff` all configurable via `ToyTransformerConfig`)
with hand-written multi-head self-attention rather than
`nn.MultiheadAttention` or any attention library. The reason isn't
NIH — it's that this whole session has been building toward two specific
correctness properties (segment-aware attention, position-id resets), and
burying them inside a library call would make the model the one place in
the codebase where those properties *aren't* directly inspectable:

- **Position embeddings are indexed by `position_id`, not absolute
  sequence position.** Since the Packer already resets `position_id` to 0
  at each segment boundary, a document that starts partway through a
  packed window gets its own local position sequence from the embedding
  table, not some arbitrary large offset inherited from whatever
  preceded it in that window.
- **Attention bias comes from `segment_id`** via
  `attention_bias_for_microbatch`, which calls
  `tds.packer.attention_bias_from_segments` — the exact same function the
  Packer's own tests use to *verify* the causal+same-segment property is
  correct. The toy model is the first real *consumer* of that function,
  not just a checker of it: the boolean mask becomes an additive float
  bias (`0` / `-1e9`) added to raw attention scores before softmax.

`ToyTransformer.__init__` takes an explicit `seed` and calls
`torch.manual_seed(seed)` before constructing any parameters — a known,
documented simplification (it mutates torch's *global* RNG state rather
than using a generator scoped to this one model), acceptable because
exactly one toy model exists per process in this project's scope.

### Loss computation: masked next-token cross-entropy

`compute_batch_loss(model, microbatch)` shifts logits/targets by one
position (`shift_logits = logits[:, :-1]`, `shift_targets =
token_ids[:, 1:]`), computes per-position cross-entropy, and multiplies by
`loss_mask[:, :-1]` — dropping the window's final position outright,
which needs no special-casing since its `loss_mask` is always 0 already
(no next-token target for a window's last position, per §5.6). Splitting
this into `forward_and_masked_loss` (graph-preserving, used for both
`compute_batch_loss`'s read-only wrapper *and* `tds/training_step.py`'s
backward pass) avoids computing the same forward pass two different ways
that could silently drift apart.

### `tds/training_step.py`: one optimizer update, evaluated before and after

The Learning Ledger's `loss_delta_before_after` (course notes: "loss delta
before and after exposure") is a property of one full optimizer update —
not one microbatch — since gradient accumulation across a step's
microbatches happens *before* the update, and "after" has to mean the
*same* data re-evaluated post-update. `run_training_step`:

1. Evaluates every microbatch in `eval()` / `no_grad()` mode — "before."
2. Switches to `train()`, backpropagates each microbatch's loss (scaled by
   `1 / len(microbatches)`, the standard gradient-accumulation
   normalization so accumulated gradients match what one big batch would
   have produced), then one `optimizer.step()`.
3. Re-evaluates the *exact same* microbatches with the updated weights —
   "after."

Nothing about ledgers, checkpoints, or crash/resume lives in this
module — it owns exactly one training step's mechanics, keeping it
reusable by whatever eventually drives a full run loop.

### Verified

- Determinism: two independently-constructed `(model, optimizer,
  microbatches)` triples, seeded identically, produce bit-for-bit
  identical `avg_loss_before`/`avg_loss_after` and identical post-step
  parameters (`torch.allclose` at `atol=1e-6`) — CPU-only PyTorch ops are
  deterministic by default, so this held without needing any special
  determinism flags.
- A large learning rate applied to one step reliably reduces that same
  step's own data's loss — the "before vs after exposure" signal made
  unambiguous by exaggerating the step size, rather than relying on a
  small, noisy natural improvement.
- Real corpus, 3 live training steps (`d_model=32`, 2 layers, `lr=1e-3`):
  loss decreased after every single exposure (`9.1448→9.1080`,
  `9.1410→9.1153`, `9.1012→9.0769`).

### Tests (`tests/test_model.py`, `tests/test_training_step.py`)

- Model determinism: identical seeds produce identical initial weights;
  different seeds produce different weights.
- `attention_bias_for_microbatch` matches `attention_bias_from_segments`
  exactly for a hand-constructed multi-sample `segment_id` array.
- `compute_batch_loss`: correct output shape (`microbatch_size,
  sequence_length - 1`); `avg_loss` matches an independently-computed
  masked weighted average; always a finite plain `float`; deterministic
  across two freshly-constructed same-seed models.
- `run_training_step`: rejects empty input and mixed-`global_step` input;
  an optimizer step measurably changes model parameters; result fields are
  well-formed; a large-learning-rate step reduces that step's own loss;
  two fully independent, identically-seeded runs produce identical losses
  and identical post-step parameters.

## 9. Learning Ledger

### The design doc's example schema has the same gap the consumption ledger's did

DATALOADER_DESIGN.md §5.9's example entry has no `run_id`/`branch_id` at
all — which would let two different runs' (or two forked branches')
entries collide on the same `(global_step, shard_id)` key. Extended the
same way as the consumption ledger: keyed by
`(run_id, branch_id, global_step, shard_id)`.

### What "before/after" and "repeated pass" actually mean here

- **`avg_token_loss`** is the *before* value from `accumulate_per_shard_loss`
  — the loss as measured when this step's data was fed in, attributed to
  the shard(s) it actually came from via each sample's
  `segment_boundaries` (a single sample can span multiple shards; each
  shard is only credited the token positions it actually contributed).
- **`loss_delta_before_after`** = after − before, both computed the same
  way, for the exact same data. Negative means this step's exposure
  measurably helped; positive means it measurably hurt (a real gradient
  spike, not a placeholder concept).
- **`repeated_pass_number`** is computed *from the consumption ledger*,
  not tracked independently: count how many of a branch's consumption
  entries at or before this `global_step` touched this `shard_id`. This
  means it can never drift from what was actually recorded as served —
  there's no second, parallel notion of "how many times has this shard
  been seen" to keep in sync. **Ordering matters**: a step's consumption
  ledger entries must be written before its learning ledger entries are
  built, since the count includes the current step.
- **`model_phase`** (`early`/`mid`/`late`/`anneal`) buckets by fraction of
  `schedule.total_steps` completed, *except* when the current stage's own
  name contains "anneal" (our curriculum configs literally name a stage
  that), which reports `"anneal"` regardless of position — a real,
  distinct phase, not just "whichever third of the run this falls into."
- **`usefulness_classification`** (`useful`/`neutral`/`harmful`) is a
  simple threshold on `loss_delta_before_after` (`±1e-3` by default) — an
  inspectable heuristic, not a claim of sophistication, and per §5.9
  explicitly not acted on automatically in this phase: "recorded for the
  next version to consume."
- **`opus_score`** was `None` for every entry when this section was first
  written (OPUS's score is per-*document*, computed once before packing,
  and `TrainingStepResult` didn't carry it through to here) -- closed in
  §14 with a real token-weighted shard-level rollup, once §12's OPUS build
  gave this something real to roll up.

### Verified against the real corpus

```
step=0 avg_loss_before=9.1448 avg_loss_after=9.1080 shards_touched=4
step=1 avg_loss_before=9.1410 avg_loss_after=9.1153 shards_touched=4
step=2 avg_loss_before=9.1012 avg_loss_after=9.0769 shards_touched=4

{
  "global_step": 0, "shard_id": "shard-000001",
  "avg_token_loss": 9.148, "loss_delta_before_after": -0.0396,
  "opus_score": null, "repeated_pass_number": 1,
  "model_phase": "early", "usefulness_classification": "useful"
}
```

### Tests (`tests/test_learning_ledger.py`)

- `accumulate_per_shard_loss`: single-shard averaging; a sample spanning
  two shards attributes each shard exactly the positions it contributed
  (not the whole sample); masked positions excluded from both the sum and
  the count.
- `model_phase_at_step`: early/mid/late boundaries by progress fraction;
  an anneal-named stage reports `"anneal"` regardless of position within
  it.
- `classify_usefulness`: useful/harmful/neutral threshold boundaries.
- `accumulate_per_shard_opus_score` (§14): a single-document shard gets
  that document's own score; a shard with multiple documents gets a
  token-weighted average (not a plain average across documents); a
  document with no recorded score, or an explicit `None` score, is
  excluded from the average rather than treated as zero.
- `build_learning_ledger_entries`: `opus_score` stays `None` when
  `opus_decisions` isn't passed (unchanged default behavior); rolls up to
  the real score when a decision list is provided. Verified against the
  real corpus: all 217 learning-ledger entries over a 50-step demo run
  carried a real `opus_score`.
- `build_learning_ledger_entries` end-to-end (real `Packer` +
  `BatchAssembler` + `ConsumptionLedger` + `run_training_step`): correct
  shard attribution and delta value (checked against the step's overall
  before/after delta in a single-shard fixture); `repeated_pass_number`
  correctly increments `1 → 2` across two real training steps touching
  the same shard.
- `LearningLedger`: append+get roundtrip; identical re-append idempotent;
  mismatched re-append rejected; reload from disk recovers prior entries;
  `for_branch` isolates entries correctly.

## 10. Checkpoint / Crash / Resume

This is the component the design doc calls out as fixing v4's core
problem, and it's where building the thing surfaced a real, non-obvious
correction to the design doc's own claim about how cheap "recompute" is.

### `tds/checkpoint.py`: exactly three values, nothing about the dataloader

`CheckpointManager.save()` writes model weights, optimizer state, and
torch's global RNG state, atomically (`torch.save` to a `.tmp` path in
the same directory, then `os.replace` -- a crash mid-write can never
leave a corrupt file at the real path), keyed by `(run_id, branch_id,
global_step)`. `weights_only=False` on load is deliberate: our checkpoint
payload carries non-tensor metadata (run_id/branch_id/global_step,
optimizer state) alongside tensors, which is safe here specifically
because these are files this project's own code wrote, never an
untrusted external checkpoint (torch 2.6+ defaults `weights_only=True`,
which would reject this payload outright).

`next_step_after_checkpoint(metadata)` is a one-line function purely so
the "resume point = checkpointed step + 1" invariant has a name instead
of being an inline `+ 1` that a future edit could quietly get wrong --
that exact off-by-one is the entire difference between "correct resume,"
"skipped batch," and "repeated batch."

`parent_branch_id`/`fork_step` fields exist on `CheckpointMetadata` now
(round-trip correctly when provided) even though Fork itself (§5.13)
isn't built yet -- the same forward-compatible-field pattern already used
for `rank`/`opus_decision_id` elsewhere.

### `tds/resume.py`: a real correction to the design doc's own claim

DATALOADER_DESIGN.md §6 describes resume as: "Recompute, don't restore:
call `cursor(seed, mixture_config, global_step + 1)`. This is arithmetic
(a formula lookup), not re-tokenization or re-reading parquet" — implying
recomputing an arbitrary step is O(1). **Building this exposed that's only
true for lane *assignment*.** The first version of `recompute_step` built
a *fresh* `Packer` and called `pack_step(resume_step)` directly on it —
and the very first real test of the full crash/resume cycle against real
multi-lane, multi-document data caught it immediately: the recomputed
batch didn't match the control run's ledger entry at all past step 0.

**Why:** `pick_lane` (which lane does this slot draw from) genuinely has
no history dependency — it's a pure hash of `(seed, global_step, slot)`,
correctly O(1) per query (see §4/§5's Cursor notes). But the Packer's
*document consumption* position is cumulative: which document comes next
in a lane's stream depends on exactly how much of that stream every
*earlier* step's windows already consumed (carry-over spans, real
document lengths — none of which reduce to a step-indexed formula the
way lane assignment does). A fresh Packer asked to jump straight to step
N has no way to know where each lane's stream actually is at that point.

**Fix:** `recompute_step` now replays every step from 0 up to and
including `global_step` on the fresh `Packer`/`BatchAssembler`, discarding
every intermediate result and keeping only the last. This is the real,
unavoidable cost of a checkpoint that stores nothing but
`(run_id, branch_id, global_step)` (deliberately, per §5.10 — no saved
file-position cursor): a small, fixed-size checkpoint traded for
`O(global_step)` resume-time recomputation, not the `O(1)` the design
doc's own wording implied. At this project's step counts (low thousands),
that replay is still a sub-second operation and strictly cheaper than
re-tokenizing or re-reading the corpus — the actual point the design doc
was making, just not literally "a formula lookup." `tds/cursor.py`'s own
docstrings already say almost exactly this about `lane_document_stream`
("no shortcut to 'the k-th document' other than counting") — this section
generalizes that same fact from one lane's stream to the whole Packer.

### `verify_resume`: the actual proof, not a trust exercise

Recomputes `resume_step`'s microbatches from scratch and compares
`shard_ids`/`token_span_ids`/`loss_mask_hash` (the exact fields the
consumption ledger already stores for verification, not full token
arrays) against what a control run's consumption ledger already has on
record for that step. A missing control entry is reported as a mismatch,
not silently skipped — there's nothing to prove resume against if the
step was never actually served by an uninterrupted run.

### Verified against the real corpus

```
[event] shards created / manifests validated (already on disk)
[PASS] checkpoint_saved step=4
[event] crash simulated after step 4
[event] run resumed from checkpoint step=4
[PASS] resume_next_batch_matched step=5
```

...matching the assignment's own execution-log event names and `[PASS]`
tags exactly.

### Tests (`tests/test_checkpoint.py`, `tests/test_crash_resume.py`)

- `CheckpointManager`: save+restore round-trips both model weights *and*
  optimizer state (checked after a real optimizer step so Adam's moment
  estimates are non-trivial, not just freshly-initialized defaults);
  loading a missing checkpoint raises `FileNotFoundError`; a successful
  save leaves no `.tmp` file behind; `latest_step` returns the correct
  max and `None` for an empty branch, isolated correctly per branch; fork
  metadata round-trips when provided.
- `next_step_after_checkpoint`: exactly one past the checkpointed step.
- **The centerpiece** (`test_resume_after_simulated_crash_matches_control_run_and_restores_weights`):
  a full control run (uninterrupted, real multi-lane/multi-shard fixture,
  real training steps) writes its consumption ledger and saves a
  checkpoint mid-run; a *completely separate* set of objects (different
  model seed, never touching the control run's `Packer` or model)
  restores from that checkpoint and calls `verify_resume` — asserting
  both that the restored weights exactly match a snapshot taken at
  checkpoint time, and that the recomputed next-step batch exactly
  matches the control ledger's entry for that step. This is the
  assignment's "prove the next batch is exactly the expected batch"
  requirement, actually proven rather than asserted.
- `verify_resume` has teeth: a genuinely different seed is detected as a
  mismatch (not rubber-stamped), and a missing control-ledger entry is
  reported explicitly rather than silently treated as a pass.

## 11. Replay / Fork

### Replay: extracted, not duplicated

§6's own framing -- "replay and audit are nearly free extensions of
[resume's] mechanism: replay is [recompute] run for an arbitrary
historical range with an equality assertion" -- meant `verify_resume`
already contained everything replay needed, just hardcoded to a single
step. Rather than write a second copy of "recompute and compare against
the ledger," `tds/replay.py`'s `replay_range` became the one real
implementation, and `tds/resume.py`'s `verify_resume` was refactored into
a one-line call to it (`replay_range(..., resume_step, resume_step + 1,
...)`) -- confirmed to be a safe, non-breaking refactor by re-running the
full crash/resume test suite unchanged afterward (one assertion needed
updating for a cosmetic wording difference between the two modules'
mismatch messages, nothing behavioral).

**Avoiding the obvious O(n²) trap**: given §10's finding that Packer
document-consumption position is cumulative (no jumping to an arbitrary
step), the naive way to "replay a range" would be calling `verify_resume`
independently once per step in the range -- each call separately replaying
from step 0. For a range of length `k` ending at step `n`, that's
`O(n) + O(n+1) + ... + O(n+k)`, quadratic in the range. `replay_range`
instead walks *one* continuous `Packer`/`BatchAssembler` from step 0
through the end of the range exactly once, comparing against the ledger
only for steps inside `[start_step, end_step)` and discarding the
"warm-up" steps before it needed only to advance the Packer's internal
position correctly -- `O(end_step)` total, matching a single `recompute_step`
call's own cost, not multiplying it by the range length.

### Fork: almost no new mechanism needed

`CheckpointManager` already carried `parent_branch_id`/`fork_step` fields
(added during the Checkpoint/Crash/Resume build specifically so this
wouldn't need retrofitting), and `ConsumptionLedger` already keys entries
by `(run_id, branch_id, microbatch_id)` with a `for_branch` filter. The
one genuinely new piece, `tds/fork.py`'s `fork_branch`, is small on
purpose: restore the parent's checkpoint into the caller's model/optimizer,
then immediately re-save it under the new `branch_id` at the same
`global_step`, tagged with the parent lineage. From there, forking into a
diverging stream is just an ordinary resume under the new
`(run_id, new_branch_id)` pair -- a different seed, mixture config, or
curriculum stage, exactly as §5.13 describes.

**Deliberately not built here**: reconstructing "one branch's full
history across a fork" (steps before the fork point living under the
*parent's* `branch_id`, steps after living under the new one) is an
Audit-level concern -- walking `parent_branch_id`/`fork_step` lineage
across branches -- not something Fork itself needs to solve. The forked
branch's ledger only ever holds its own post-fork entries. Closed in §13
by `tds.audit.branch_lineage`/`audit_branch_lineage`.

### Verified against the real corpus

```
[PASS] checkpoint_saved step=2
[PASS] historical_stream_replayed matched=True steps=[0, 1, 2]
[PASS] branch_forked parent=main fork_step=2 new_branch=fork-1
fork-1 lanes served post-fork: [['code', 'general_web', ...], ...]
```

### Tests (`tests/test_replay.py`, `tests/test_fork.py`)

- `replay_range`: a fully-recorded range matches, including starting from
  step 0; a mismatched seed is detected across the *whole* range, not
  just one step; a range extending past what's actually been recorded
  correctly reports only the unrecorded steps as mismatched while the
  recorded ones still match; invalid ranges (`start >= end`, negative
  `start`) raise `ValueError`.
- `fork_branch`: creates a checkpoint carrying correct
  `parent_branch_id`/`fork_step` lineage; genuinely restores the parent's
  weights into the new branch (not just metadata); rejects forking to the
  parent's own `branch_id`.
- **The centerpiece** (`test_branches_share_history_up_to_fork_and_diverge_after`):
  a parent branch runs to a fork point and keeps training on its own seed;
  a fork restores that exact checkpoint into a fresh model (verified via a
  parameter snapshot taken at fork time) and continues on a *different*
  seed. Post-fork ledger entries diverge between the two branches (proving
  the fork genuinely changed the data stream), while the forked branch's
  ledger holds no entries at or before the fork point at all -- that
  shared history lives only under the parent's `branch_id`, exactly as
  designed.

## 12. OPUS Selector: from stub to real, config-toggleable selection

Per DATALOADER_DESIGN.md §5.5, OPUS was deliberately scoped as an identity
pass-through stub while the toy model didn't exist yet (see the
`dataloader-design-doc` memory's scoping note). That blocker is gone now
that §8 built a real toy transformer for the Learning Ledger -- this
section replaces the stub with real proxy scoring, gated by
`OpusConfig.enabled` in `configs/opus*.yaml`.

### Integration point: a pre-packing filter, not a Packer/Cursor change

The design doc says OPUS "sits between the cursor and the packer." Taken
literally that could mean weaving a scoring call into the Packer's
window-filling loop -- but that would mean the Packer needs a live model
whenever OPUS is on, breaking the property every component since Cursor
has carefully preserved: Cursor and Packer are pure functions of frozen
inputs, and never touch a model at all. Instead, OPUS runs once, entirely
*before* any `Packer` or `Cursor` exists: `apply_opus_selection` scores
every document in the raw `lane_pools`
(`ManifestStore.document_pool_by_lane()`) and returns a *filtered*
`lane_pools` dict, which gets handed to `Packer(..., lane_pools=filtered_pools)`
with zero changes to `Packer` or `Cursor`. A rejected document simply
never appears in the pool the Packer ever sees -- verified directly in
`tests/test_opus.py`'s `TestOpusIntegratesWithPacker`.

### Scoring: reusing `compute_batch_loss`, not re-deriving it

`score_candidate` scores one document by average per-token loss under a
given model snapshot, up to `max_sequence_length` tokens. Rather than
re-deriving position/attention handling for a lone document, it builds a
single-row, single-segment `Microbatch` (segment_id all zeros -- one
document is the degenerate case of one segment spanning the whole window)
and calls `tds.model.compute_batch_loss` directly, so scoring can never
silently diverge from how loss is computed everywhere else in the
project.

Decision policy, per document:

```
score < reject_below  -> "rejected"  (low_proxy_utility)
score > defer_above    -> "deferred"  (anomalous_high_loss)
otherwise              -> "accepted"
```

A document whose lane is in `protected_lanes` is always accepted
(`protected_floor_override=True` if it would otherwise have been
rejected/deferred) -- mirroring the mixture compiler's own floor-via-
repetition philosophy: a scarce, protected capability lane must never be
zeroed out by the proxy's own judgment, however confident that judgment is.

### A real finding: an untrained model has no discriminating signal

Before finalizing `configs/opus.yaml`'s thresholds, scoring the real
corpus with a freshly-initialized model showed scores clustered *tightly*
around `ln(vocab_size)` (~9.0 for this corpus's ~8000-token vocab):

```
n=30  min=9.070  max=9.281  mean=9.175  std=0.047
```

This is expected and honest, not a bug: a randomly-initialized model
predicts every token with roughly uniform probability, so every document
looks equally "surprising." There is no real utility signal yet to act
on. After 40 real training steps (`tds.training_step.run_training_step`,
same corpus, same model architecture), the same candidates spread out
substantially:

```
n=48  min=7.350  max=11.345  mean=8.365  std=1.246
```

`reject_below`/`defer_above` in the shipped configs are set wide enough
that neither an untrained nor a lightly-trained model produces a
degenerate all-one-bucket result -- but the real, meaningful splits this
section demonstrates below all come from the *trained* checkpoint. This
is a genuine, useful thing to know operationally: OPUS-style proxy
scoring is only as informative as the model doing the scoring, and a
production run would want to score with a checkpoint that's already seen
real training, not the initial random weights.

### A genuinely meaningful finding, once trained: `indic` is hard for this model

Running selection (real corpus, 40-step-trained checkpoint, no protected
lanes) actually deferred real documents:

```
accepted=832 rejected=0 deferred=18
  indic          accepted=30 rejected=0 deferred=18
  (all other lanes: 0 deferred)
```

All 18 deferrals were `indic`-lane documents -- a real, plausible signal:
Hindi/Bangla-script content, tokenized byte-level and under-represented
in this corpus, is genuinely harder for a 2-layer toy model to predict
than English web/code/qa text after only 40 steps. Re-running with
`protected_lanes: [indic, instruction]` (the shipped default) rescues
exactly those documents:

```
accepted=850 rejected=0 deferred=0
  indic          accepted=48 rejected=0 deferred=0, 18 floor-rescued
```

This is precisely the floor-rescue mechanism working as designed, driven
by a real, data-derived signal rather than a synthetic test case.

### Freezing: OPUS's output is not reproducible from static config alone

Every other frozen artifact in this project (tokenizer, mixture schedule)
is a pure function of its config -- recompiling with the same inputs
always gives the same result. OPUS selection is different: it depends on
a *specific model snapshot*, and a model that's since been trained
further would score every candidate differently. So `freeze_opus_selection`/
`load_frozen_opus_selection` (hash-pinned, same pattern as
`freeze_schedule`/`load_frozen_tokenizer`) exist specifically so a later
consumer loads the *exact* decisions made at selection time, and never
silently re-runs OPUS against whatever the "current" model happens to be.

### Wiring into the consumption ledger: a real schema refinement

`ConsumptionLedger.build_ledger_entry`'s `opus_decision_id` field used to
be a flat `None` per sample (§7, written when OPUS was still a stub).
Now that decisions are real, it's a list per sample -- one entry per
*unique* document contributing to that sample, mirroring how `shard_ids`
already deduplicates per sample, just keyed by document instead of shard
(`f"opus-{shard_id}-{document_id}"`, `tds.opus`'s own candidate_id
format). A new `opus_enabled` flag on `build_ledger_entry` controls
whether real ids or `None` get written -- derived directly from the
`(shard_id, document_id)` pair with **no lookup dict needed**, because by
construction only *accepted* candidates ever reach the Packer: a
document's mere presence in a packed sample already proves OPUS said yes.

Existing tests needed updating for the shape change
(`test_per_sample_fields_are_parallel_lists_matching_row_order`); no
other behavior changed.

### Config: the on/off toggle

`OpusConfig.enabled` (`configs/opus.yaml`/`configs/opus_toy.yaml`) is the
toggle. `enabled: false` writes a pass-through decision log -- every
candidate accepted, `opus_score: null` -- **without constructing or
running any model at all**, not just a model that happens to always say
yes. `checkpoint_path: ""` (the shipped default) means "a freshly-
initialized model seeded by `seed`," since no persistent training loop
exists yet to have produced a real checkpoint to point at; setting it to
an actual `tds.checkpoint.CheckpointManager`-written `.pt` path scores
with that snapshot instead (demonstrated, not shipped, in the finding
above).

### Tests (`tests/test_opus.py`, plus additions to `tests/test_consumption_ledger.py`)

- `score_candidate`: deterministic for identical inputs; a document too
  short to have any next-token prediction scores `0.0`; always a finite
  float.
- `apply_opus_selection` threshold logic, using a mocked `score_candidate`
  so branching is tested independent of what a real forward pass produces:
  low scores rejected and excluded from the filtered pool; high scores
  deferred and excluded; middle scores accepted with a real
  `effective_token_estimate`; a protected lane is rescued from rejection
  (`protected_floor_override=True`); disabled selection accepts everything
  without a model at all; enabling without a model raises.
- `filtered_pools_from_decisions` reconstructs the same filtered pools an
  in-memory `apply_opus_selection` call produced.
- `freeze_opus_selection`/`load_frozen_opus_selection`: round-trip
  recovers identical decisions and the same `decisions_hash`; a missing
  frozen selection raises `FileNotFoundError`; tampering is detected and
  raises `ValueError`.
- **Integration**: a rejected document, via a real `Packer` fed the
  filtered pool, never appears in any packed sample across many steps --
  the property this whole integration design exists to guarantee.
- `build_ledger_entry` (consumption ledger): `opus_decision_id` uses the
  real `opus-{shard_id}-{document_id}` format when `opus_enabled=True`,
  matching each sample's actual unique contributing documents.

## 13. Audit / Evidence Bundle -- and `scripts/run_demo.py`

The last item on the component list, and the one that required building
the thing every other component was implicitly aimed at: a single command
running the complete demonstration, per the assignment's own
`python run_demo.py` requirement. Audit and Evidence Bundle only have
something real to report on once a full run has actually happened, so
this section is both the reporting/analysis layer (`tds/audit.py`,
`tds/evidence.py`) and the orchestration that produces the data for it to
report on (`scripts/run_demo.py`).

### `tds/audit.py`: ledger-only, by design, with one honest limitation

§5.14 says audit should reconstruct contribution, utilization, and
throughput "purely by reading the consumption + learning ledgers, no
re-running of training required." `audit_range` respects that literally:
it never touches a `Packer`, a model, or `tds.replay` -- everything comes
from `token_span_ids`, `shard_ids`, and `mixture_lane` already sitting in
consumption-ledger entries. Two things fell out of taking that
constraint seriously:

- **`packing_utilization` is provably `1.0`, not just reported as such.**
  This project's Packer never pads (every window always carries a
  carry-over slice or a fresh document, always exactly filling
  `sequence_length` -- see §5). `audit_range` doesn't assume that; it sums
  each sample's actual served-token count from `token_span_ids` and
  divides by allocated capacity, so a real regression that started
  under-filling windows would show up here as a number below `1.0`, not
  silently pass.
- **"Useful loss-bearing tokens" can only be *estimated* from ledger data
  alone.** The consumption ledger stores `loss_mask_hash`, not the array
  itself (deliberately -- §5.8). So `audit_range` estimates useful tokens
  as "every position except each sample's structurally-guaranteed final
  masked position" -- exactly right for `concatenate_and_chop` lanes,
  an *overestimate* for `structure_preserving` lanes (their prompt-masked
  positions aren't ledger-visible). The report says this explicitly
  (`estimate_caveat`) rather than presenting an estimate as exact. Getting
  the *exact* count means recomputing the sample -- see the extension
  below, which closes this the same way `tds.replay` closes the analogous
  gap for hash verification.

#### Exact useful tokens via recomputation (extension, closed)

`tds.audit.exact_useful_tokens_range(start_step, end_step, seed, schedule,
lane_pools, manifest_store, shards_dir, microbatch_size)` is the "just
recompute it" escape hatch `estimate_caveat` points to, given its own
function rather than folded into `audit_range` so the ledger-only
function stays ledger-only (its docstring's whole point). It walks one
fresh `Packer`/`BatchAssembler` from step 0 through `end_step` -- same
reasoning as `tds.replay.replay_range`: the Packer's per-lane document
position is cumulative, so there's no jumping straight to `start_step` --
and sums the model-ready `loss_mask` array Packer actually produced, so a
structure-preserving lane's prompt-masked positions are counted exactly
rather than assumed away. It never touches the consumption ledger (no
`run_id`/`branch_id` needed): unlike `replay_range`, which recomputes
*to compare against records*, this recomputes because the records
structurally can't hold what's needed (loss_mask arrays), so there's
nothing to compare against, only a real number to produce.

`scripts/run_demo.py` calls it right after `audit_range`, over the same
`[0, total_steps)` range, and adds a "Useful token accounting" evidence
row whose PASS condition is genuinely two-part: `total_served_tokens`
from the fresh recompute must equal the ledger-derived total from
`audit_range` (same deterministic stream, computed two different ways --
this doubles as an unplanned extra check that the ledger and the real
packed stream never silently drift apart), and `exact_useful_tokens` must
be `<= estimated_useful_tokens` (masking can only ever *add* zeroed
positions beyond the one structural one the estimate already accounts
for, never remove one, so the estimate is a real upper bound by
construction, not just usually true).

**Verified against both corpora**: on the toy and real corpus (neither
`--num-steps 50` demo run touches, by chance, a structure-preserving
document with a real detected prompt boundary -- see below), the exact
and estimated counts came out identical, which is itself the correct
degenerate-case answer. To confirm the *general* case actually diverges
and this isn't just an untested code path, `tests/test_audit.py` adds a
synthetic structure-preserving fixture (reusing `test_packer.py`'s own
`TestStructurePreservingMasking` scenario: prompt token count 3, response
4, on an 8-token window) where `exact_useful_tokens` (4 per sample) is
provably less than the `concatenate_and_chop` estimate (7 per sample).

Separately, inspecting the real corpus's own "instruction" lane
manifests turned up a fact worth recording here rather than losing it:
16 of its 17 documents have no detectable `"output:"` marker at all and
degrade to `response_start_token == start_token` (zero prompt tokens,
per `tds/shard_builder.py`'s documented graceful-degradation path) --
only one document (`doc-000359` in `shard-000022`, ~838 prompt tokens)
carries a real, non-degenerate split. That document happens to fall
outside both the 50-step and a 300-step demo window (it's `instruction`,
a scarce, low-share lane); a full 1,659-step run would eventually reach
it and show `exact_useful_tokens < estimated_useful_tokens` for real, not
just in the synthetic test.
- **Throughput cannot come from ledgers at all.** Wall-clock time is
  inherently a live-run fact; `StepTiming`/`throughput_report` take
  timings recorded during the actual run (`scripts/run_demo.py` wraps
  each `run_training_step` call in `time.perf_counter()`), not something
  reconstructed after the fact.

#### Fork/branch lineage traversal (extension, closed)

§11's own note punted this here explicitly: "reconstructing 'one branch's
full lineage across a fork' is an Audit-level concern... not something
[Fork] needs to solve." `audit_range` took that literally -- it reports
on exactly one `branch_id`'s own ledger entries, which for a forked
branch is *only its post-fork history*. `fork_branch` never duplicates a
parent's pre-fork entries under the child's `branch_id` (§11's own
`test_branches_share_history_up_to_fork_and_diverge_after` proves this:
the forked branch's ledger has zero entries at or before the fork step),
so calling `audit_range` on a forked branch alone silently under-reports
-- or, worse, if asked for a range starting at 0, raises `ValueError`
outright, since there's nothing there to find.

Two new pieces in `tds/audit.py` close this:

- **`branch_lineage(checkpoints, run_id, branch_id)`** walks a branch's
  ancestry backwards to the run's root. The key fact this relies on:
  `fork_branch` writes exactly one checkpoint that carries
  `parent_branch_id`/`fork_step` metadata for a given branch -- its very
  first one -- so `CheckpointManager.earliest_step` (new, mirroring the
  existing `latest_step`) always lands on the checkpoint holding the
  lineage this branch needs. Walk: load that checkpoint's metadata, note
  its parent, jump to the parent, repeat until a checkpoint has no
  parent (the root). Returns the chain oldest-first, so index order
  doubles as ancestry order.
- **`audit_branch_lineage(run_id, branch_id, end_step, ...)`** stitches
  consumption-ledger entries across every branch_id the lineage passes
  through, splitting at each fork boundary exactly where `fork_branch`
  itself split history: a branch's segment runs from
  `next_step_after_checkpoint`'s value (or 0, for the root) through
  either the *next* fork in the chain or `end_step` for the target
  branch itself. The actual audit math (shards/lanes touched, packing
  utilization, the useful-tokens estimate) is unchanged -- refactored out
  of `audit_range` into a shared `_aggregate_consumption_entries(entries,
  schedule)` so both functions run the identical computation, one over a
  single branch's entries, one over a stitched multi-branch list. For a
  branch that was never forked, `audit_branch_lineage` and `audit_range`
  produce identical reports (a real invariant test, not just intended
  behavior).

`scripts/run_demo.py` calls `audit_branch_lineage(run_id, "fork-1",
fork_end_step, ...)` right after the fork training loop and adds a "Fork
lineage" evidence row. On the real corpus this reconstructs `fork-1`'s
224-sample complete history across `['main', 'fork-1']` -- the 26 shared
pre-fork steps (steps [0, 25], living under `main`) plus `fork-1`'s own 2
post-fork steps, correctly stitched at the fork boundary without ever
double-counting or missing the shared prefix.

### `tds/evidence.py`: structurally unable to hardcode a result

The assignment is explicit: "the evidence bundle must be generated by the
implementation. Hardcoded evidence will not be accepted." `build_evidence_bundle`
enforces this by construction rather than by discipline: it takes a list
of already-computed `EvidenceRow(requirement, passed, evidence)` values
and only ever assembles/formats them -- there's no code path inside this
module that could decide "PASS" on its own. Every row's `passed` value in
`scripts/run_demo.py` is wired directly to a real check that already ran
earlier in the same script (a `ResumeVerificationResult.matched`, an
actual `len(blocked) > 0`, an `audit_report` field), so evidence.json can
be traced back to the exact computation that produced each verdict.

### `scripts/run_demo.py`: ties every component built this session together

Sequence: eval registry -> pipeline (tokenizer/shards/manifests, filtered
by the firewall) -> mixture compilation -> OPUS selection -> N real
training steps (writing consumption + learning ledgers, one checkpoint
saved partway through) -> crash simulation + resume (brand-new model/
optimizer/Packer, restoring only from the checkpoint + frozen config,
verified against the already-recorded ledger) -> replay of the
pre-checkpoint history -> a fork diverging on a different seed -> audit +
throughput -> the evidence bundle. Every `[event]`/`[PASS]` log line in
the assignment's own execution-log spec is emitted, in the order they
actually happen (the eval firewall runs *before* "shards created," since
shards are only ever built from the admitted set -- the assignment's own
listed order isn't necessarily temporal).

**A real scale correction made after actually running it**: the real
corpus's compiled schedule has 1,659 steps. An uncapped default would
mean training all 1,659 steps, *then* `replay_range` re-walking roughly
half of them again, *then* the fork re-walking a similar span a third
time -- not a "quick demo" by any reasonable measure, and not the point
of this assignment ("the goal is not scale"). Caught by actually running
it against the real corpus (it didn't finish in a reasonable time) rather
than reasoning about it in the abstract. Fixed with a default cap
(`min(schedule.total_steps, 50)`), overridable via `--num-steps` for
anyone who wants a full run. The toy corpus's schedule (27 steps) was
never affected and runs its full, uncapped schedule by default.

### Verified

Both corpora, full command: `python scripts/run_demo.py --corpus toy` /
`--corpus real`. Real corpus, capped at 50 steps, completes in ~15
seconds and produces `overall: PASS` with all nine evidence rows passing.
Toy corpus runs its full 27-step schedule and also passes all nine.

**Determinism check**: ran the real-corpus demo twice independently and
diffed `evidence.json`. Every field was byte-for-byte identical --
tokenizer hash, OPUS decisions hash, resumed batch ids, mixture-compliance
shares (down to floating-point deltas), learning-ledger entry count --
*except* the throughput numbers (`7569.6` vs. `7703.6` tokens/sec), which
correctly differ, since wall-clock timing is a real hardware measurement
that should vary between runs, not something the rest of this project's
reproducibility guarantee was ever supposed to cover.

### Tests (`tests/test_audit.py`, `tests/test_evidence.py`)

- `audit_range`: correctly reports shards/lanes/stages touched and total
  samples for a range; `packing_utilization` is exactly `1.0` (verified,
  not assumed) with `total_capacity_tokens == total_served_tokens`;
  estimated useful tokens excludes exactly one position per sample;
  narrowing the range only includes those steps' data; rejects an invalid
  range, a range with nothing recorded, and an unknown branch.
- `mixture_compliance_report`: planned/actual/delta reported per lane and
  internally consistent (`delta == actual - planned`); realized shares
  land in the expected neighborhood of the target mixture (checked with a
  tolerance appropriate to a small stochastic sample, not exact equality
  -- see `tds/cursor.py`'s own convergence-over-many-draws property).
- `exact_useful_tokens_range`: matches `audit_range`'s estimate exactly
  when no structure-preserving masking applies; a synthetic
  structure-preserving fixture proves genuine divergence (4 exact vs. 7
  estimated per sample); breaks down by shard and lane; narrowing the
  range only includes those steps; rejects an invalid range.
- `throughput_report`: aggregates tokens/sec correctly; rejects empty
  input.
- `branch_lineage`: an unforked branch is its own single-element chain; a
  single fork produces a two-link root-then-child chain; a *chain* of two
  forks (fork-1 forked from main, fork-2 forked from fork-1 later)
  produces the correct three-link chain with the right `fork_step`/
  `parent_branch_id` at each link; raises for a branch with no checkpoint
  recorded at all (an unresolvable lineage, distinct from an empty one).
- `audit_branch_lineage`: for an unforked branch, produces a report
  identical to `audit_range` field-for-field, plus the lineage list; for
  a forked branch, correctly pulls in the parent's pre-fork entries
  (verified both by an exact expected sample count and by showing it's
  strictly more than what reading the forked branch's own ledger alone
  would give); rejects a non-positive `end_step`; raises when a lineage
  resolves but its ledger has no entries in range.
- `build_evidence_bundle`/`write_evidence_bundle`: correct PASS/FAIL
  string mapping; overall is PASS only when every row passes (and
  vacuously PASS for zero rows); markdown contains every requirement,
  result, and evidence string; writing creates parent directories and
  produces `evidence.json`/`evidence.md` with matching content.

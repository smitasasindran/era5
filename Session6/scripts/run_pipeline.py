#!/usr/bin/env python
"""Run the full pipeline end to end, from a config file: tokenizer -> shards
-> (optionally) mixture schedule.

    python scripts/run_pipeline.py                                  # configs/pipeline.yaml (real corpus)
    python scripts/run_pipeline.py --config configs/pipeline_toy.yaml # tiny hand-authored fixture

All tunables (corpus, output dirs, vocab size, shard token budget, packing
policy) live in the config file -- see configs/pipeline.yaml and
configs/pipeline_toy.yaml for the two ready-made profiles. To run a one-off
variation (e.g. best_fit packing), copy one of those files and point
--config at your copy, rather than passing flags here.

If the config's `curriculum` field names a CurriculumConfig YAML (both
ready-made profiles set this), the mixture schedule is compiled and frozen
as the pipeline's last step, against the manifests this same run just
built -- not whatever happens to be sitting in `data/manifests/` from some
earlier run. That removes the old footgun where compiling a curriculum
required first remembering to (re)run this script against the matching
corpus (see IMPLEMENTATION_NOTES.md §3). Leave `curriculum` blank to skip
this step, or run scripts/compile_mixture.py separately to recompile a
curriculum without rebuilding shards.

Because the tokenizer is trained *from* whichever corpus the config names,
switching corpora necessarily means a new frozen tokenizer (a new
tokenizer_hash). That makes any shards left over from a *previous* corpus
invalid alongside the new ones, and the manifest store would (correctly)
refuse to mix them in -- shard_id "shard-000000" from a toy-corpus build
and "shard-000000" from a real-corpus build have different content_hash
values, and appending one where the other already exists is exactly the
mutation the manifest store exists to reject. So every run here clears the
config's tokenizer_dir, shards_dir, and manifests_dir first, then rebuilds
all three fully from the specified corpus.

This is a development/validation convenience for iterating on one corpus
at a time -- not the same thing as a finished real run's artifacts being
mutable, which they never are. If you want two profiles' builds to coexist
on disk simultaneously, give them different shards_dir/manifests_dir/
tokenizer_dir in their config files.
"""

import argparse
import dataclasses
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tds.config import CurriculumConfig, DEFAULT_CONFIG_PATH, PipelineConfig  # noqa: E402
from tds.corpus import load_corpus  # noqa: E402
from tds.eval_firewall import filter_training_documents  # noqa: E402
from tds.eval_registry import EvalRegistry  # noqa: E402
from tds.manifest_store import ManifestStore  # noqa: E402
from tds.mixture_compiler import (  # noqa: E402
    ScarcityError,
    compile_curriculum,
    freeze_schedule,
    print_schedule_report,
)
from tds.shard_builder import ShardBuilderConfig, build_shards  # noqa: E402
from tds.tokenizer_utils import train_tokenizer  # noqa: E402
from tds.toy_corpus import DEFAULT_TOY_CORPUS_PATH, write_toy_corpus  # noqa: E402


def resolve_corpus_path(corpus_value: str) -> Path:
    if corpus_value == "toy":
        return write_toy_corpus(DEFAULT_TOY_CORPUS_PATH)
    path = Path(corpus_value)
    return path if path.is_absolute() else ROOT / path


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to a pipeline config YAML file (see configs/).",
    )
    args = parser.parse_args()

    config = PipelineConfig.from_yaml(args.config).resolved(root=ROOT)
    print(f"Config: {args.config}")
    print(
        f"  corpus={config.corpus} vocab_size={config.vocab_size} "
        f"shard_token_budget={config.shard_token_budget} packing_policy={config.packing_policy}"
    )

    corpus_path = resolve_corpus_path(config.corpus)
    if not corpus_path.exists():
        parser.error(f"corpus file not found: {corpus_path}")

    for d in (config.tokenizer_dir, config.shards_dir, config.manifests_dir):
        if Path(d).exists():
            shutil.rmtree(d)
    print(f"Cleared {config.tokenizer_dir}, {config.shards_dir}, {config.manifests_dir}")

    documents = load_corpus(corpus_path)
    print(f"Loaded {len(documents)} documents from {corpus_path}")

    # Eval/test firewall: runs before anything else touches these documents
    # -- neither the tokenizer nor the shard builder below ever sees the
    # original `documents` list again, only `admitted`. A held-out document
    # is blocked by content hash, so it's caught even if some future corpus
    # gives it a different document_id or source.
    registry = EvalRegistry(config.eval_registry_dir)
    admitted, blocked = filter_training_documents(documents, registry)

    Path(config.manifests_dir).mkdir(parents=True, exist_ok=True)
    blocked_log_path = Path(config.manifests_dir) / "eval_firewall_events.jsonl"
    with open(blocked_log_path, "w") as f:
        for event in blocked:
            f.write(json.dumps(dataclasses.asdict(event)) + "\n")

    if blocked:
        print(f"[PASS] eval_shard_blocked: {len(blocked)} document(s) blocked")
        for event in blocked:
            print(
                f"    {event.document_id} (source_document_id={event.source_document_id}) "
                f"matches benchmark={event.benchmark_id} version={event.version_tag}"
            )
    else:
        print("[INFO] eval firewall active: 0 documents blocked (no overlap with the registry found)")
    print(f"Admitted {len(admitted)}/{len(documents)} documents to the training pool")

    tokenizer, tok_manifest = train_tokenizer(
        (d.text for d in admitted), config.tokenizer_dir, vocab_size=config.vocab_size
    )
    print(f"Trained tokenizer -> {config.tokenizer_dir}/tokenizer.json")
    print(f"tokenizer_hash: {tok_manifest['tokenizer_hash']}")

    shard_config = ShardBuilderConfig(
        shard_token_budget=config.shard_token_budget,
        shards_dir=config.shards_dir,
        manifests_dir=config.manifests_dir,
        packing_policy=config.packing_policy,
    )
    manifests = build_shards(admitted, tokenizer, tok_manifest["tokenizer_hash"], shard_config)

    total_tokens = sum(m["token_count"] for m in manifests)
    lanes = sorted({m["capability_lane"] for m in manifests})
    print(f"\nBuilt {len(manifests)} shards ({config.packing_policy} packing), {total_tokens} tokens total")
    print(f"Lanes: {lanes}")
    for lane in lanes:
        lane_manifests = [m for m in manifests if m["capability_lane"] == lane]
        lane_tokens = sum(m["token_count"] for m in lane_manifests)
        capacity = len(lane_manifests) * config.shard_token_budget
        utilization = lane_tokens / capacity if capacity else 0.0
        print(
            f"  {lane}: {len(lane_manifests)} shards, {lane_tokens} tokens, "
            f"{utilization:.1%} of allocated shard capacity"
        )

    if not config.curriculum:
        return

    print(f"\nCompiling mixture schedule from {config.curriculum}")
    curriculum_config = CurriculumConfig.from_yaml(config.curriculum).resolved(root=ROOT)
    # Compile against the manifests this run just built, not whatever
    # manifests_dir the curriculum YAML happens to name -- that's what
    # makes this step no longer depend on remembering to run the matching
    # profile first (see IMPLEMENTATION_NOTES.md §3).
    curriculum_config.manifests_dir = config.manifests_dir

    lane_available_tokens = ManifestStore(curriculum_config.manifests_dir).lane_token_totals()
    try:
        schedule = compile_curriculum(
            curriculum_config.stages,
            lane_available_tokens,
            curriculum_config.global_batch_size,
            curriculum_config.scarcity_policy,
        )
    except ScarcityError as e:
        parser.error(str(e))
        return  # unreachable, parser.error exits, but keeps type-checkers happy

    print()
    print_schedule_report(schedule)

    manifest = freeze_schedule(schedule, curriculum_config.output_path)
    print(f"Wrote compiled schedule -> {curriculum_config.output_path}")
    print(f"  schedule_hash={manifest['schedule_hash']}")


if __name__ == "__main__":
    main()

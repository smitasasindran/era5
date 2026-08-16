#!/usr/bin/env python
"""Run the tokenizer + shard-builder pipeline end to end, from a config file.

    python scripts/run_pipeline.py                                  # configs/pipeline.yaml (real corpus)
    python scripts/run_pipeline.py --config configs/pipeline_toy.yaml # tiny hand-authored fixture

All tunables (corpus, output dirs, vocab size, shard token budget, packing
policy) live in the config file -- see configs/pipeline.yaml and
configs/pipeline_toy.yaml for the two ready-made profiles. To run a one-off
variation (e.g. best_fit packing), copy one of those files and point
--config at your copy, rather than passing flags here.

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
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tds.config import DEFAULT_CONFIG_PATH, PipelineConfig  # noqa: E402
from tds.corpus import load_corpus  # noqa: E402
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

    tokenizer, tok_manifest = train_tokenizer(
        (d.text for d in documents), config.tokenizer_dir, vocab_size=config.vocab_size
    )
    print(f"Trained tokenizer -> {config.tokenizer_dir}/tokenizer.json")
    print(f"tokenizer_hash: {tok_manifest['tokenizer_hash']}")

    shard_config = ShardBuilderConfig(
        shard_token_budget=config.shard_token_budget,
        shards_dir=config.shards_dir,
        manifests_dir=config.manifests_dir,
        packing_policy=config.packing_policy,
    )
    manifests = build_shards(documents, tokenizer, tok_manifest["tokenizer_hash"], shard_config)

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


if __name__ == "__main__":
    main()

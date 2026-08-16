#!/usr/bin/env python
"""Run the tokenizer + shard-builder pipeline end to end, for one corpus.

    python scripts/run_pipeline.py                  # real corpus (default)
    python scripts/run_pipeline.py --corpus toy       # tiny hand-authored fixture
    python scripts/run_pipeline.py --corpus path/to/other.parquet

Always writes to the same data/tokenizer, data/shards, data/manifests --
there is only ever one "current" pipeline build, regardless of which
corpus produced it.

Because the tokenizer is trained *from* whichever corpus is passed in,
switching corpora necessarily means a new frozen tokenizer (a new
tokenizer_hash). That makes any shards left over from a *previous* corpus
invalid alongside the new ones, and the manifest store would (correctly)
refuse to mix them in -- shard_id "shard-000000" from a toy-corpus build
and "shard-000000" from a real-corpus build have different content_hash
values, and appending one where the other already exists is exactly the
mutation the manifest store exists to reject. So every run here clears
data/tokenizer, data/shards, and data/manifests first, then rebuilds all
three fully from the specified corpus.

This is a development/validation convenience for iterating on one corpus
at a time -- not the same thing as a finished real run's artifacts being
mutable, which they never are. If you want two corpora's builds to coexist
on disk simultaneously, point --shards-dir/--manifests-dir/--tokenizer-dir
at separate locations explicitly.
"""

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tds.corpus import load_corpus  # noqa: E402
from tds.shard_builder import ShardBuilderConfig, build_shards  # noqa: E402
from tds.tokenizer_utils import train_tokenizer  # noqa: E402
from tds.toy_corpus import DEFAULT_TOY_CORPUS_PATH, write_toy_corpus  # noqa: E402

DEFAULT_REAL_CORPUS = ROOT / "data" / "corpus" / "small_shard.parquet"
DEFAULT_TOKENIZER_DIR = ROOT / "data" / "tokenizer"
DEFAULT_SHARDS_DIR = ROOT / "data" / "shards"
DEFAULT_MANIFESTS_DIR = ROOT / "data" / "manifests"


def resolve_corpus_path(corpus_arg: str) -> Path:
    if corpus_arg == "toy":
        return write_toy_corpus(DEFAULT_TOY_CORPUS_PATH)
    return Path(corpus_arg)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--corpus",
        default=str(DEFAULT_REAL_CORPUS),
        help="Path to a parquet file, or 'toy' for the small hand-authored fixture.",
    )
    parser.add_argument("--tokenizer-dir", default=str(DEFAULT_TOKENIZER_DIR))
    parser.add_argument("--shards-dir", default=str(DEFAULT_SHARDS_DIR))
    parser.add_argument("--manifests-dir", default=str(DEFAULT_MANIFESTS_DIR))
    parser.add_argument("--vocab-size", type=int, default=8000)
    parser.add_argument("--shard-token-budget", type=int, default=50_000)
    args = parser.parse_args()

    corpus_path = resolve_corpus_path(args.corpus)
    if not corpus_path.exists():
        parser.error(f"corpus file not found: {corpus_path}")

    for d in (Path(args.tokenizer_dir), Path(args.shards_dir), Path(args.manifests_dir)):
        if d.exists():
            shutil.rmtree(d)
    print(f"Cleared {args.tokenizer_dir}, {args.shards_dir}, {args.manifests_dir}")

    documents = load_corpus(corpus_path)
    print(f"Loaded {len(documents)} documents from {corpus_path}")

    tokenizer, tok_manifest = train_tokenizer(
        (d.text for d in documents), args.tokenizer_dir, vocab_size=args.vocab_size
    )
    print(f"Trained tokenizer -> {args.tokenizer_dir}/tokenizer.json")
    print(f"tokenizer_hash: {tok_manifest['tokenizer_hash']}")
    print(f"vocab_size: {tok_manifest['vocab_size']}")

    config = ShardBuilderConfig(
        shard_token_budget=args.shard_token_budget,
        shards_dir=args.shards_dir,
        manifests_dir=args.manifests_dir,
    )
    manifests = build_shards(documents, tokenizer, tok_manifest["tokenizer_hash"], config)

    total_tokens = sum(m["token_count"] for m in manifests)
    lanes = sorted({m["capability_lane"] for m in manifests})
    print(f"\nBuilt {len(manifests)} shards, {total_tokens} tokens total")
    print(f"Lanes: {lanes}")
    for lane in lanes:
        lane_manifests = [m for m in manifests if m["capability_lane"] == lane]
        lane_tokens = sum(m["token_count"] for m in lane_manifests)
        print(f"  {lane}: {len(lane_manifests)} shards, {lane_tokens} tokens")


if __name__ == "__main__":
    main()

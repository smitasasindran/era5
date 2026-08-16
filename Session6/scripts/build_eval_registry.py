#!/usr/bin/env python
"""Register held-out content in the eval firewall's registry.

    python scripts/build_eval_registry.py --config configs/eval_registry.yaml
    python scripts/build_eval_registry.py --config configs/eval_registry_toy.yaml

Loads a corpus, looks up specific documents by their internal document_id,
and registers each one's content hash as held-out under a benchmark_id --
so run_pipeline.py's eval firewall will refuse to ever let that exact
content into a training shard, regardless of what document_id/source a
future corpus file happens to give it.

Unlike the tokenizer or shards, this registry is NOT cleared between
runs -- it's meant to accumulate held-out fingerprints across benchmarks
(and corpora) over time. Re-running with the same benchmark_id and
documents is a no-op; registering a *different* benchmark under a content
hash that's already claimed by another benchmark is an error, not a
silent overwrite.
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tds.config import DEFAULT_EVAL_REGISTRY_CONFIG_PATH, EvalRegistryConfig  # noqa: E402
from tds.corpus import load_corpus  # noqa: E402
from tds.eval_registry import EvalRegistry  # noqa: E402
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
    parser.add_argument("--config", default=str(DEFAULT_EVAL_REGISTRY_CONFIG_PATH))
    args = parser.parse_args()

    config = EvalRegistryConfig.from_yaml(args.config).resolved(root=ROOT)

    corpus_path = resolve_corpus_path(config.corpus)
    documents = load_corpus(corpus_path)
    by_id = {d.document_id: d for d in documents}

    missing = [doc_id for doc_id in config.held_out_document_ids if doc_id not in by_id]
    if missing:
        parser.error(f"document_id(s) not found in corpus {corpus_path}: {missing}")

    registry = EvalRegistry(config.registry_dir)
    for doc_id in config.held_out_document_ids:
        doc = by_id[doc_id]
        record = registry.register_text(
            doc.text, benchmark_id=config.benchmark_id, version_tag=config.version_tag
        )
        print(f"Registered {doc_id} -> {record['content_hash']} under benchmark {config.benchmark_id!r}")

    print(f"\nEval registry at {config.registry_dir} now holds {len(registry.all())} held-out hash(es) total.")


if __name__ == "__main__":
    main()

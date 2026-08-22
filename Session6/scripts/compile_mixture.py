#!/usr/bin/env python
"""Recompile a curriculum into an executable mixture schedule, without
rebuilding shards.

    python scripts/compile_mixture.py                                     # configs/curriculum.yaml (real corpus)
    python scripts/compile_mixture.py --config configs/curriculum_toy.yaml  # toy corpus

run_pipeline.py already runs this step automatically as part of a full
pipeline run (when its config's `curriculum` field is set), compiling
against the manifests that same run just built. Use this script standalone
only when iterating on a curriculum YAML against shards that already
exist on disk, without rerunning the tokenizer/shard-builder step.

Reads the shard manifests already built by scripts/run_pipeline.py (for
whichever corpus the curriculum targets) to find out how many tokens each
capability lane actually has available, checks that against each stage's
planned lane shares -- cumulatively across stages, since a lane's supply is
shared across the whole curriculum -- and writes the compiled result to the
config's output_path. Run scripts/run_pipeline.py against the matching
corpus first; this script does not build shards itself.

Part of the standalone per-stage workflow (see run_pipeline.py's own
docstring) -- not needed for, and not read by, scripts/run_demo.py.
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tds.config import DEFAULT_CURRICULUM_CONFIG_PATH, CurriculumConfig  # noqa: E402
from tds.manifest_store import ManifestStore  # noqa: E402
from tds.mixture_compiler import (  # noqa: E402
    ScarcityError,
    compile_curriculum,
    freeze_schedule,
    print_schedule_report,
)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default=str(DEFAULT_CURRICULUM_CONFIG_PATH))
    args = parser.parse_args()

    config = CurriculumConfig.from_yaml(args.config).resolved(root=ROOT)
    print(f"Config: {args.config}")
    print(f"  global_batch_size={config.global_batch_size} scarcity_policy={config.scarcity_policy}")

    store = ManifestStore(config.manifests_dir)
    lane_available_tokens = store.lane_token_totals()
    if not lane_available_tokens:
        parser.error(
            f"no shard manifests found under {config.manifests_dir} -- "
            "run scripts/run_pipeline.py against the matching corpus first"
        )
    print(f"Available supply (from {config.manifests_dir}): {lane_available_tokens}")

    try:
        schedule = compile_curriculum(
            config.stages, lane_available_tokens, config.global_batch_size, config.scarcity_policy
        )
    except ScarcityError as e:
        parser.error(str(e))
        return  # unreachable, parser.error exits, but keeps type-checkers happy

    print()
    print_schedule_report(schedule)

    manifest = freeze_schedule(schedule, config.output_path)
    print(f"Wrote compiled schedule -> {config.output_path}")
    print(f"  schedule_hash={manifest['schedule_hash']}")


if __name__ == "__main__":
    main()

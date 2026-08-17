#!/usr/bin/env python
"""Compile curriculum stages into an executable mixture schedule.

    python scripts/compile_mixture.py                                     # configs/curriculum.yaml (real corpus)
    python scripts/compile_mixture.py --config configs/curriculum_toy.yaml  # toy corpus

Reads the shard manifests already built by scripts/run_pipeline.py (for
whichever corpus the curriculum targets) to find out how many tokens each
capability lane actually has available, checks that against each stage's
planned lane shares -- cumulatively across stages, since a lane's supply is
shared across the whole curriculum -- and writes the compiled result to the
config's output_path. Run scripts/run_pipeline.py against the matching
corpus first; this script does not build shards itself.
"""

import argparse
import dataclasses
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tds.config import DEFAULT_CURRICULUM_CONFIG_PATH, CurriculumConfig  # noqa: E402
from tds.manifest_store import ManifestStore  # noqa: E402
from tds.mixture_compiler import ScarcityError, compile_curriculum  # noqa: E402


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
    for cs in schedule.stages:
        print(
            f"Stage {cs.stage.stage!r}: steps [{cs.step_start}, {cs.step_end}), "
            f"span {cs.stage_token_span} tokens, sequence_length={cs.stage.sequence_length}"
        )
        for lane, plan in sorted(cs.lane_plans.items()):
            repeat_note = (
                f", repeat_factor={plan.repeat_factor:.2f}x"
                if plan.repeat_factor is not None and plan.repeat_factor > 1.0 + 1e-9
                else ""
            )
            print(
                f"    {lane:<14} target={plan.target_weight:.2%} effective={plan.effective_weight:.2%} "
                f"[{plan.status}]{repeat_note}"
            )
        if cs.unallocated_share > 1e-9:
            print(f"    [WARN] unallocated_share={cs.unallocated_share:.2%} of this stage is unmet")
        print()

    output_path = Path(config.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(dataclasses.asdict(schedule), f, indent=2)
    print(f"Wrote compiled schedule -> {output_path}")


if __name__ == "__main__":
    main()

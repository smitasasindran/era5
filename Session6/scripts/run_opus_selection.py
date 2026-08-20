#!/usr/bin/env python
"""Run OPUS candidate selection: score every document in every lane's pool
under a model snapshot, and freeze the accept/reject/defer decisions.

    python scripts/run_opus_selection.py                                # configs/opus.yaml (real corpus)
    python scripts/run_opus_selection.py --config configs/opus_toy.yaml  # toy corpus

If the config's `enabled` field is false, this writes a pass-through
decision log instead -- every candidate accepted, no model constructed or
run at all -- so downstream code always has a frozen OPUS artifact to load
the same way, whether or not selection is actually turned on.

Reads the shard manifests already built by scripts/run_pipeline.py (for
whichever corpus this config targets); this script does not build shards
itself.
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tds.config import DEFAULT_OPUS_CONFIG_PATH, OpusConfig  # noqa: E402
from tds.manifest_store import ManifestStore  # noqa: E402
from tds.model import ToyTransformer, ToyTransformerConfig  # noqa: E402
from tds.opus import apply_opus_selection, freeze_opus_selection  # noqa: E402
from tds.tokenizer_utils import load_frozen_tokenizer  # noqa: E402


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default=str(DEFAULT_OPUS_CONFIG_PATH))
    args = parser.parse_args()

    config = OpusConfig.from_yaml(args.config).resolved(root=ROOT)
    print(f"Config: {args.config}")
    print(f"  enabled={config.enabled}")

    store = ManifestStore(config.manifests_dir)
    lane_pools = store.document_pool_by_lane()
    if not lane_pools:
        parser.error(
            f"no shard manifests found under {config.manifests_dir} -- "
            "run scripts/run_pipeline.py against the matching corpus first"
        )
    total_candidates = sum(len(pool) for pool in lane_pools.values())
    print(f"Candidates: {total_candidates} documents across {len(lane_pools)} lanes")

    model = None
    if config.enabled:
        _, tok_manifest = load_frozen_tokenizer(config.tokenizer_dir)
        model_config = ToyTransformerConfig(
            vocab_size=tok_manifest["vocab_size"],
            max_sequence_length=config.max_sequence_length,
            d_model=config.d_model,
            n_layers=config.n_layers,
            n_heads=config.n_heads,
            d_ff=config.d_ff,
        )
        model = ToyTransformer(model_config, seed=config.seed)
        if config.checkpoint_path:
            import torch

            payload = torch.load(config.checkpoint_path, weights_only=False)
            model.load_state_dict(payload["model_state_dict"])
            print(f"  scoring with checkpoint: {config.checkpoint_path}")
        else:
            print(f"  scoring with a freshly-initialized model (seed={config.seed}, no real checkpoint)")

    _, decisions = apply_opus_selection(
        lane_pools,
        model,
        store,
        config.shards_dir,
        config.max_sequence_length,
        config.reject_below,
        config.defer_above,
        protected_lanes=frozenset(config.protected_lanes),
        enabled=config.enabled,
    )

    by_status = {"accepted": 0, "rejected": 0, "deferred": 0}
    by_lane_status = {}
    for d in decisions:
        by_status[d.status] += 1
        by_lane_status.setdefault(d.capability_lane, {"accepted": 0, "rejected": 0, "deferred": 0})
        by_lane_status[d.capability_lane][d.status] += 1

    print()
    print(f"accepted={by_status['accepted']} rejected={by_status['rejected']} deferred={by_status['deferred']}")
    for lane in sorted(by_lane_status):
        counts = by_lane_status[lane]
        rescued = sum(1 for d in decisions if d.capability_lane == lane and d.protected_floor_override)
        rescue_note = f", {rescued} floor-rescued" if rescued else ""
        print(
            f"  {lane:<14} accepted={counts['accepted']} rejected={counts['rejected']} "
            f"deferred={counts['deferred']}{rescue_note}"
        )

    manifest = freeze_opus_selection(decisions, config.output_path)
    print(f"\nWrote OPUS decisions -> {config.output_path}")
    print(f"  decisions_hash={manifest['decisions_hash']}")


if __name__ == "__main__":
    main()

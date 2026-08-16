"""Pipeline configuration: one YAML file per profile, one field per tunable.

Replaces what used to be a long list of run_pipeline.py CLI flags -- every
parameter for a given run (which corpus, tokenizer/shard/manifest output
dirs, vocab size, shard token budget, packing policy) now lives in a single
readable file instead of being reconstructed from a command line.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = ROOT / "configs" / "pipeline.yaml"


@dataclass
class PipelineConfig:
    corpus: str = "data/corpus/small_shard.parquet"  # path, or the literal "toy"
    tokenizer_dir: str = "data/tokenizer"
    shards_dir: str = "data/shards"
    manifests_dir: str = "data/manifests"
    vocab_size: int = 8000
    shard_token_budget: int = 50_000
    packing_policy: str = "greedy"  # see tds.shard_builder.PACKING_POLICIES

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PipelineConfig":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"config file not found: {path}")

        with open(path) as f:
            raw = yaml.safe_load(f) or {}

        known = {f.name for f in fields(cls)}
        unknown = set(raw) - known
        if unknown:
            raise ValueError(
                f"unknown key(s) in {path}: {sorted(unknown)} -- known keys are {sorted(known)}"
            )

        return cls(**raw)

    def resolved(self, root: Path = ROOT) -> "PipelineConfig":
        """Return a copy with relative directory paths resolved against `root`,
        so behavior doesn't depend on the caller's current working directory.
        `corpus` is left untouched here -- it may be the "toy" sentinel
        rather than a path; callers resolve it separately."""
        values = asdict(self)
        for key in ("tokenizer_dir", "shards_dir", "manifests_dir"):
            p = Path(values[key])
            values[key] = str(p if p.is_absolute() else root / p)
        return PipelineConfig(**values)

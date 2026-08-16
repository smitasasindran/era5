"""Pipeline configuration: one YAML file per profile, one field per tunable.

Replaces what used to be a long list of CLI flags -- every parameter for a
given run now lives in a single readable file instead of being
reconstructed from a command line.

Two config shapes:

- `PipelineConfig` drives scripts/run_pipeline.py (corpus -> tokenizer ->
  shards, filtered through the eval firewall).
- `EvalRegistryConfig` drives scripts/build_eval_registry.py (which
  documents get registered as held-out, under which benchmark).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import List

import yaml

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = ROOT / "configs" / "pipeline.yaml"
DEFAULT_EVAL_REGISTRY_CONFIG_PATH = ROOT / "configs" / "eval_registry.yaml"


def _from_yaml(cls, path: str | Path):
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


def _resolved_dirs(instance, dir_fields: tuple, root: Path):
    values = asdict(instance)
    for key in dir_fields:
        p = Path(values[key])
        values[key] = str(p if p.is_absolute() else root / p)
    return type(instance)(**values)


@dataclass
class PipelineConfig:
    corpus: str = "data/corpus/small_shard.parquet"  # path, or the literal "toy"
    tokenizer_dir: str = "data/tokenizer"
    shards_dir: str = "data/shards"
    manifests_dir: str = "data/manifests"
    eval_registry_dir: str = "data/eval_registry"
    vocab_size: int = 8000
    shard_token_budget: int = 50_000
    packing_policy: str = "greedy"  # see tds.shard_builder.PACKING_POLICIES

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PipelineConfig":
        return _from_yaml(cls, path)

    def resolved(self, root: Path = ROOT) -> "PipelineConfig":
        """Return a copy with relative directory paths resolved against `root`,
        so behavior doesn't depend on the caller's current working directory.
        `corpus` is left untouched here -- it may be the "toy" sentinel
        rather than a path; callers resolve it separately."""
        return _resolved_dirs(
            self, ("tokenizer_dir", "shards_dir", "manifests_dir", "eval_registry_dir"), root
        )


@dataclass
class EvalRegistryConfig:
    corpus: str = "data/corpus/small_shard.parquet"  # path, or the literal "toy"
    registry_dir: str = "data/eval_registry"
    benchmark_id: str = "session6-holdout-v1"
    version_tag: str = "v1"
    held_out_document_ids: List[str] = field(default_factory=list)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "EvalRegistryConfig":
        return _from_yaml(cls, path)

    def resolved(self, root: Path = ROOT) -> "EvalRegistryConfig":
        return _resolved_dirs(self, ("registry_dir",), root)

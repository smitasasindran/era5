"""Pipeline configuration: one YAML file per profile, one field per tunable.

Replaces what used to be a long list of CLI flags -- every parameter for a
given run now lives in a single readable file instead of being
reconstructed from a command line.

Three config shapes:

- `PipelineConfig` drives scripts/run_pipeline.py (corpus -> tokenizer ->
  shards, filtered through the eval firewall, then optionally compiled into
  a mixture schedule if `curriculum` names a CurriculumConfig YAML).
- `EvalRegistryConfig` drives scripts/build_eval_registry.py (which
  documents get registered as held-out, under which benchmark).
- `CurriculumConfig` drives scripts/compile_mixture.py (curriculum stages ->
  a compiled schedule, checked against actual shard supply) -- also loaded
  directly by run_pipeline.py when its config's `curriculum` field is set.
- `OpusConfig` drives scripts/run_opus_selection.py (score every candidate
  document under a model snapshot and freeze accept/reject/defer
  decisions) -- `enabled: false` writes a pass-through decision log
  instead of running any model.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import List

import yaml

from .mixture_compiler import MixtureStage

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = ROOT / "configs" / "pipeline.yaml"
DEFAULT_EVAL_REGISTRY_CONFIG_PATH = ROOT / "configs" / "eval_registry.yaml"
DEFAULT_CURRICULUM_CONFIG_PATH = ROOT / "configs" / "curriculum.yaml"
DEFAULT_OPUS_CONFIG_PATH = ROOT / "configs" / "opus.yaml"


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
    curriculum: str = ""  # path to a CurriculumConfig YAML; "" skips mixture compilation

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PipelineConfig":
        return _from_yaml(cls, path)

    def resolved(self, root: Path = ROOT) -> "PipelineConfig":
        """Return a copy with relative directory paths resolved against `root`,
        so behavior doesn't depend on the caller's current working directory.
        `corpus` is left untouched here -- it may be the "toy" sentinel
        rather than a path; callers resolve it separately."""
        instance = _resolved_dirs(
            self, ("tokenizer_dir", "shards_dir", "manifests_dir", "eval_registry_dir"), root
        )
        if instance.curriculum:
            p = Path(instance.curriculum)
            instance.curriculum = str(p if p.is_absolute() else root / p)
        return instance


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


@dataclass
class CurriculumConfig:
    manifests_dir: str = "data/manifests"
    output_path: str = "data/mixture_schedule.json"
    global_batch_size: int = 8
    scarcity_policy: str = "reduce_share"  # see tds.mixture_compiler.compile_curriculum
    stages: List[MixtureStage] = field(default_factory=list)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "CurriculumConfig":
        """Custom (not `_from_yaml`): `stages` holds MixtureStage instances,
        not raw dicts, so it needs its own parsing -- and correspondingly
        `resolved()` below can't use the generic asdict()-based
        `_resolved_dirs` either, since asdict() would flatten those nested
        dataclasses into plain dicts that don't reconstruct automatically."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"config file not found: {path}")

        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        raw = dict(raw)
        stage_dicts = raw.pop("stages", [])

        known = {f.name for f in fields(cls)}
        unknown = set(raw) - known
        if unknown:
            raise ValueError(
                f"unknown key(s) in {path}: {sorted(unknown)} -- known keys are {sorted(known)}"
            )

        stage_fields = {f.name for f in fields(MixtureStage)}
        stages = []
        for i, stage_dict in enumerate(stage_dicts):
            unknown_stage_keys = set(stage_dict) - stage_fields
            if unknown_stage_keys:
                raise ValueError(
                    f"unknown key(s) in stage #{i} of {path}: {sorted(unknown_stage_keys)} -- "
                    f"known keys are {sorted(stage_fields)}"
                )
            stages.append(MixtureStage(**stage_dict))

        return cls(stages=stages, **raw)

    def resolved(self, root: Path = ROOT) -> "CurriculumConfig":
        def resolve(value: str) -> str:
            p = Path(value)
            return str(p if p.is_absolute() else root / p)

        return CurriculumConfig(
            manifests_dir=resolve(self.manifests_dir),
            output_path=resolve(self.output_path),
            global_batch_size=self.global_batch_size,
            scarcity_policy=self.scarcity_policy,
            stages=self.stages,
        )


@dataclass
class OpusConfig:
    enabled: bool = False  # false: pass-through, every candidate accepted, no model run at all
    manifests_dir: str = "data/manifests"
    shards_dir: str = "data/shards"
    tokenizer_dir: str = "data/tokenizer"
    output_path: str = "data/opus_decisions.json"
    checkpoint_path: str = ""  # "" -> a freshly-initialized model seeded by `seed`, not a real checkpoint
    seed: int = 0
    max_sequence_length: int = 128
    d_model: int = 32
    n_layers: int = 2
    n_heads: int = 2
    d_ff: int = 64
    reject_below: float = 1.0  # score below this -> "rejected" (low_proxy_utility)
    defer_above: float = 8.0  # score above this -> "deferred" (anomalous_high_loss)
    protected_lanes: List[str] = field(default_factory=list)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "OpusConfig":
        return _from_yaml(cls, path)

    def resolved(self, root: Path = ROOT) -> "OpusConfig":
        instance = _resolved_dirs(
            self, ("manifests_dir", "shards_dir", "tokenizer_dir", "output_path"), root
        )
        if instance.checkpoint_path:
            p = Path(instance.checkpoint_path)
            instance.checkpoint_path = str(p if p.is_absolute() else root / p)
        return instance

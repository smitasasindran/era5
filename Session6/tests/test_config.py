import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tds.config import CurriculumConfig, EvalRegistryConfig, PipelineConfig  # noqa: E402
from tds.mixture_compiler import MixtureStage  # noqa: E402


class TestPipelineConfig(unittest.TestCase):
    def test_defaults_when_yaml_is_empty(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "empty.yaml"
            path.write_text("")
            config = PipelineConfig.from_yaml(path)
            self.assertEqual(config, PipelineConfig())

    def test_partial_override(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "config.yaml"
            path.write_text("packing_policy: best_fit\nvocab_size: 123\n")
            config = PipelineConfig.from_yaml(path)
            self.assertEqual(config.packing_policy, "best_fit")
            self.assertEqual(config.vocab_size, 123)
            self.assertEqual(config.shard_token_budget, PipelineConfig().shard_token_budget)

    def test_unknown_key_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "config.yaml"
            path.write_text("typo_field: 1\n")
            with self.assertRaises(ValueError):
                PipelineConfig.from_yaml(path)

    def test_missing_file_is_rejected(self):
        with self.assertRaises(FileNotFoundError):
            PipelineConfig.from_yaml("/nonexistent/config.yaml")

    def test_resolved_makes_relative_dirs_absolute_against_root(self):
        config = PipelineConfig(tokenizer_dir="data/tokenizer")
        root = Path("/some/root")
        resolved = config.resolved(root=root)
        self.assertEqual(resolved.tokenizer_dir, str(root / "data" / "tokenizer"))

    def test_resolved_leaves_corpus_untouched(self):
        config = PipelineConfig(corpus="toy")
        resolved = config.resolved(root=Path("/some/root"))
        self.assertEqual(resolved.corpus, "toy")

    def test_resolved_leaves_absolute_dirs_untouched(self):
        config = PipelineConfig(shards_dir="/already/absolute")
        resolved = config.resolved(root=Path("/some/root"))
        self.assertEqual(resolved.shards_dir, "/already/absolute")

    def test_resolved_makes_curriculum_path_absolute_when_set(self):
        config = PipelineConfig(curriculum="configs/curriculum.yaml")
        resolved = config.resolved(root=Path("/some/root"))
        self.assertEqual(resolved.curriculum, str(Path("/some/root/configs/curriculum.yaml")))

    def test_resolved_leaves_blank_curriculum_untouched(self):
        config = PipelineConfig()
        resolved = config.resolved(root=Path("/some/root"))
        self.assertEqual(resolved.curriculum, "")


class TestEvalRegistryConfig(unittest.TestCase):
    def test_defaults_when_yaml_is_empty(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "empty.yaml"
            path.write_text("")
            config = EvalRegistryConfig.from_yaml(path)
            self.assertEqual(config, EvalRegistryConfig())

    def test_held_out_document_ids_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "config.yaml"
            path.write_text("held_out_document_ids:\n  - doc-000000\n  - doc-000500\n")
            config = EvalRegistryConfig.from_yaml(path)
            self.assertEqual(config.held_out_document_ids, ["doc-000000", "doc-000500"])

    def test_unknown_key_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "config.yaml"
            path.write_text("typo_field: 1\n")
            with self.assertRaises(ValueError):
                EvalRegistryConfig.from_yaml(path)

    def test_default_held_out_list_does_not_leak_between_instances(self):
        a = EvalRegistryConfig()
        a.held_out_document_ids.append("doc-000000")
        b = EvalRegistryConfig()
        self.assertEqual(b.held_out_document_ids, [])

    def test_resolved_makes_registry_dir_absolute_against_root(self):
        config = EvalRegistryConfig(registry_dir="data/eval_registry")
        resolved = config.resolved(root=Path("/some/root"))
        self.assertEqual(resolved.registry_dir, str(Path("/some/root/data/eval_registry")))

    def test_resolved_leaves_corpus_untouched(self):
        config = EvalRegistryConfig(corpus="toy")
        resolved = config.resolved(root=Path("/some/root"))
        self.assertEqual(resolved.corpus, "toy")


CURRICULUM_YAML = """
global_batch_size: 4
scarcity_policy: repeat
stages:
  - stage: foundation
    token_start: 0
    token_end: 100
    sequence_length: 8
    mixture:
      code: 0.6
      qa: 0.4
    protected_floors:
      qa: 0.1
    warmup_tokens: 10
"""


class TestCurriculumConfig(unittest.TestCase):
    def test_defaults_when_yaml_is_empty(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "empty.yaml"
            path.write_text("")
            config = CurriculumConfig.from_yaml(path)
            self.assertEqual(config, CurriculumConfig())

    def test_stages_parse_into_mixture_stage_instances(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "config.yaml"
            path.write_text(CURRICULUM_YAML)
            config = CurriculumConfig.from_yaml(path)
            self.assertEqual(len(config.stages), 1)
            self.assertIsInstance(config.stages[0], MixtureStage)
            self.assertEqual(config.stages[0].mixture, {"code": 0.6, "qa": 0.4})
            self.assertEqual(config.stages[0].protected_floors, {"qa": 0.1})
            self.assertEqual(config.global_batch_size, 4)
            self.assertEqual(config.scarcity_policy, "repeat")

    def test_microbatch_size_defaults_to_zero_meaning_unset(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "config.yaml"
            path.write_text(CURRICULUM_YAML)
            config = CurriculumConfig.from_yaml(path)
            self.assertEqual(config.microbatch_size, 0)

    def test_unknown_top_level_key_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "config.yaml"
            path.write_text("typo_field: 1\n")
            with self.assertRaises(ValueError):
                CurriculumConfig.from_yaml(path)

    def test_unknown_stage_key_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "config.yaml"
            path.write_text("stages:\n  - stage: a\n    typo_field: 1\n")
            with self.assertRaises(ValueError):
                CurriculumConfig.from_yaml(path)

    def test_default_stages_list_does_not_leak_between_instances(self):
        a = CurriculumConfig()
        a.stages.append(
            MixtureStage(stage="x", token_start=0, token_end=10, sequence_length=1, mixture={"code": 1.0})
        )
        b = CurriculumConfig()
        self.assertEqual(b.stages, [])

    def test_resolved_makes_dirs_absolute_but_leaves_stages_untouched(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "config.yaml"
            path.write_text(CURRICULUM_YAML)
            config = CurriculumConfig.from_yaml(path)

            resolved = config.resolved(root=Path("/some/root"))
            self.assertEqual(resolved.manifests_dir, str(Path("/some/root/data/manifests")))
            self.assertEqual(resolved.output_path, str(Path("/some/root/data/mixture_schedule.json")))
            self.assertEqual(resolved.stages, config.stages)

    def test_resolved_defaults_unset_microbatch_size_to_global_batch_size(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "config.yaml"
            path.write_text(CURRICULUM_YAML)  # global_batch_size: 4, no microbatch_size
            config = CurriculumConfig.from_yaml(path)
            resolved = config.resolved(root=Path("/some/root"))
            self.assertEqual(resolved.microbatch_size, 4)

    def test_resolved_keeps_explicit_microbatch_size(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "config.yaml"
            path.write_text(CURRICULUM_YAML + "microbatch_size: 2\n")
            config = CurriculumConfig.from_yaml(path)
            resolved = config.resolved(root=Path("/some/root"))
            self.assertEqual(resolved.microbatch_size, 2)


if __name__ == "__main__":
    unittest.main()

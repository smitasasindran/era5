import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tds.config import EvalRegistryConfig, PipelineConfig  # noqa: E402


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


if __name__ == "__main__":
    unittest.main()

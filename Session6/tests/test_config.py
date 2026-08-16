import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tds.config import PipelineConfig  # noqa: E402


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


if __name__ == "__main__":
    unittest.main()

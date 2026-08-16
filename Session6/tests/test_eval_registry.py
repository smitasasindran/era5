import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tds.eval_registry import EvalRegistry, EvalRegistryError  # noqa: E402
from tds.hashing import sha256_text  # noqa: E402


class TestEvalRegistry(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmpdir.name)

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_register_then_get(self):
        registry = EvalRegistry(self.dir)
        registry.register_text("hello world", benchmark_id="bench-a", version_tag="v1")
        record = registry.get(sha256_text("hello world"))
        self.assertIsNotNone(record)
        self.assertEqual(record["benchmark_id"], "bench-a")

    def test_unregistered_text_returns_none(self):
        registry = EvalRegistry(self.dir)
        self.assertIsNone(registry.get(sha256_text("never registered")))

    def test_reregistering_same_text_and_benchmark_is_idempotent(self):
        registry = EvalRegistry(self.dir)
        registry.register_text("hello world", benchmark_id="bench-a", version_tag="v1")
        registry.register_text("hello world", benchmark_id="bench-a", version_tag="v1")
        lines = self.dir.joinpath("index.jsonl").read_text().strip().splitlines()
        self.assertEqual(len(lines), 1)

    def test_same_text_under_a_different_benchmark_is_rejected(self):
        registry = EvalRegistry(self.dir)
        registry.register_text("hello world", benchmark_id="bench-a", version_tag="v1")
        with self.assertRaises(EvalRegistryError):
            registry.register_text("hello world", benchmark_id="bench-b", version_tag="v1")

    def test_reloading_from_disk_recovers_prior_entries(self):
        registry = EvalRegistry(self.dir)
        registry.register_text("hello world", benchmark_id="bench-a", version_tag="v1")

        reloaded = EvalRegistry(self.dir)
        self.assertEqual(len(reloaded.all()), 1)
        self.assertIsNotNone(reloaded.get(sha256_text("hello world")))

    def test_accumulates_across_multiple_benchmarks(self):
        registry = EvalRegistry(self.dir)
        registry.register_text("doc one", benchmark_id="bench-a", version_tag="v1")
        registry.register_text("doc two", benchmark_id="bench-b", version_tag="v1")
        self.assertEqual(len(registry.all()), 2)


if __name__ == "__main__":
    unittest.main()

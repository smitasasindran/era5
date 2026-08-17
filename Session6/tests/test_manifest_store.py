import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tds.manifest_store import ManifestStore, ManifestStoreError  # noqa: E402


def make_manifest(
    shard_id="shard-000000", content_hash="sha256:aaa", capability_lane="general_web", token_count=10
):
    return {
        "shard_id": shard_id,
        "content_hash": content_hash,
        "tokenizer_hash": "sha256:tok",
        "capability_lane": capability_lane,
        "token_count": token_count,
        "document_count": 1,
        "document_spans": [],
    }


class TestManifestStore(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmpdir.name)

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_append_then_get(self):
        store = ManifestStore(self.dir)
        store.append(make_manifest())
        self.assertEqual(store.get("shard-000000")["content_hash"], "sha256:aaa")

    def test_append_writes_index_and_per_shard_file(self):
        store = ManifestStore(self.dir)
        store.append(make_manifest())

        index_lines = (self.dir / "index.jsonl").read_text().strip().splitlines()
        self.assertEqual(len(index_lines), 1)
        self.assertEqual(json.loads(index_lines[0])["shard_id"], "shard-000000")

        per_shard = json.loads((self.dir / "shard-000000.json").read_text())
        self.assertEqual(per_shard["shard_id"], "shard-000000")

    def test_identical_reappend_is_idempotent(self):
        store = ManifestStore(self.dir)
        store.append(make_manifest())
        store.append(make_manifest())  # same shard_id, same hash -- no-op, not an error

        index_lines = (self.dir / "index.jsonl").read_text().strip().splitlines()
        self.assertEqual(len(index_lines), 1)

    def test_mutation_is_rejected(self):
        store = ManifestStore(self.dir)
        store.append(make_manifest(content_hash="sha256:aaa"))
        with self.assertRaises(ManifestStoreError):
            store.append(make_manifest(content_hash="sha256:bbb"))

    def test_reloading_from_disk_recovers_prior_entries(self):
        store = ManifestStore(self.dir)
        store.append(make_manifest())

        reloaded = ManifestStore(self.dir)
        self.assertEqual(len(reloaded.all()), 1)
        self.assertEqual(reloaded.get("shard-000000")["content_hash"], "sha256:aaa")

    def test_lane_token_totals_sums_by_lane(self):
        store = ManifestStore(self.dir)
        store.append(make_manifest("shard-000000", "sha256:a", capability_lane="code", token_count=100))
        store.append(make_manifest("shard-000001", "sha256:b", capability_lane="code", token_count=50))
        store.append(make_manifest("shard-000002", "sha256:c", capability_lane="qa", token_count=30))
        self.assertEqual(store.lane_token_totals(), {"code": 150, "qa": 30})

    def test_lane_token_totals_is_empty_for_a_fresh_store(self):
        store = ManifestStore(self.dir)
        self.assertEqual(store.lane_token_totals(), {})


if __name__ == "__main__":
    unittest.main()

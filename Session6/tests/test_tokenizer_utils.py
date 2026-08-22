import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tds.tokenizer_utils import load_frozen_tokenizer, train_tokenizer  # noqa: E402

SAMPLE_TEXTS = [
    "the quick brown fox jumps over the lazy dog",
    "def add(a, b):\n    return a + b",
    "यह एक परीक्षण वाक्य है",  # Hindi test sentence -- exercises non-ASCII bytes
    "another line of plain english text for training",
] * 20  # repeat so BPE has something to merge


class TestTrainAndLoadFrozenTokenizer(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.out_dir = Path(self.tmpdir.name)

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_train_then_load_roundtrip(self):
        _, train_manifest = train_tokenizer(SAMPLE_TEXTS, self.out_dir, vocab_size=1000, min_frequency=1)
        tokenizer, load_manifest = load_frozen_tokenizer(self.out_dir)

        self.assertEqual(train_manifest["tokenizer_hash"], load_manifest["tokenizer_hash"])
        self.assertIsNotNone(tokenizer.token_to_id("<eos>"))
        self.assertIsNotNone(tokenizer.token_to_id("<pad>"))

    def test_byte_level_handles_non_ascii_without_unk(self):
        tokenizer, _ = train_tokenizer(SAMPLE_TEXTS, self.out_dir, vocab_size=1000, min_frequency=1)
        ids = tokenizer.encode("यह एक नया वाक्य है जो प्रशिक्षण में नहीं था").ids
        unk_id = tokenizer.token_to_id("<unk>")
        self.assertNotIn(unk_id, ids)

    def test_load_fails_without_a_trained_tokenizer(self):
        empty_dir = Path(tempfile.mkdtemp())
        with self.assertRaises(FileNotFoundError):
            load_frozen_tokenizer(empty_dir)

    def test_load_detects_tampering(self):
        train_tokenizer(SAMPLE_TEXTS, self.out_dir, vocab_size=1000, min_frequency=1)
        tokenizer_path = self.out_dir / "tokenizer.json"
        with open(tokenizer_path, "a") as f:
            f.write(" ")  # append a byte -- content changes, hash no longer matches

        with self.assertRaises(ValueError):
            load_frozen_tokenizer(self.out_dir)


if __name__ == "__main__":
    unittest.main()

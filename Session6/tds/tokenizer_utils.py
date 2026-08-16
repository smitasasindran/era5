"""Frozen tokenizer: train once (setup), then load with hash verification.

The shard builder never trains a tokenizer -- it only loads one produced by
scripts/build_tokenizer.py and verifies its content hash hasn't drifted
since training. That verification is the actual meaning of "frozen": not
that the file can't technically be edited, but that anything downstream
refuses to run against a tokenizer whose hash doesn't match what was
recorded when it was frozen.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable, Tuple

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

SPECIAL_TOKENS = ["<pad>", "<eos>", "<unk>"]

TOKENIZER_FILENAME = "tokenizer.json"
MANIFEST_FILENAME = "tokenizer_manifest.json"


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def train_tokenizer(
    texts: Iterable[str],
    out_dir: str | Path,
    vocab_size: int = 8000,
    min_frequency: int = 2,
) -> Tuple[Tokenizer, dict]:
    """One-time setup step. Trains a byte-level BPE tokenizer and freezes it.

    Byte-level pre-tokenization means every possible input can always be
    encoded (no true out-of-vocabulary case), which keeps this small and
    self-contained without needing script-specific normalization rules.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=SPECIAL_TOKENS,
        # Force the full 256-byte alphabet into the base vocab regardless of
        # what the training corpus happens to contain -- otherwise a byte
        # sequence absent from training text (e.g. an unseen script) falls
        # back to <unk>, defeating the point of byte-level tokenization.
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tokenizer.train_from_iterator(texts, trainer=trainer)

    tokenizer_path = out_dir / TOKENIZER_FILENAME
    tokenizer.save(str(tokenizer_path))
    tokenizer_hash = sha256_file(tokenizer_path)

    manifest = {
        "tokenizer_hash": tokenizer_hash,
        "vocab_size": tokenizer.get_vocab_size(),
        "special_tokens": {
            "pad": tokenizer.token_to_id("<pad>"),
            "eos": tokenizer.token_to_id("<eos>"),
            "unk": tokenizer.token_to_id("<unk>"),
        },
        "trained_on": "Session 6 corpus (data/corpus/small_shard.parquet)",
    }
    manifest_path = out_dir / MANIFEST_FILENAME
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    return tokenizer, manifest


def load_frozen_tokenizer(out_dir: str | Path) -> Tuple[Tokenizer, dict]:
    """Load a previously frozen tokenizer, verifying its hash hasn't drifted."""
    out_dir = Path(out_dir)
    tokenizer_path = out_dir / TOKENIZER_FILENAME
    manifest_path = out_dir / MANIFEST_FILENAME

    if not tokenizer_path.exists() or not manifest_path.exists():
        raise FileNotFoundError(
            f"No frozen tokenizer at {out_dir}. Run scripts/build_tokenizer.py first."
        )

    with open(manifest_path) as f:
        manifest = json.load(f)

    actual_hash = sha256_file(tokenizer_path)
    if actual_hash != manifest["tokenizer_hash"]:
        raise ValueError(
            "Frozen tokenizer has changed since it was trained: "
            f"expected {manifest['tokenizer_hash']}, found {actual_hash}. "
            "A changed tokenizer is a new artifact -- retrain to get a new "
            "tokenizer_hash rather than editing this one in place."
        )

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer, manifest

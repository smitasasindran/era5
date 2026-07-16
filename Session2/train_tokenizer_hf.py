"""
Train a single shared BPE tokenizer (vocab_size=10000) jointly across all 4
language corpora, using the HuggingFace `tokenizers` library instead of a
hand-rolled BPE implementation.
"""
from tokenizers import Tokenizer, models, pre_tokenizers, trainers

LANGS = ["en", "hi", "te", "mr"]
VOCAB_SIZE = 10000


def corpus_files(weights=None):
    """weights: dict lang -> int repeat count, to bias the trainer's pair
    frequency counts toward languages that need more merges."""
    weights = weights or {l: 1 for l in LANGS}
    files = []
    for l in LANGS:
        files.extend([f"corpus/india_{l}.txt"] * weights.get(l, 1))
    return files


def train_joint_tokenizer(vocab_size=VOCAB_SIZE, min_frequency=1, weights=None):
    tokenizer = Tokenizer(models.BPE(unk_token=None))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        show_progress=True,
        special_tokens=[],
    )
    tokenizer.train(corpus_files(weights), trainer)
    return tokenizer


if __name__ == "__main__":
    import sys
    weights = {"en": 1, "hi": 1, "te": 1, "mr": 1}
    if len(sys.argv) > 1:
        # e.g. "en=5,hi=1,te=1,mr=1"
        for pair in sys.argv[1].split(","):
            k, v = pair.split("=")
            weights[k] = int(v)
    out = sys.argv[2] if len(sys.argv) > 2 else "tokenizer_hf.json"

    tok = train_joint_tokenizer(weights=weights)
    print(f"Final vocab size: {tok.get_vocab_size()}  weights={weights}")
    tok.save(out)
    print(f"Saved {out}")

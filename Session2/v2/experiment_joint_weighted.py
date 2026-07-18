"""
Experiment: single joint tokenizer (all 4 languages pooled into one training
run) with oversampling weights, mirroring the instructor's reported choice
of {"en": 3, "hi": 4, "te": 4, "mai": 2} (Marathi standing in for their
Maithili -- same reasoning as elsewhere in this project: both share
Devanagari with Hindi).

This is a different strategy from our official v2 pipeline (which splits
into 3 script-safe groups -- English, Telugu, Hindi+Marathi-joint -- and
does a symmetric equalization search over vocab quotas). This script exists
purely for comparison, not to replace that pipeline.
"""
import sys
from tokenizers import Tokenizer, models, pre_tokenizers, trainers, decoders
from train_utils import make_normalizer, corpus_path, fertility_for, UNK_TOKEN

DEFAULT_WEIGHTS = {"en": 3, "hi": 4, "te": 4, "mr": 2}


def train_joint(weights, vocab_size=10000):
    files = []
    for lang, w in weights.items():
        files.extend([corpus_path(lang)] * w)

    tok = Tokenizer(models.BPE(unk_token=UNK_TOKEN))
    tok.normalizer = make_normalizer()
    tok.pre_tokenizer = pre_tokenizers.Metaspace()
    tok.decoder = decoders.Metaspace()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size, min_frequency=1, show_progress=False, special_tokens=[UNK_TOKEN]
    )
    tok.train(files, trainer)
    return tok


def evaluate(tok):
    results = {}
    for lang in ["en", "hi", "te", "mr"]:
        ratio, tokens, units = fertility_for(tok, lang)
        results[lang] = ratio
    spread = max(results.values()) - min(results.values())
    return results, spread


if __name__ == "__main__":
    out_path = sys.argv[1] if len(sys.argv) > 1 else "tokenizer_joint_weighted.json"

    tok = train_joint(DEFAULT_WEIGHTS)
    print(f"actual vocab: {tok.get_vocab_size()}  weights: {DEFAULT_WEIGHTS}")

    results, spread = evaluate(tok)
    for lang, r in results.items():
        print(f"  {lang}: fertility={r:.4f}")
    print(f"Spread = {spread:.4f}   Score = 1000/spread ≈ {1000/spread:.1f}" if spread > 0 else "Spread = 0")

    tok.save(out_path)
    print(f"Saved {out_path}")

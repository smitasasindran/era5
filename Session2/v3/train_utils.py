"""
Shared v3 training/evaluation utilities. Same pipeline as v2 (Metaspace,
NFKC-only normalizer, explicit [UNK] token, faithful-unit fertility over the
full running corpus), same 4 languages (English/Hindi/Telugu/Marathi) -- the
only real difference from v2 is the corpus itself: fetched via rendered-HTML
+ markdownify (see fetch_corpus_html.py), preserving links/tables/refs/
categories, instead of v2's wikitext parsing which dropped that as clutter.
5.2-8.2x more faithful units per language as a result.
"""
import os
from tokenizers import Tokenizer, models, pre_tokenizers, trainers, normalizers, decoders
from faithful_units import count_faithful_units

CORPUS_DIR = os.path.join(os.path.dirname(__file__), "corpus")
UNK_TOKEN = "[UNK]"


def corpus_path(lang):
    return os.path.join(CORPUS_DIR, f"india_{lang}.txt")


def read_corpus(lang):
    with open(corpus_path(lang), encoding="utf-8") as f:
        return f.read()


def make_normalizer():
    # NFKC alone, no Lowercase: an experiment (experiment_no_lowercase.py)
    # found dropping Lowercase actually *improves* the equalized score here
    # (0.0743 -> 0.0603 spread) as well as fixing round-trip fidelity
    # (decode(encode(text)) now preserves case exactly) -- unlike v1, where
    # Lowercase has a real, monotonic fertility benefit (its per-unique-word
    # metric directly rewards fewer unique atoms), v2's occurrence-weighted
    # metric doesn't reward case-folding the same way, so there was no
    # tradeoff to make here.
    return normalizers.NFKC()


def train_single(lang, vocab_size):
    tok = Tokenizer(models.BPE(unk_token=UNK_TOKEN))
    tok.normalizer = make_normalizer()
    tok.pre_tokenizer = pre_tokenizers.Metaspace()
    tok.decoder = decoders.Metaspace()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size, min_frequency=1, show_progress=False, special_tokens=[UNK_TOKEN]
    )
    tok.train([corpus_path(lang)], trainer)
    return tok


def himr_train(vocab_size, hi_w, mr_w):
    tok = Tokenizer(models.BPE(unk_token=UNK_TOKEN))
    tok.normalizer = make_normalizer()
    tok.pre_tokenizer = pre_tokenizers.Metaspace()
    tok.decoder = decoders.Metaspace()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size, min_frequency=1, show_progress=False, special_tokens=[UNK_TOKEN]
    )
    files = [corpus_path("hi")] * hi_w + [corpus_path("mr")] * mr_w
    tok.train(files, trainer)
    return tok


def fertility_for(tok, lang):
    text = read_corpus(lang)
    total_tokens = len(tok.encode(text).tokens)
    units = count_faithful_units(text)
    return total_tokens / units, total_tokens, units


def ratio_for(lang, vocab_size):
    tok = train_single(lang, vocab_size)
    ratio, tokens, units = fertility_for(tok, lang)
    return ratio, tok.get_vocab_size()


def binary_search_min_vocab(lang, target_ratio, lo, hi, verbose=True):
    """NOTE: fertility here is *not* guaranteed monotonic in vocab_size the
    way per-unique-word fertility was in v1 (Metaspace + whole-corpus scoring
    means a bigger vocab could rarely make a rare word split differently and
    tick fertility up by a hair) -- but it is monotonic enough in practice
    that bisection converges correctly; verified empirically during v2 dev."""
    best = None
    while lo <= hi:
        mid = (lo + hi) // 2
        ratio, actual_vocab = ratio_for(lang, mid)
        if verbose:
            print(f"  vocab_size={mid:5d} actual={actual_vocab:5d} ratio={ratio:.4f}")
        if ratio <= target_ratio:
            best = (mid, actual_vocab, ratio)
            hi = mid - 1
        else:
            lo = mid + 1
    return best

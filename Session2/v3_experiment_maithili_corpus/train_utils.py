"""
Shared v3 training/evaluation utilities. Same pipeline as v2 (Metaspace,
NFKC-only normalizer, explicit [UNK] token, faithful-unit fertility over the
full running corpus) but on the instructor's richer corpus (rendered-HTML
Markdown, not wikitext-derived) and with Maithili (mai) as the 4th language
instead of Marathi -- Maithili still shares Devanagari with Hindi, so the
same joint-training reasoning applies.

Purpose: an apples-to-apples test of our allocation-search strategy against
the instructor's corpus, to separate "their corpus is bigger/richer" from
"their training strategy is better" -- see report.md.
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


def himai_train(vocab_size, hi_w, mai_w):
    tok = Tokenizer(models.BPE(unk_token=UNK_TOKEN))
    tok.normalizer = make_normalizer()
    tok.pre_tokenizer = pre_tokenizers.Metaspace()
    tok.decoder = decoders.Metaspace()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size, min_frequency=1, show_progress=False, special_tokens=[UNK_TOKEN]
    )
    files = [corpus_path("hi")] * hi_w + [corpus_path("mai")] * mai_w
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

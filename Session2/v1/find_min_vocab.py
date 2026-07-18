"""Binary-search the minimum single-language vocab size needed to hit a target fertility ratio.

Uses a Lowercase normalizer: real, free reduction in English's unique-atom
count (case variants like "India"/"india" collapse into one atom). Devanagari
and Telugu have no case, so it's a no-op there -- safe to apply everywhere.

Uses an explicit unk_token="[UNK]" (matching v2): unrecognized characters
become a real, visible, counted token instead of silently vanishing."""
import sys
from tokenizers import Tokenizer, models, pre_tokenizers, trainers, normalizers

UNK_TOKEN = "[UNK]"


def train_single(lang, vocab_size):
    tok = Tokenizer(models.BPE(unk_token=UNK_TOKEN))
    tok.normalizer = normalizers.Lowercase()
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size, min_frequency=1, show_progress=False, special_tokens=[UNK_TOKEN]
    )
    tok.train([f"corpus/india_{lang}.txt"], trainer)
    return tok


def unique_words(lang, tok):
    with open(f"corpus/india_{lang}.txt", encoding="utf-8") as f:
        text = f.read()
    if tok.normalizer:
        text = tok.normalizer.normalize_str(text)
    pretoks = tok.pre_tokenizer.pre_tokenize_str(text)
    return sorted(set(p for p, _ in pretoks))


def ratio_for(lang, vocab_size):
    tok = train_single(lang, vocab_size)
    words = unique_words(lang, tok)
    total_tokens = sum(len(e.tokens) for e in tok.encode_batch(words))
    return total_tokens / len(words), tok.get_vocab_size()


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


if __name__ == "__main__":
    lang = sys.argv[1] if len(sys.argv) > 1 else "en"
    target = float(sys.argv[2]) if len(sys.argv) > 2 else 1.2
    lo = int(sys.argv[3]) if len(sys.argv) > 3 else 100
    hi = int(sys.argv[4]) if len(sys.argv) > 4 else 9000
    print(f"Searching min vocab_size for {lang} to reach ratio <= {target} in [{lo}, {hi}]")
    result = binary_search_min_vocab(lang, target, lo, hi)
    print(f"\nResult: {result}")

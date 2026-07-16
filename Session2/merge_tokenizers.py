"""
Build the final combined tokenizer from 3 script-safe groups:
  - English (Latin), trained alone
  - Telugu (Telugu script), trained alone
  - Hindi + Marathi (share Devanagari), trained JOINTLY as one coherent BPE
    chain (weight hi=1, mr=2) so there's no cross-language merge-priority
    interference within the pair.

Concatenating merges *across* these 3 groups is safe because Latin, Telugu
script, and Devanagari never overlap -- unlike merging independently-trained
Hindi and Marathi tables, which corrupts each other's merge order (see
optimize_allocation_v2.py docstring).
"""
import json
import sys
from tokenizers import Tokenizer, models, pre_tokenizers, trainers
from find_min_vocab import train_single

HI_W, MR_W = 1, 2


def himr_train(vocab_size, hi_w=HI_W, mr_w=MR_W):
    tok = Tokenizer(models.BPE(unk_token=None))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, min_frequency=1, show_progress=False, special_tokens=[])
    files = ["corpus/india_hi.txt"] * hi_w + ["corpus/india_mr.txt"] * mr_w
    tok.train(files, trainer)
    return tok


def merge_groups(group_tokenizers, out_path):
    combined_vocab = {}
    combined_merges = []
    seen_pairs = set()

    for tok in group_tokenizers:
        d = json.loads(tok.to_str())
        g_vocab = d["model"]["vocab"]
        g_merges = d["model"]["merges"]

        for tok_str in g_vocab.keys():
            if tok_str not in combined_vocab:
                combined_vocab[tok_str] = len(combined_vocab)

        for pair in g_merges:
            a, b = pair
            key = (a, b)
            if key in seen_pairs:
                continue
            seen_pairs.add(key)
            combined_merges.append([a, b])
            merged_tok = a + b
            if merged_tok not in combined_vocab:
                combined_vocab[merged_tok] = len(combined_vocab)

    print(f"Combined vocab size: {len(combined_vocab)}")

    final = Tokenizer(models.BPE(vocab=combined_vocab, merges=[tuple(m) for m in combined_merges], unk_token=None))
    final.pre_tokenizer = pre_tokenizers.Whitespace()
    final.save(out_path)
    print(f"Saved {out_path}")
    return final


if __name__ == "__main__":
    quotas = {"en": 5333, "te": 1813, "himr": 2854}
    out_path = sys.argv[1] if len(sys.argv) > 1 else "tokenizer_final.json"

    en_tok = train_single("en", quotas["en"])
    te_tok = train_single("te", quotas["te"])
    himr_tok = himr_train(quotas["himr"])

    merge_groups([en_tok, te_tok, himr_tok], out_path)

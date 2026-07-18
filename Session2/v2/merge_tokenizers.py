"""
Builds the final v2 tokenizer from 3 script-safe groups (English/Latin,
Telugu/Telugu-script, Hindi+Marathi/Devanagari-joint), same concatenation
strategy as v1's merge_tokenizers.py (safe because the 3 groups' scripts
never overlap -- see that file's docstring for why concatenating
independently-trained *same-script* tokenizers would corrupt merge order).

Only real difference from v1: unk_token="[UNK]" instead of None, and the
Metaspace pre-tokenizer + NFKC normalizer (no Lowercase -- see
train_utils.make_normalizer()) are attached to the final tokenizer too --
plus a `decoders.Metaspace()` decoder, which v1 has no equivalent of.
Metaspace's replacement character ("▁") is specifically
what makes decode() invertible: it marks where a real space was, so decode()
can convert it back, reconstructing spacing/punctuation-adjacency correctly.
Whitespace (v1) has no such marker -- once it splits "India," into "India"
+ "," it's structurally impossible to recover that they were adjacent
(vs. separated by a space) from the token sequence alone.
"""
import json
import sys
from tokenizers import Tokenizer, models, pre_tokenizers, decoders
from train_utils import train_single, himr_train, make_normalizer, UNK_TOKEN
from optimize_allocation import compute_quotas


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

    final = Tokenizer(
        models.BPE(vocab=combined_vocab, merges=[tuple(m) for m in combined_merges], unk_token=UNK_TOKEN)
    )
    final.normalizer = make_normalizer()
    final.pre_tokenizer = pre_tokenizers.Metaspace()
    final.decoder = decoders.Metaspace()
    final.save(out_path)
    print(f"Saved {out_path}")
    return final


if __name__ == "__main__":
    out_path = sys.argv[1] if len(sys.argv) > 1 else "tokenizer_final.json"

    quotas = compute_quotas()

    en_tok = train_single("en", quotas["en"])
    te_tok = train_single("te", quotas["te"])
    himr_tok = himr_train(quotas["himr"], quotas["hi_w"], quotas["mr_w"])

    merge_groups([en_tok, te_tok, himr_tok], out_path)

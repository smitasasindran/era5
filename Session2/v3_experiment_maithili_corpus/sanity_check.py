"""
Quick sanity checks for tokenizer_final.json -- rerun after any retraining:
  - vocab size
  - unk_token behavior (an unrecognized character should become a real,
    visible [UNK] token, not silently vanish)
  - decode(encode(text)) round-trip -- v2's Metaspace decoder should
    reconstruct spacing/punctuation-adjacency exactly (unlike v1)

Usage: python sanity_check.py [tokenizer_final.json]
"""
import json
import sys
from tokenizers import Tokenizer

ROUND_TRIP_TEXTS = [
    "India, officially the Republic of India",
    "Hello World! 2024",
    "India's population",
]

UNK_TEST_TEXTS = [
    "Hello World! 2024",
    "hello 中文 world",
    "emoji 😀 test",
]


def main(path):
    tok = Tokenizer.from_file(path)
    print(f"Tokenizer: {path}")
    print(f"vocab_size: {tok.get_vocab_size()}")
    print(f"model.unk_token: {json.loads(tok.to_str())['model'].get('unk_token')}")
    print(f"decoder configured: {tok.decoder is not None}")

    print("\n--- unk_token behavior ---")
    for text in UNK_TEST_TEXTS:
        enc = tok.encode(text)
        print(f"{text!r}")
        print(f"  tokens: {enc.tokens}")
        print(f"  ids:    {enc.ids}")

    print("\n--- decode(encode(text)) round-trip ---")
    for text in ROUND_TRIP_TEXTS:
        decoded = tok.decode(tok.encode(text).ids)
        match = "MATCH" if decoded == text else "differs"
        print(f"  original: {text!r}")
        print(f"  decoded:  {decoded!r}  [{match}]")
        print()


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "tokenizer_final.json"
    main(path)

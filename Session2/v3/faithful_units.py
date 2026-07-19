"""
faithful_unit = one contiguous Unicode letter/mark/number run, OR one visible
non-space punctuation/symbol character (each counted individually, not
grouped into a run).

This is deliberately decoupled from whatever pre-tokenizer the trained
tokenizer itself uses (Metaspace here) -- it's a fixed, tokenizer-agnostic
linguistic unit count, so fertility measures real compression rather than
being an artifact of the tokenizer's own word-boundary choices. Counted on
the RAW corpus text (pre-normalization) since it's meant to reflect the true
visible content, not whatever the tokenizer normalizes it to.

fertility(language) = token_count(language) / faithful_unit_count(language)
(token_count / faithful_unit_count computed over the FULL running corpus
text, not per unique word -- this is why fertility can drop below 1.0: a
single learned merge can span multiple faithful units, e.g. a word fused
with its trailing punctuation or with an adjacent short function word.)
"""
import unicodedata


def _is_lmn(ch):
    return unicodedata.category(ch)[0] in ("L", "M", "N")


def count_faithful_units(text):
    units = 0
    in_run = False
    for ch in text:
        if ch.isspace():
            in_run = False
            continue
        if _is_lmn(ch):
            if not in_run:
                units += 1
                in_run = True
        else:
            units += 1
            in_run = False
    return units


if __name__ == "__main__":
    import sys

    for path in sys.argv[1:]:
        with open(path, encoding="utf-8") as f:
            text = f.read()
        print(f"{path}: {count_faithful_units(text)} faithful units ({len(text)} chars)")

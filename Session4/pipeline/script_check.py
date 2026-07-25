"""Script-purity diagnostic.

For a document, what fraction of its script-attributable characters fall in
a single dominant Unicode script block? This is a text-only signal -- no
language label needed, so it works on any dataset, not just ones that happen
to carry a `language` column.

Useful for two things:
  - sanity-checking the Brahmic-joiner-preservation work in clean_text(): if
    a corpus is supposed to be mostly Devanagari/Tamil/etc., the dominant
    script distribution here should reflect that.
  - surfacing documents worth a second look: low purity can mean legitimate
    code-switching, but it can also mean a legacy non-Unicode font (e.g.
    Krutidev for Hindi, TSCII for Tamil) that rendered correctly in a
    browser but produced garbage codepoints once scraped as plain text --
    that shows up here as a document with an unexpectedly Latin-heavy
    histogram. This module only surfaces the signal; deciding what counts as
    "unexpected" needs a declared language to compare against, which is an
    analysis-time choice, not something this pipeline assumes exists.

This is diagnostic only -- it does not change what clean_text() does.
"""

import re

# (script_name, low, high) -- the scripts actually present in this course's
# dataset, plus Latin. Extend this list if you bring in others (Telugu,
# Sinhala, Perso-Arabic, ...).
SCRIPT_BLOCKS = [
    ("devanagari", 0x0900, 0x097F),  # hi, mr, sa, ne
    ("bengali", 0x0980, 0x09FF),  # bn, as
    ("gurmukhi", 0x0A00, 0x0A7F),  # pa
    ("gujarati", 0x0A80, 0x0AFF),  # gu
    ("oriya", 0x0B00, 0x0B7F),  # or
    ("tamil", 0x0B80, 0x0BFF),  # ta
    ("telugu", 0x0C00, 0x0C7F),
    ("kannada", 0x0C80, 0x0CFF),  # kn
    ("malayalam", 0x0D00, 0x0D7F),  # ml
    ("latin", 0x0041, 0x024F),  # a-z, A-Z, Latin-1 Supplement, Latin Extended
]

_SCRIPT_RE = {
    name: re.compile("[\\u{:04x}-\\u{:04x}]".format(lo, hi))
    for name, lo, hi in SCRIPT_BLOCKS
}

# Characters that aren't evidence of any particular script and shouldn't
# count toward the histogram at all: whitespace, digits, common punctuation.
_NEUTRAL_RE = re.compile(r"[\s0-9.,!?;:'\"()\[\]{}\-/\\@#$%^&*_+=<>~`|]")


def script_histogram(text):
    """Character counts per known script block. Empty dict if none found."""
    if not text:
        return {}
    counts = {name: len(pat.findall(text)) for name, pat in _SCRIPT_RE.items()}
    return {k: v for k, v in counts.items() if v > 0}


def script_purity(text):
    """Returns {"dominant_script": str, "purity_ratio": float}.

    dominant_script is "" when the text has no script-attributable
    characters at all (pure numbers/punctuation/whitespace, or empty).
    purity_ratio = dominant script's char count / all script-attributable
    chars found -- 1.0 means script-pure, lower means a meaningful chunk is
    in some other script.
    """
    hist = script_histogram(text)
    total = sum(hist.values())
    if total == 0:
        return {"dominant_script": "", "purity_ratio": 0.0}
    dominant = max(hist, key=hist.get)
    return {"dominant_script": dominant, "purity_ratio": hist[dominant] / total}

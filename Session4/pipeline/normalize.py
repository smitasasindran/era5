"""Step 1 of the data-cleaning pipeline: Normalize + Clean.

clean_text() does, in order:
  1. repair mojibake (ftfy, if installed) -- fixes double-encoded UTF-8 text
     like "\xc3\xa9" that should have been "\xe9"; NFC alone does NOT catch
     this, it only picks one canonical encoding for whatever characters are
     already there, mojibake characters are already "valid" unicode, just
     the wrong ones.
  2. NFC-normalize unicode (one canonical encoding per character)
  3. unescape HTML entities (&amp; -> "&", numeric entities -> the real char)
  4. drop the literal U+FFFD replacement character
  5. strip invisible noise: C0/C1 controls, ZWSP, BOM, bidi overrides
     -- but NEVER touch ZWNJ (U+200C) / ZWJ (U+200D): these are real
     characters in Brahmic scripts (they control conjunct formation) and
     stripping them mangles Indic text.
  6. flag "ghost" special tokens (<|endoftext|>, [USER], |<user>|, ...)
     instead of silently deleting them -- the set of real special tokens
     must be a deliberate decision made before the tokenizer is built, not
     an accident of whatever pattern happened to appear in the corpus.
  7. prose only: de-hyphenate line-wrapped words ("exam-\\nple" -> "example"),
     the artifact of a fixed-width layout (PDF/OCR sources) rather than a
     real compound word -- must run before whitespace collapsing, while the
     real newline is still there to detect.
  8. tidy whitespace -- collapsed to single spaces for prose, but only
     trimmed line-by-line for code, since indentation is meaningful there.

The hash used for exact-dedup should be computed on the OUTPUT of
clean_text(), not on the raw string -- see content_hash().

This file is kept pure-ASCII on purpose: the noise code points below are
built from integer code points at import time rather than typed as literal
invisible characters, so nothing here is silently unreadable in an editor,
a diff, or a terminal.
"""

import hashlib
import html
import re
import unicodedata

try:
    import ftfy
except ImportError:
    ftfy = None

# ---------------------------------------------------------------------------
# Noise: characters stripped unconditionally, regardless of script/content type.
# ZWNJ (U+200C) and ZWJ (U+200D) are intentionally absent from this list --
# see module docstring.
#
# Each range carries a category tag ("control" or "zero_width") so the report
# stats below (garbage_control_chars, zero_width_noise_chars) can be derived
# from this single source of truth instead of a second hand-maintained list
# that could quietly drift out of sync with what NOISE_RE actually strips.
# ---------------------------------------------------------------------------
_NOISE_RANGES = [
    (0x00, 0x08, "control"),      # C0 controls (keep \t=0x09 \n=0x0A \r=0x0D)
    (0x0B, 0x0C, "control"),
    (0x0E, 0x1F, "control"),
    (0x7F, 0x7F, "control"),
    (0x80, 0x9F, "control"),      # C1 controls
    (0x200B, 0x200B, "zero_width"),  # ZWSP - zero width space
    (0x200E, 0x200F, "zero_width"),  # LRM / RLM marks
    (0x202A, 0x202E, "zero_width"),  # bidi embedding/override
    (0x2060, 0x2064, "zero_width"),  # word joiner, invisible math operators
    (0xFEFF, 0xFEFF, "zero_width"),  # BOM / zero width no-break space
]


def _build_class_pattern(ranges):
    return "[" + "".join(
        "\\u{:04x}".format(lo) if lo == hi else "\\u{:04x}-\\u{:04x}".format(lo, hi)
        for lo, hi, _ in ranges
    ) + "]"


NOISE_RE = re.compile(_build_class_pattern(_NOISE_RANGES))
_CONTROL_CHAR_RE = re.compile(_build_class_pattern([r for r in _NOISE_RANGES if r[2] == "control"]))
_ZERO_WIDTH_NOISE_RE = re.compile(_build_class_pattern([r for r in _NOISE_RANGES if r[2] == "zero_width"]))

REPLACEMENT_CHAR = chr(0xFFFD)
ZWNJ = chr(0x200C)
ZWJ = chr(0x200D)

# ---------------------------------------------------------------------------
# Ghost special tokens: patterns that LOOK like control/special tokens but
# were never registered with a tokenizer. We flag, we do not delete -- the
# course notes are explicit that these decide the tokenizer's special-token
# vocabulary, so silently dropping them would hide that decision.
#
# Matching is restricted to a curated vocabulary of conversation/control-role
# words (user, system, assistant, endoftext, ...) rather than "any bracketed
# word" -- an unrestricted pattern also matches ordinary HTML tags (<h3>,
# <video>) and acronyms ([ERISA], [NASA]) in web-scraped text, which drowns
# out the real signal. Run the Extraction stage first to remove HTML
# altogether; this vocabulary restriction is the second line of defense for
# whatever leaks through anyway.
# ---------------------------------------------------------------------------
ROLE_TOKENS = [
    "user", "system", "assistant", "human", "bot", "ai",
    "instruction", "inst", "input", "output", "response",
    "prompt", "context", "question", "answer",
    "endoftext", "startoftext", "pad", "unk", "bos", "eos",
    "sep", "cls", "mask", "im_start", "im_end", "sys", "s",
]
# "s" alone is only meaningful as the sentencepiece-style <s>/</s> marker --
# in [brackets] or |<pipes>| a bare "s" is just as likely to be ordinary text
# ("occupant[s]", "http[s]", "np.stack([s])"), so it's excluded from every
# pattern except angle_bare.
_ANGLE_ROLE_TOKENS = ROLE_TOKENS
_BRACKET_ROLE_TOKENS = [t for t in ROLE_TOKENS if t != "s"]

_angle_alt = "|".join(re.escape(w) for w in sorted(_ANGLE_ROLE_TOKENS, key=len, reverse=True))
_bracket_alt = "|".join(re.escape(w) for w in sorted(_BRACKET_ROLE_TOKENS, key=len, reverse=True))

GHOST_TOKEN_PATTERNS = {
    "pipe_angle": re.compile(rf"<\|/?(?:{_bracket_alt})\|>", re.IGNORECASE),          # <|endoftext|>, <|im_start|>
    "bracket_upper": re.compile(rf"\[/?(?:{_bracket_alt})\]", re.IGNORECASE),         # [USER], [INST], [/INST]
    "pipe_wrapped_angle": re.compile(rf"\|<\/?(?:{_bracket_alt})>\|", re.IGNORECASE),  # |<user>|
    "angle_bare": re.compile(rf"<\/?(?:{_angle_alt})>", re.IGNORECASE),              # <user>, <sep>, <s>, </s>
    "double_angle_sys": re.compile(r"<<\s*/?SYS\s*>>", re.IGNORECASE),                # <<SYS>>, <</SYS>>
}

# Crude "does this look like code" heuristic for --content-type auto.
# Not a real classifier -- that's the job of the later Quality Filtering /
# Language ID steps. This is just enough to pick a whitespace strategy.
CODE_HINT_RE = re.compile(
    r"(^[ \t]{2,}\S)|(\bdef \w+\(.*\):)|(\bimport \w+)|(;\s*$)|(=>|::)|(\{\s*$)",
    re.MULTILINE,
)


def guess_content_type(s):
    if not s:
        return "prose"
    hits = len(CODE_HINT_RE.findall(s))
    return "code" if hits >= 3 else "prose"


def detect_ghost_tokens(s):
    """Return {pattern_name: [all matched substrings]} for patterns that hit."""
    found = {}
    for name, pat in GHOST_TOKEN_PATTERNS.items():
        matches = pat.findall(s)
        if matches:
            found[name] = matches
    return found


def _collapse_whitespace_prose(s):
    return re.sub(r"\s+", " ", s).strip()


# A hyphen at a line break, surrounded by lowercase letters, is the classic
# fixed-width-layout wrap artifact (PDF/OCR text) -- rejoin it. Anything
# else (hyphen before/after a digit, uppercase letter, or punctuation) is
# left alone since it's more likely a real compound/range/bullet.
_HYPHEN_WRAP_RE = re.compile(r"(?<=[a-z])-\r?\n(?=[a-z])")


def _dehyphenate_wrapped_lines(s):
    return _HYPHEN_WRAP_RE.sub("", s)


def _repair_mojibake(s):
    if ftfy is None:
        return s
    return ftfy.fix_text(s)


def _tidy_whitespace_code(s):
    """Preserve structure (newlines, indentation) -- only trim trailing
    per-line whitespace and collapse runs of >1 blank line down to 1."""
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in s.split("\n")]
    out = []
    blank_run = 0
    for line in lines:
        if line == "":
            blank_run += 1
            if blank_run > 1:
                continue
        else:
            blank_run = 0
        out.append(line)
    return "\n".join(out).strip("\n")


def clean_text(s, content_type="prose"):
    """Returns (cleaned_text, ghost_tokens_dict).

    content_type: "prose" collapses all whitespace to single spaces;
    "code" preserves indentation/newline structure instead.
    """
    if not s:
        return "", {}

    s = _repair_mojibake(s)
    s = unicodedata.normalize("NFC", s)
    s = html.unescape(s)
    s = s.replace(REPLACEMENT_CHAR, "")
    s = NOISE_RE.sub("", s)  # ZWNJ (U+200C) / ZWJ (U+200D) survive this untouched

    ghosts = detect_ghost_tokens(s)

    if content_type == "code":
        s = _tidy_whitespace_code(s)
    else:
        s = _dehyphenate_wrapped_lines(s)
        s = _collapse_whitespace_prose(s)

    return s, ghosts


def content_hash(clean_s):
    """SHA1 over the CLEANED text. Compute after clean_text(), not before --
    two documents that only differ by encoding noise should hash identically."""
    return hashlib.sha1(clean_s.encode("utf-8")).hexdigest()


def analyze_noise(raw, cleaned):
    """Diagnostic counts for the report, broken out to match the course
    notes' own categories. These are read-only observations about what
    clean_text() did/found -- they don't drive cleaning themselves, NOISE_RE
    is still the single source of truth for what gets stripped.

    - garbage_control_chars: C0/C1 control chars removed (the notes' own
      phrase: invisible chars that "carry no meaning and only produce
      garbage tokens").
    - zero_width_noise_chars: ZWSP/BOM/bidi-override/word-joiner chars
      removed -- noise, as distinct from the Brahmic joiners below.
    - broken_utf_replacement_chars: count of literal U+FFFD in the raw text,
      a signal the source had already-corrupted/mis-decoded UTF-8 before it
      ever reached this pipeline.
    - zwnj_kept / zwj_kept: count of ZWNJ/ZWJ present in the CLEANED text --
      confirms the Brahmic-joiner preservation guarantee actually held for
      this document, rather than just asserting it in a comment.
    """
    if not raw:
        return {
            "garbage_control_chars": 0,
            "zero_width_noise_chars": 0,
            "broken_utf_replacement_chars": 0,
            "zwnj_kept": 0,
            "zwj_kept": 0,
        }
    return {
        "garbage_control_chars": len(_CONTROL_CHAR_RE.findall(raw)),
        "zero_width_noise_chars": len(_ZERO_WIDTH_NOISE_RE.findall(raw)),
        "broken_utf_replacement_chars": raw.count(REPLACEMENT_CHAR),
        "zwnj_kept": cleaned.count(ZWNJ),
        "zwj_kept": cleaned.count(ZWJ),
    }

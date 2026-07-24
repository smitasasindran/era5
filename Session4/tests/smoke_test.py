#!/usr/bin/env python3
"""Smoke test for pipeline.normalize.clean_text().

Not a pytest suite -- just run it directly:
    python tests/smoke_test.py

Every noise/joiner character is built with chr(0x....) rather than typed
literally, so this file stays pure-ASCII and the intent of every check is
visible in the diff/terminal instead of hiding in invisible bytes.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.normalize import analyze_noise, clean_text, content_hash, detect_ghost_tokens  # noqa: E402

ZWSP = chr(0x200B)
BOM = chr(0xFEFF)
LRM = chr(0x200E)
BIDI_OVERRIDE = chr(0x202E)
REPLACEMENT = chr(0xFFFD)
ZWNJ = chr(0x200C)
ZWJ = chr(0x200D)
C0_VERTICAL_TAB = chr(0x0B)

checks_run = 0
checks_failed = 0


def check(label, condition):
    global checks_run, checks_failed
    checks_run += 1
    status = "PASS" if condition else "FAIL"
    if not condition:
        checks_failed += 1
    print(f"[{status}] {label}")


def show(label, raw, cleaned):
    print(f"\n--- {label} ---")
    print("raw    :", repr(raw))
    print("cleaned:", repr(cleaned))


# 1. English prose: HTML entities, BOM, ZWSP, bidi override, replacement char, extra whitespace
raw_en = BOM + "Hello &amp; welcome" + ZWSP + LRM + "   to   the  " + BIDI_OVERRIDE + "course" + REPLACEMENT + ".\n\n\nBye."
cleaned_en, ghosts_en = clean_text(raw_en, content_type="prose")
show("English prose", raw_en, cleaned_en)
check("BOM stripped", BOM not in cleaned_en)
check("ZWSP stripped", ZWSP not in cleaned_en)
check("bidi override stripped", BIDI_OVERRIDE not in cleaned_en)
check("U+FFFD replacement char stripped", REPLACEMENT not in cleaned_en)
check("HTML entity unescaped", "&amp;" not in cleaned_en and "&" in cleaned_en)
check("whitespace collapsed to single spaces", "  " not in cleaned_en)
check("newlines collapsed in prose mode", "\n" not in cleaned_en)

# 2. Devanagari (Hindi) text with a real ZWNJ conjunct plus ZWSP noise mixed in.
# ZWNJ here forces a half-form instead of the default conjunct ligature -- a real
# linguistic choice, not noise -- and must survive cleaning untouched.
devanagari_word = chr(0x0915) + ZWNJ + chr(0x0937)  # KA + ZWNJ + SSA
raw_hi = devanagari_word + ZWSP + " " + chr(0x0928) + chr(0x092E) + chr(0x0938) + chr(0x094D) + chr(0x0924) + chr(0x0947)
cleaned_hi, _ = clean_text(raw_hi, content_type="prose")
show("Devanagari with ZWNJ", raw_hi, cleaned_hi)
check("ZWNJ preserved in Brahmic text", ZWNJ in cleaned_hi)
check("ZWSP still stripped even mixed with Indic text", ZWSP not in cleaned_hi)
check("ZWJ would also survive if present", clean_text(ZWJ, content_type="prose")[0] == ZWJ)

# 3. Code: indentation and blank-line structure must NOT be collapsed to single spaces.
raw_code = "def add(a, b):" + BOM + "\n    return a + b\n\n\n\nprint(add(1, 2))   \n"
cleaned_code, _ = clean_text(raw_code, content_type="code")
show("Python code", raw_code, cleaned_code)
check("code keeps newlines", "\n" in cleaned_code)
check("code keeps leading indentation", "\n    return" in cleaned_code)
check("code collapses runs of blank lines to one", "\n\n\n" not in cleaned_code)
check("code still strips BOM noise", BOM not in cleaned_code)
check("code strips trailing per-line whitespace", not any(line.endswith(" ") for line in cleaned_code.split("\n")))

# 4. Ghost special tokens: three spellings of the same intent should all be flagged.
raw_ghost = "Hi <|endoftext|> and [USER] said |<user>| plus a <sep> tag."
_, ghosts = clean_text(raw_ghost, content_type="prose")
show("Ghost tokens", raw_ghost, ghosts)
check("<|endoftext|> flagged as pipe_angle", "<|endoftext|>" in ghosts.get("pipe_angle", []))
check("[USER] flagged as bracket_upper", "[USER]" in ghosts.get("bracket_upper", []))
check("|<user>| flagged as pipe_wrapped_angle", "|<user>|" in ghosts.get("pipe_wrapped_angle", []))
check("<sep> flagged as angle_bare", "<sep>" in ghosts.get("angle_bare", []))

# 4b. The ghost-token vocab is restricted to conversation/control-role words on
# purpose -- ordinary HTML tags and acronyms must NOT be flagged (this is what
# was polluting ghost_token_examples with <h3>/<video>/[ERISA] before the fix).
raw_html_noise = "See the <h3>heading</h3> and <video>clip</video> under the [ERISA] plan, [NASA] said."
_, ghosts_noise = clean_text(raw_html_noise, content_type="prose")
show("HTML tags / acronyms (must NOT be flagged)", raw_html_noise, ghosts_noise)
check("no ghost tokens flagged in HTML-tag/acronym noise", ghosts_noise == {})

# 4c. Common chat-template markers beyond the three original spellings.
raw_chat = "<<SYS>>be helpful<</SYS>> [INST]hi[/INST] <|im_start|>user says </s>"
_, ghosts_chat = clean_text(raw_chat, content_type="prose")
show("Chat-template markers", raw_chat, ghosts_chat)
check("<<SYS>> flagged as double_angle_sys", "<<SYS>>" in ghosts_chat.get("double_angle_sys", []))
check("[INST] flagged as bracket_upper", "[INST]" in ghosts_chat.get("bracket_upper", []))
check("<|im_start|> flagged as pipe_angle", "<|im_start|>" in ghosts_chat.get("pipe_angle", []))
check("</s> flagged as angle_bare", "</s>" in ghosts_chat.get("angle_bare", []))

# 4d. Bare "s" in [brackets] is ordinary text (plural/regex/code notation), not
# a special token -- only <s>/</s> (angle_bare) should ever match it.
raw_bare_s = "occupant[s] and http[s] and np.stack([s])"
_, ghosts_bare_s = clean_text(raw_bare_s, content_type="prose")
show("Bare 's' in brackets (must NOT be flagged)", raw_bare_s, ghosts_bare_s)
check("no ghost tokens flagged for bracketed bare 's'", ghosts_bare_s == {})

# 4e. analyze_noise() report stats: garbage control chars, zero-width noise,
# broken-utf, and the ZWNJ/ZWJ "kept" counters, checked against a doc that
# exercises every category at once.
raw_noise_doc = (
    BOM + "Hello" + ZWSP + " world" + C0_VERTICAL_TAB
    + REPLACEMENT + REPLACEMENT
    + " " + devanagari_word + ZWJ
)
cleaned_noise_doc, _ = clean_text(raw_noise_doc, content_type="prose")
stats = analyze_noise(raw_noise_doc, cleaned_noise_doc)
show("analyze_noise() input", raw_noise_doc, cleaned_noise_doc)
print("stats:", stats)
check("garbage_control_chars counts the vertical tab", stats["garbage_control_chars"] == 1)
check("zero_width_noise_chars counts BOM + ZWSP", stats["zero_width_noise_chars"] == 2)
check("broken_utf_replacement_chars counts both U+FFFD", stats["broken_utf_replacement_chars"] == 2)
check("zwnj_kept counts the surviving ZWNJ", stats["zwnj_kept"] == 1)
check("zwj_kept counts the surviving ZWJ", stats["zwj_kept"] == 1)
check("analyze_noise on empty input is all zeros", analyze_noise("", "") == {
    "garbage_control_chars": 0,
    "zero_width_noise_chars": 0,
    "broken_utf_replacement_chars": 0,
    "zwnj_kept": 0,
    "zwj_kept": 0,
})

# 5. C0 control char (vertical tab) is noise and must go; \t \n \r must survive.
raw_ctrl = "line one" + C0_VERTICAL_TAB + "\tstill\nhere\r\n"
cleaned_ctrl, _ = clean_text(raw_ctrl, content_type="code")
show("Control chars (code mode)", raw_ctrl, cleaned_ctrl)
check("vertical tab (C0 control) stripped", C0_VERTICAL_TAB not in cleaned_ctrl)

# 6. Hash is computed on cleaned text -- two docs differing only by encoding noise
# should collapse to the same hash (this is what makes exact-dedup work downstream).
noisy_a = "Hello" + ZWSP + " World" + BOM
noisy_b = BOM + "Hello World"
cleaned_a, _ = clean_text(noisy_a, content_type="prose")
cleaned_b, _ = clean_text(noisy_b, content_type="prose")
check(
    "hash matches for two docs differing only by encoding noise",
    content_hash(cleaned_a) == content_hash(cleaned_b),
)

# 7. Empty / falsy input should not blow up.
check("empty string handled", clean_text("", content_type="prose") == ("", {}))
check("None-ish handled", clean_text(None, content_type="prose") == ("", {}))

# 8. Mojibake repair (ftfy): double-encoded UTF-8 text should come back as the
# real characters. "cafe" + U+00E9 (e-acute) mis-decoded as latin-1 then
# re-encoded as UTF-8 is the textbook mojibake case -- NFC alone does NOT fix
# this, since every byte involved is already a "valid" (wrong) character.
mojibake_raw = "caf" + chr(0x00E9).encode("utf-8").decode("latin-1")
cleaned_mojibake, _ = clean_text(mojibake_raw, content_type="prose")
show("Mojibake repair", mojibake_raw, cleaned_mojibake)
try:
    import ftfy as _ftfy_probe  # noqa: F401
    check("mojibake repaired to the real character", cleaned_mojibake == "caf" + chr(0x00E9))
except ImportError:
    check("ftfy not installed -- mojibake left as-is (expected without the dependency)", True)

# 9. De-hyphenation: a word wrapped across a line break by fixed-width layout
# (PDF/OCR) should be rejoined; a genuine end-of-line hyphenated/compound
# word before a capitalized new sentence is NOT touched by this heuristic
# (only lowercase-hyphen-newline-lowercase triggers it).
raw_wrapped = "This is an exam-\nple of wrapped text, not a re-\nsult of anything else."
cleaned_wrapped, _ = clean_text(raw_wrapped, content_type="prose")
show("De-hyphenation (prose)", raw_wrapped, cleaned_wrapped)
check("wrapped word rejoined: example", "example" in cleaned_wrapped)
check("wrapped word rejoined: result", "result" in cleaned_wrapped)
check("no stray hyphen left behind", "exam-" not in cleaned_wrapped and "re-" not in cleaned_wrapped)

raw_code_hyphen = "x = 1\ny = flag-\nvalue"
cleaned_code_hyphen, _ = clean_text(raw_code_hyphen, content_type="code")
show("De-hyphenation must NOT run for code", raw_code_hyphen, cleaned_code_hyphen)
check("code mode leaves a line-end hyphen untouched", "flag-\nvalue" in cleaned_code_hyphen)

print(f"\n{checks_run - checks_failed}/{checks_run} checks passed.")
if checks_failed:
    sys.exit(1)

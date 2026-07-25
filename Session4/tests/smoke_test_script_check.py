#!/usr/bin/env python3
"""Smoke test for pipeline.script_check.script_purity().

Run directly:
    python tests/smoke_test_script_check.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.script_check import script_purity  # noqa: E402

checks_run = 0
checks_failed = 0


def check(label, condition):
    global checks_run, checks_failed
    checks_run += 1
    status = "PASS" if condition else "FAIL"
    if not condition:
        checks_failed += 1
    print(f"[{status}] {label}")


def show(label, text, result):
    print(f"\n--- {label} ---")
    print("text  :", repr(text))
    print("result:", result)


# 1. Pure Devanagari (Hindi) text should read as fully devanagari-pure.
hindi_text = chr(0x0928) + chr(0x092E) + chr(0x0938) + chr(0x094D) + chr(0x0924) + chr(0x0947)  # namaste
result_hi = script_purity(hindi_text)
show("Pure Devanagari", hindi_text, result_hi)
check("dominant script is devanagari", result_hi["dominant_script"] == "devanagari")
check("purity is 1.0 for script-pure text", result_hi["purity_ratio"] == 1.0)

# 2. Pure Tamil text.
tamil_text = "".join(chr(c) for c in [0x0BB5, 0x0BA3, 0x0B95, 0x0BCD, 0x0B95, 0x0BAE, 0x0BCD])
result_ta = script_purity(tamil_text)
show("Pure Tamil", tamil_text, result_ta)
check("dominant script is tamil", result_ta["dominant_script"] == "tamil")
check("purity is 1.0 for script-pure text", result_ta["purity_ratio"] == 1.0)

# 3. English prose (Latin script), including ordinary punctuation/digits which
# must NOT count as evidence of any script (they're script-neutral).
english_text = "Hello, world! This costs $42.50 (approx)."
result_en = script_purity(english_text)
show("English prose with punctuation/digits", english_text, result_en)
check("dominant script is latin", result_en["dominant_script"] == "latin")
check("purity is 1.0 -- punctuation/digits don't dilute it", result_en["purity_ratio"] == 1.0)

# 4. Code-switched text: mostly Devanagari with a short English aside should
# have devanagari as dominant but purity meaningfully below 1.0.
mixed_text = hindi_text * 5 + " (see also: reference guide)"
result_mixed = script_purity(mixed_text)
show("Code-switched Devanagari + English", mixed_text, result_mixed)
check("dominant script is still devanagari", result_mixed["dominant_script"] == "devanagari")
check("purity is below 1.0 due to the English aside", 0.0 < result_mixed["purity_ratio"] < 1.0)

# 5. Pure numbers/punctuation/whitespace: no script signal at all.
no_script_text = "12345 - 67.89 % ()"
result_none = script_purity(no_script_text)
show("No script-attributable characters", no_script_text, result_none)
check("dominant_script is empty string", result_none["dominant_script"] == "")
check("purity_ratio is 0.0", result_none["purity_ratio"] == 0.0)

# 6. Empty input handled without raising.
check("empty string handled", script_purity("") == {"dominant_script": "", "purity_ratio": 0.0})

print(f"\n{checks_run - checks_failed}/{checks_run} checks passed.")
if checks_failed:
    sys.exit(1)

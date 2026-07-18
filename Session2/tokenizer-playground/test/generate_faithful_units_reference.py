"""
Generates reference faithful-unit counts (using the real Python
faithful_units.py from v2/) for test/test_strings.json, so
compare_faithful_units.js can verify the JS port in tokenizer.js matches.

Usage: python test/generate_faithful_units_reference.py
"""
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "v2"))
from faithful_units import count_faithful_units

TEST_STRINGS_PATH = "test/test_strings.json"
OUT_PATH = "test/reference_faithful_units.json"

if __name__ == "__main__":
    with open(TEST_STRINGS_PATH, encoding="utf-8") as f:
        test_strings = json.load(f)

    results = [{"text": s, "units": count_faithful_units(s)} for s in test_strings]

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Wrote {len(results)} cases to {OUT_PATH}")

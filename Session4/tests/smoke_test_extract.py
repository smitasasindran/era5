#!/usr/bin/env python3
"""Smoke test for pipeline.extract.extract_content().

Run directly:
    python tests/smoke_test_extract.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.extract import (  # noqa: E402
    _extract_with_bs4,
    _strip_tags_lightweight,
    extract_content,
    looks_like_html_document,
)

checks_run = 0
checks_failed = 0


def check(label, condition):
    global checks_run, checks_failed
    checks_run += 1
    status = "PASS" if condition else "FAIL"
    if not condition:
        checks_failed += 1
    print(f"[{status}] {label}")


def show(label, raw, extracted, meta):
    print(f"\n--- {label} ---")
    print("raw      :", repr(raw))
    print("extracted:", repr(extracted))
    print("meta     :", meta)


# 1. A full HTML page with nav/footer/cookie-banner boilerplate around a real article.
raw_page = """
<!doctype html>
<html>
<head><title>Example</title></head>
<body>
<nav>Home | About | Contact</nav>
<div class="cookie-banner">We use cookies to improve your experience. Accept all cookies.</div>
<article>
<h1>The Real Headline</h1>
<p>This is the actual article body that we want to keep for training.</p>
<p>It has more than one paragraph of real content in it.</p>
</article>
<footer>Copyright 2024 Example Corp. All rights reserved.</footer>
</body>
</html>
"""
extracted_page, meta_page = extract_content(raw_page)
show("Full HTML page", raw_page, extracted_page, meta_page)
check("detected as HTML document", looks_like_html_document(raw_page))
check("used trafilatura or bs4 tier", meta_page["path"] in ("trafilatura", "bs4"))
check("keeps the real headline", "The Real Headline" in extracted_page)
check("keeps the real article body", "actual article body" in extracted_page)
check("drops the nav breadcrumb", "Home | About | Contact" not in extracted_page)
check("drops the cookie banner", "cookies" not in extracted_page.lower())
check("drops the footer copyright line", "All rights reserved" not in extracted_page)

# 2. Plain prose with only a couple of leftover inline tags (the common case in
# already-scraped corpora, e.g. Common Crawl derivatives) should NOT be
# misdetected as a full HTML document, and should just get tags stripped.
raw_snippet = "Check out this <video>cool clip</video> and read the <h3>heading</h3> below."
extracted_snippet, meta_snippet = extract_content(raw_snippet)
show("Plain prose with stray inline tags", raw_snippet, extracted_snippet, meta_snippet)
check("NOT detected as a full HTML document", not looks_like_html_document(raw_snippet))
check("inline tags path used", meta_snippet["path"] == "inline_tags_stripped")
check("tag markup removed", "<video>" not in extracted_snippet and "<h3>" not in extracted_snippet)
check("surrounding words kept", "cool clip" in extracted_snippet and "heading" in extracted_snippet)

# 3. Plain prose with no tags at all should pass through unchanged.
raw_plain = "Just an ordinary sentence with no markup at all."
extracted_plain, meta_plain = extract_content(raw_plain)
show("Plain prose, no tags", raw_plain, extracted_plain, meta_plain)
check("passthrough path used", meta_plain["path"] == "passthrough")
check("text unchanged", extracted_plain == raw_plain)

# 4. Empty input handled without raising.
check("empty string handled", extract_content("") == ("", {"path": "empty"}))
check("None handled", extract_content(None) == ("", {"path": "empty"}))

# 5. Tables: flattening a real <table> with get_text() jumbles cells together
# with no separator -- worse than dropping it -- so both the bs4 tier and the
# regex-only fallback tier drop table content entirely rather than flattening.
raw_table_page = """
<article>
<p>Intro paragraph before the table.</p>
<table><tr><th>Name</th><th>Score</th></tr><tr><td>Alice</td><td>90</td></tr></table>
<p>Outro paragraph after the table.</p>
</article>
"""
bs4_result = _extract_with_bs4(raw_table_page)
show("bs4 tier: table dropped", raw_table_page, bs4_result, {})
check("bs4 tier keeps intro paragraph", bs4_result is not None and "Intro paragraph" in bs4_result)
check("bs4 tier keeps outro paragraph", bs4_result is not None and "Outro paragraph" in bs4_result)
check("bs4 tier drops table cell content", bs4_result is not None and "Alice" not in bs4_result and "90" not in bs4_result)

raw_table_snippet = "Before table <table><tr><td>Alice</td><td>90</td></tr></table> after table"
lightweight_result = _strip_tags_lightweight(raw_table_snippet)
show("Regex fallback tier: table dropped", raw_table_snippet, lightweight_result, {})
check("lightweight tier keeps text before the table", "Before table" in lightweight_result)
check("lightweight tier keeps text after the table", "after table" in lightweight_result)
check("lightweight tier drops table cell content", "Alice" not in lightweight_result and "90" not in lightweight_result)

print(f"\n{checks_run - checks_failed}/{checks_run} checks passed.")
if checks_failed:
    sys.exit(1)

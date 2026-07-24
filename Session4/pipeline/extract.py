"""Stage 0 of the pipeline: Extraction.

Runs BEFORE Normalize + Clean. Turns raw HTML into clean prose: pulls the
article body out of the page and drops everything around it -- nav,
boilerplate, cookie banners, footers. Feeding un-extracted HTML into
clean_text() is what produces the ghost-token noise (<h3>, <video>, ...)
seen in earlier runs; this stage exists to remove that at the source.

Three tiers, tried in order, so the pipeline degrades gracefully depending
on what's installed and what the input actually looks like:

  1. trafilatura (if installed) -- purpose-built main-content extraction,
     best quality for genuine HTML pages.
  2. BeautifulSoup structural removal -- drop <script>/<style>/<nav>/
     <footer>/<header>/<aside>, and any element whose class/id names a
     cookie banner, subscribe prompt, sidebar, ad, etc.; then prefer an
     <article>/<main>/role="main"] container if one exists.
  3. Regex tag-stripping + boilerplate-line filter -- last resort, and also
     the path used for text that only has a few incidental leftover tags
     rather than a full HTML document (the common case in pre-scraped
     corpora, e.g. a stray <video> or <h3> that survived an upstream
     cleaner).

extract_content() never raises on malformed input and never drops a
document to empty because extraction failed -- worst case it falls back to
the raw text unchanged, so Normalize + Clean still has something to work
with.
"""

import re
import warnings

try:
    from bs4 import XMLParsedAsHTMLWarning

    warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
except ImportError:
    pass

_HTML_DOC_RE = re.compile(r"<!doctype\s+html|<html[\s>]", re.IGNORECASE)
_TAG_RE = re.compile(r"<[^>]+>")

# Page-chrome tags: seeing any of these is a strong signal of a real HTML
# document (nav/footer/etc. don't show up in inline leftover-tag prose),
# regardless of document length.
_STRUCTURAL_TAG_RE = re.compile(
    r"</?(?:html|body|head|nav|footer|header|aside|script|style|form|main|article)\b",
    re.IGNORECASE,
)

# Tag density above this fraction of the string signals a (near-)full HTML
# document rather than prose with a few incidental leftover tags. Only
# trusted past a minimum length -- a short snippet with just two or three
# inline tags (e.g. "<video>...</video>") can otherwise cross this ratio
# without being a real page.
_TAG_DENSITY_THRESHOLD = 0.02
_TAG_DENSITY_MIN_LEN = 200

_DROP_TAG_NAMES = [
    "script", "style", "nav", "footer", "header", "aside", "form", "noscript", "iframe",
    # Flattening a real <table> with .get_text() loses row/column structure
    # entirely -- cells run together with no separator, which reads as
    # worse noise than just dropping the table. Drop rather than flatten.
    "table",
]

# Same rationale as the "table" entry above, for the regex-only fallback
# tier (no bs4/trafilatura available): drop whole table blocks rather than
# stripping just the tags and leaving jumbled cell text behind.
_TABLE_BLOCK_RE = re.compile(r"<table\b[^>]*>.*?</table>", re.IGNORECASE | re.DOTALL)

_BOILERPLATE_ATTR_HINTS = (
    "cookie", "consent", "gdpr", "banner", "subscribe", "newsletter",
    "sidebar", "breadcrumb", "advert", "promo", "footer", "nav-", "navbar",
    "menu", "social-share", "related-posts", "comment", "popup", "modal",
)

# Boilerplate LINES that survive as plain text even after tags are gone
# (cookie banners / nav breadcrumbs baked into a plain <p> or <div>, no
# identifying class/id to key off of).
_BOILERPLATE_LINE_RE = re.compile(
    r"^\s*("
    r"we use cookies.{0,80}"
    r"|accept (all )?cookies.{0,40}"
    r"|this (website|site) uses cookies.{0,80}"
    r"|subscribe to our newsletter.{0,80}"
    r"|sign up for our newsletter.{0,80}"
    r"|all rights reserved\.?"
    r"|copyright \xa9? ?\d{4}.{0,80}"
    r"|skip to (main )?content"
    r"|home\s*\|\s*about\s*\|\s*contact.{0,40}"
    r")\s*$",
    re.IGNORECASE,
)


def looks_like_html_document(s):
    if not s:
        return False
    if _HTML_DOC_RE.search(s):
        return True
    if _STRUCTURAL_TAG_RE.search(s):
        return True
    if len(s) < _TAG_DENSITY_MIN_LEN:
        return False
    tag_chars = sum(len(m.group(0)) for m in _TAG_RE.finditer(s))
    return (tag_chars / len(s)) > _TAG_DENSITY_THRESHOLD


def _strip_boilerplate_lines(s):
    kept = [ln for ln in s.split("\n") if not _BOILERPLATE_LINE_RE.match(ln)]
    return "\n".join(kept)


def _strip_tags_lightweight(s):
    s = _TABLE_BLOCK_RE.sub(" ", s)
    return _strip_boilerplate_lines(_TAG_RE.sub(" ", s))


def _extract_with_trafilatura(html_str):
    try:
        import trafilatura
    except ImportError:
        return None
    try:
        return trafilatura.extract(html_str, include_comments=False, include_tables=False, favor_recall=True)
    except Exception:
        return None


def _extract_with_bs4(html_str):
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        return None
    try:
        soup = BeautifulSoup(html_str, "html.parser")
    except Exception:
        return None

    # find_all(True) below collects the whole tag list up front, but
    # decompose()-ing a tag also invalidates its still-to-be-visited
    # children in that same list (their .attrs gets cleared) -- skip
    # anything already decomposed as a side effect of an earlier iteration.
    for tag_name in _DROP_TAG_NAMES:
        for tag in soup.find_all(tag_name):
            if not tag.decomposed:
                tag.decompose()

    for tag in soup.find_all(True):
        if tag.decomposed:
            continue
        classes = tag.get("class") or []
        attr_blob = " ".join(classes + [tag.get("id", "") or ""]).lower()
        if any(hint in attr_blob for hint in _BOILERPLATE_ATTR_HINTS):
            tag.decompose()

    main = soup.find("article") or soup.find("main") or soup.find(attrs={"role": "main"})
    root = main if main is not None else soup

    text = root.get_text(separator="\n")
    return _strip_boilerplate_lines(text)


def extract_content(raw, is_html_hint=None):
    """Returns (extracted_text, meta) where meta = {"path": <which tier ran>}."""
    if not raw:
        return "", {"path": "empty"}

    is_html = is_html_hint if is_html_hint is not None else looks_like_html_document(raw)

    if is_html:
        extracted = _extract_with_trafilatura(raw)
        if extracted:
            return extracted, {"path": "trafilatura"}

        extracted = _extract_with_bs4(raw)
        if extracted:
            return extracted, {"path": "bs4"}

        return _strip_tags_lightweight(raw), {"path": "html_fallback_stripped"}

    if _TAG_RE.search(raw):
        return _strip_tags_lightweight(raw), {"path": "inline_tags_stripped"}

    return raw, {"path": "passthrough"}

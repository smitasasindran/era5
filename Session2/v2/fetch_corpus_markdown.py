"""
Fetch the raw wikitext of the "India" article in each language and convert
it to clean, "wiki-faithful" Markdown -- preserving headings, bold/italic,
links, lists, and tables -- instead of the flattened plain-text extract used
in v1 (which drops all structure and a lot of infobox/table content).

The structural markup (##, **, |, [text](target), list bullets) is itself
highly repetitive across the whole article and, since it's plain ASCII, is
shared cheaply across all 4 languages' scripts -- part of why this corpus
should compress much better than v1's plain text.
"""
import html
import re
import requests
import mwparserfromhell as mwph

TITLES = {
    "en": "India",
    "hi": "भारत",
    "te": "భారతదేశం",
    "mr": "भारत",
}

CITATION_TEMPLATE_PREFIXES = ("cite", "citation", "sfn", "refn", "efn", "reflist", "r ")


def fetch_wikitext(lang, title):
    url = f"https://{lang}.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "prop": "revisions",
        "rvprop": "content",
        "rvslots": "main",
        "titles": title,
        "format": "json",
        "redirects": 1,
    }
    r = requests.get(url, params=params, headers={"User-Agent": "tokenizer-assignment/1.0"})
    r.raise_for_status()
    pages = r.json()["query"]["pages"]
    page = next(iter(pages.values()))
    return page["revisions"][0]["slots"]["main"]["*"]


def plain(wikicode_fragment):
    """Fallback for nested content: strip to plain text rather than recursing."""
    if wikicode_fragment is None:
        return ""
    raw = str(wikicode_fragment)
    if not raw.strip():
        return ""
    try:
        return mwph.parse(raw).strip_code().strip()
    except Exception:
        return raw.strip()


def convert_table(tag):
    """Best-effort wikitable -> markdown table. Not pixel-perfect (no
    rowspan/colspan), but preserves cell content faithfully as text."""
    body = str(tag.contents) if tag.contents else ""
    rows = []
    current_row = []
    for line in body.split("\n"):
        line = line.strip()
        if not line or line.startswith("|-") or line.startswith("|}"):
            if current_row:
                rows.append(current_row)
                current_row = []
            continue
        if line.startswith("!"):
            cells = [plain(c.split("|")[-1]) for c in line[1:].split("!!")]
            current_row.extend(cells)
        elif line.startswith("|"):
            cells = [plain(c) for c in line[1:].split("||")]
            current_row.extend(cells)
    if current_row:
        rows.append(current_row)

    if not rows:
        return ""

    out = ["| " + " | ".join(rows[0]) + " |"]
    out.append("|" + "|".join(["---"] * len(rows[0])) + "|")
    for row in rows[1:]:
        out.append("| " + " | ".join(row) + " |")
    return "\n".join(out) + "\n\n"


def convert_template(template):
    name = str(template.name).strip().lower()
    if any(name.startswith(p) for p in CITATION_TEMPLATE_PREFIXES):
        return ""
    # Infobox-style templates: render each parameter's value as plain text,
    # skipping short/empty/purely-numeric-key params.
    parts = []
    for param in template.params:
        value = plain(param.value)
        if value and len(value) > 1:
            parts.append(value)
    return (", ".join(parts) + "\n\n") if parts else ""


def convert_node(node):
    tname = type(node).__name__

    if tname == "Text":
        return str(node)

    if tname == "Heading":
        return "\n" + "#" * node.level + " " + plain(node.title) + "\n\n"

    if tname == "Wikilink":
        text = plain(node.text) if node.text else plain(node.title)
        return f"[{text}]({plain(node.title)})"

    if tname == "ExternalLink":
        text = plain(node.title) if node.title else str(node.url)
        return f"[{text}]({node.url})"

    if tname == "Template":
        return convert_template(node)

    if tname == "Comment":
        return ""

    if tname == "HTMLEntity":
        try:
            return html.unescape(str(node))
        except Exception:
            return str(node)

    if tname == "Tag":
        tag = node.tag.lower() if node.tag else ""
        wiki_markup = node.wiki_markup or ""

        if tag == "ref" or tag == "references":
            return ""
        if tag == "table" or wiki_markup == "{|":
            return convert_table(node)
        if tag == "b":
            return f"**{plain(node.contents)}**"
        if tag == "i":
            return f"*{plain(node.contents)}*"
        if tag == "li":
            prefix = "1. " if wiki_markup.strip("*") == "#" or wiki_markup == "#" else "- "
            return "\n" + prefix
        if tag in ("br",):
            return "\n"
        # Unknown tag: keep the inner text, drop the wrapper.
        return plain(node.contents) if node.contents else ""

    # Argument or anything else unhandled: best-effort plain text.
    return plain(node)


def wikitext_to_markdown(wikitext):
    code = mwph.parse(wikitext)
    parts = [convert_node(n) for n in code.nodes]
    text = "".join(parts)

    # Cleanup: collapse excess blank lines / spaces left by dropped nodes.
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    return text.strip()


if __name__ == "__main__":
    import os

    out_dir = os.path.join(os.path.dirname(__file__), "corpus")
    os.makedirs(out_dir, exist_ok=True)

    for lang, title in TITLES.items():
        wikitext = fetch_wikitext(lang, title)
        markdown = wikitext_to_markdown(wikitext)
        out_path = os.path.join(out_dir, f"india_{lang}.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(markdown)
        print(f"{lang}: {len(wikitext)} wikitext chars -> {len(markdown)} markdown chars -> {out_path}")

"""
Fetch India Wikipedia pages via the rendered-HTML REST endpoint and convert
to Markdown with markdownify, preserving links/tables/references/navboxes/
categories -- adapted from solution2/build_wiki_faithful_markdown.py, kept
for our own 4 languages (English/Hindi/Telugu/Marathi) instead of swapping
in a different language set.

This is the key corpus difference vs v2: v2 parsed raw *wikitext* with
mwparserfromhell and deliberately dropped citations/refs/infobox structure
as "clutter". This fetches the fully-rendered HTML (same content a browser
shows) and strips only script/style/meta -- keeping everything else, which
is why the instructor's corpus was 5-8x richer than ours and achieved much
lower absolute fertility.
"""
import json
import re
import time
from pathlib import Path
from urllib.parse import quote, urljoin

import requests
from bs4 import BeautifulSoup
from markdownify import markdownify as md

OUT = Path(__file__).resolve().parent / "corpus"
USER_AGENT = "tokenizer-assignment-v3/1.0"

PAGES = {
    "en": ("English", "India"),
    "hi": ("Hindi", "भारत"),
    "te": ("Telugu", "భారతదేశం"),
    "mr": ("Marathi", "भारत"),
}


def get(url):
    return requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=(8, 30))


def absolutize_links(soup, lang):
    base = f"https://{lang}.wikipedia.org/wiki/"
    for tag in soup.find_all(["a", "img", "source"]):
        attr = "href" if tag.name == "a" else "src"
        value = tag.get(attr)
        if not value:
            continue
        if value.startswith("//"):
            tag[attr] = "https:" + value
        elif value.startswith("./"):
            tag[attr] = urljoin(base, value[2:])
        elif value.startswith("/"):
            tag[attr] = urljoin(f"https://{lang}.wikipedia.org", value)


def strip_only_technical_noise(node, soup):
    for tag in node(["script", "style", "meta"]):
        tag.decompose()
    for tag in node.find_all("link"):
        rel = " ".join(tag.get("rel") or [])
        href = tag.get("href") or ""
        if "mw:PageProp/Category" in rel and href:
            tag.replace_with(soup.new_string(f"\nCategory: {href}\n"))
        else:
            tag.decompose()


def normalize_markdown(markdown):
    markdown = markdown.replace("\xa0", " ")
    markdown = re.sub(r"\n{4,}", "\n\n\n", markdown)
    markdown = re.sub(r"[ \t]+\n", "\n", markdown)
    return markdown.strip() + "\n"


def build_one(lang, title):
    OUT.mkdir(parents=True, exist_ok=True)
    url = f"https://{lang}.wikipedia.org/api/rest_v1/page/html/{quote(title)}"
    md_path = OUT / f"india_{lang}.txt"
    meta_path = OUT / f"india_{lang}.meta.json"

    res = get(url)
    res.raise_for_status()

    soup = BeautifulSoup(res.text, "lxml")
    body = soup.find("body") or soup
    strip_only_technical_noise(body, soup)
    absolutize_links(body, lang)
    markdown = normalize_markdown(md(str(body), heading_style="ATX", bullets="-", strip=["span"]))

    md_path.write_text(markdown, encoding="utf-8")
    meta = {
        "lang": lang,
        "title": title,
        "source_url": url,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "chars": len(markdown),
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


if __name__ == "__main__":
    for code, (name, title) in PAGES.items():
        meta = build_one(code, title)
        print(f"{code} {name}: {meta['chars']} chars -> {OUT / f'india_{code}.txt'}")

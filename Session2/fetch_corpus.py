"""Fetch plaintext extracts of the 'India' Wikipedia page in four languages."""
import requests
import os

TITLES = {
    "en": "India",
    "hi": "भारत",
    "te": "భారతదేశం",
    "mr": "भारत",
}

OUT_DIR = os.path.join(os.path.dirname(__file__), "corpus")
os.makedirs(OUT_DIR, exist_ok=True)


def fetch(lang, title):
    url = f"https://{lang}.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "prop": "extracts",
        "explaintext": 1,
        "titles": title,
        "format": "json",
        "redirects": 1,
    }
    r = requests.get(url, params=params, headers={"User-Agent": "tokenizer-assignment/1.0"})
    r.raise_for_status()
    pages = r.json()["query"]["pages"]
    page = next(iter(pages.values()))
    return page.get("extract", "")


if __name__ == "__main__":
    for lang, title in TITLES.items():
        text = fetch(lang, title)
        out_path = os.path.join(OUT_DIR, f"india_{lang}.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"{lang}: {len(text)} chars -> {out_path}")

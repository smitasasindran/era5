#!/usr/bin/env python3
"""Add or refresh one dataset profile in report/index.html's embedded data block.

The HTML file is the single source of truth for both structure and data --
this script only replaces the JSON inside the `<script id="report-data">`
block, which holds a list of named "profiles" (one per dataset the report
can show via its dropdown). Every number, bar chart, and percentage on the
page recomputes from whichever profile is selected, at render time.

Each profile = {dataset info, extraction report, normalize report}. This
script updates exactly one profile (matched by --profile-key) and leaves
every other profile in the file untouched.

Usage
-----
Add or refresh a profile, with its narrative copy in a small JSON file:
    python report/build_report.py \\
        --profile-key c4_realnewslike \\
        --meta report/profiles/c4_realnewslike.meta.json \\
        --extraction out/step0_c4_realnewslike_extracted_report.json \\
        --normalize out/step1_c4_realnewslike_clean_report.json

--meta is a JSON object with: label, description, highlight, facts (a list
of [label, value] pairs for the dataset-facts table), and optional narrative
asides: script_sanity_note, ghost_token_note, finding (a concise "we found
and fixed a bug" callout, separate from highlight's general dataset
description). See report/profiles/*.meta.json for real examples.

List the profiles currently in the file:
    python report/build_report.py --list

Reorder the dropdown (any keys not listed keep their relative order, appended
at the end):
    python report/build_report.py --order github_code_python,openr1_math,c4_realnewslike,capstone

By default this edits report/index.html in place. Pass --output to write
somewhere else instead and leave index.html untouched.
"""

import argparse
import json
import re
import sys
from pathlib import Path
from html.parser import HTMLParser

DATA_BLOCK_RE = re.compile(
    r'(<script id="report-data" type="application/json">)(.*?)(</script>)',
    re.DOTALL,
)

VOID_TAGS = {"meta", "link", "br", "img", "input", "hr", "area", "base", "col", "embed", "source", "track", "wbr"}


class TagBalanceChecker(HTMLParser):
    def __init__(self):
        super().__init__()
        self.stack = []
        self.errors = []

    def handle_starttag(self, tag, attrs):
        if tag not in VOID_TAGS:
            self.stack.append(tag)

    def handle_startendtag(self, tag, attrs):
        pass  # self-closed tag (<meta ... />) -- neither opens nor needs a matching close

    def handle_endtag(self, tag):
        if not self.stack:
            self.errors.append(f"stray closing tag: {tag}")
            return
        if self.stack[-1] == tag:
            self.stack.pop()
        else:
            self.errors.append(f"mismatch: expected close for {self.stack[-1]!r}, got {tag!r}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--profile-key", default=None, help="Short id for this profile, e.g. 'c4_realnewslike'")
    p.add_argument("--meta", default=None, help="Path to a JSON file with label/description/highlight/facts/...")
    p.add_argument("--extraction", default=None, help="Path to the Extraction stage's report JSON (step0)")
    p.add_argument("--normalize", default=None, help="Path to the Normalize+Clean stage's report JSON (step1)")
    p.add_argument("--samples", default=None, help="Path to a samples JSON from report/fetch_samples.py")
    p.add_argument("--dataset-label", default=None, help="Override --meta's label")
    p.add_argument("--dataset-description", default=None, help="Override --meta's description")
    p.add_argument("--html", default=str(Path(__file__).parent / "index.html"), help="HTML file to read/update")
    p.add_argument("--output", default=None, help="Write to this path instead of updating --html in place")
    p.add_argument("--list", action="store_true", help="List existing profile keys/labels and exit")
    p.add_argument("--order", default=None, help="Comma-separated profile keys, in the desired dropdown order")
    return p.parse_args()


def load_data_block(html):
    m = DATA_BLOCK_RE.search(html)
    if not m:
        print("ERROR: could not find the #report-data script block in the HTML file.", file=sys.stderr)
        sys.exit(1)
    return m, json.loads(m.group(2))


def write_validated(html, m, new_data, out_path):
    new_json = json.dumps(new_data, indent=2, ensure_ascii=False)
    new_html = html[:m.start(2)] + "\n" + new_json + "\n" + html[m.end(2):]

    checker = TagBalanceChecker()
    checker.feed(new_html)
    if checker.stack or checker.errors:
        print("ERROR: refreshed HTML failed a tag-balance check -- not writing.", file=sys.stderr)
        print(f"  unclosed at EOF: {checker.stack}", file=sys.stderr)
        print(f"  errors: {checker.errors}", file=sys.stderr)
        sys.exit(1)

    out_path.write_text(new_html, encoding="utf-8")
    return new_html


def main():
    args = parse_args()
    html_path = Path(args.html)
    html = html_path.read_text(encoding="utf-8")
    m, current_data = load_data_block(html)
    profiles = current_data.get("profiles", [])

    if args.list:
        for p in profiles:
            print(f"{p['key']:20s} {p['dataset'].get('label', '')}")
        return

    if args.order:
        wanted = [k.strip() for k in args.order.split(",") if k.strip()]
        by_key = {p["key"]: p for p in profiles}
        unknown = [k for k in wanted if k not in by_key]
        if unknown:
            print(f"ERROR: --order names unknown profile key(s): {unknown}", file=sys.stderr)
            sys.exit(1)
        ordered = [by_key[k] for k in wanted] + [p for p in profiles if p["key"] not in wanted]
        out_path = Path(args.output) if args.output else html_path
        write_validated(html, m, {"profiles": ordered}, out_path)
        print(f"Wrote {out_path}")
        print(f"  profile order now: {[p['key'] for p in ordered]}")
        return

    if not args.profile_key or not args.extraction or not args.normalize:
        print("ERROR: --profile-key, --extraction, and --normalize are all required "
              "(unless using --list).", file=sys.stderr)
        sys.exit(1)

    meta = {}
    if args.meta:
        meta = json.loads(Path(args.meta).read_text(encoding="utf-8"))

    existing_profile = next((p for p in profiles if p["key"] == args.profile_key), None)

    extraction_report = json.loads(Path(args.extraction).read_text(encoding="utf-8"))
    normalize_report = json.loads(Path(args.normalize).read_text(encoding="utf-8"))

    dataset = {
        "label": meta.get("label", args.profile_key),
        "description": meta.get("description", ""),
        "highlight": meta.get("highlight", ""),
        "facts": meta.get("facts", []),
    }
    if meta.get("script_sanity_note"):
        dataset["script_sanity_note"] = meta["script_sanity_note"]
    if meta.get("finding"):
        dataset["finding"] = meta["finding"]
    if args.samples:
        dataset["samples"] = json.loads(Path(args.samples).read_text(encoding="utf-8"))
    elif existing_profile and "samples" in existing_profile.get("dataset", {}):
        # Carry over samples from a previous run instead of silently dropping
        # them when a refresh call doesn't repeat --samples.
        dataset["samples"] = existing_profile["dataset"]["samples"]
    if args.dataset_label is not None:
        dataset["label"] = args.dataset_label
    if args.dataset_description is not None:
        dataset["description"] = args.dataset_description

    if meta.get("ghost_token_note"):
        normalize_report = dict(normalize_report, ghost_token_note=meta["ghost_token_note"])

    new_profile = {
        "key": args.profile_key,
        "dataset": dataset,
        "extraction": extraction_report,
        "normalize": normalize_report,
    }

    profiles = [p for p in profiles if p["key"] != args.profile_key]
    profiles.append(new_profile)
    new_data = {"profiles": profiles}

    out_path = Path(args.output) if args.output else html_path
    write_validated(html, m, new_data, out_path)

    print(f"Wrote {out_path}")
    print(f"  profile: {args.profile_key} ({dataset['label']}), {normalize_report.get('n_docs')} documents")
    print(f"  profiles now in file: {[p['key'] for p in profiles]}")


if __name__ == "__main__":
    main()

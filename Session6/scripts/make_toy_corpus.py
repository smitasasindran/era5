#!/usr/bin/env python
"""Write the tiny hand-authored toy corpus to data/corpus/toy_corpus.parquet.

Run this once (or whenever tds/toy_corpus.py's TOY_DOCUMENTS changes) to
regenerate the fixture. scripts/run_toy_pipeline.py calls this
automatically, so you rarely need to run this one directly.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tds.toy_corpus import DEFAULT_TOY_CORPUS_PATH, TOY_DOCUMENTS, write_toy_corpus  # noqa: E402


def main():
    out_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_TOY_CORPUS_PATH
    path = write_toy_corpus(out_path)
    print(f"Wrote {len(TOY_DOCUMENTS)} documents -> {path}")


if __name__ == "__main__":
    main()

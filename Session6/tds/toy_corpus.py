"""A tiny, hand-authored corpus for fast, human-checkable pipeline validation.

Not a replacement for the real corpus (data/corpus/small_shard.parquet) --
this exists purely so the shard builder / manifest pipeline can be run and
its output verified by eye in under a second: 12 short documents spanning
all 5 capability lanes the real corpus also produces, short enough that
every one of them is readable right here.

Document at index 4 (toy-0005, document_id "doc-000004") is the one
registered as held-out by configs/eval_registry_toy.yaml. Document at
index 11 (toy-0012) is a deliberate exact-text duplicate of it under a
completely different source/domain/lane -- added specifically to prove the
eval firewall matches on content hash, not on document id, source, or
lane.
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

DEFAULT_TOY_CORPUS_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "corpus" / "toy_corpus.parquet"
)

# (id, source, domain, language, text) -- same columns load_corpus() expects
# from any real corpus file: id, source, domain, language, text.
TOY_DOCUMENTS = [
    ("toy-0001", "toy_web", "web", "en",
     "Local bakery wins award for best sourdough bread in the county fair this year."),
    ("toy-0002", "toy_news", "news", "en",
     "City council approves new bike lanes downtown after months of public debate."),
    ("toy-0003", "toy_code", "code", "en",
     "def add(a, b):\n    return a + b\n"),
    ("toy-0004", "toy_code", "code", "en",
     "def factorial(n):\n    if n == 0:\n        return 1\n    return n * factorial(n - 1)\n"),
    ("toy-0005", "toy_stackexchange", "code", "en",
     "Q: How do I check if a Python list is empty?\nA: Use `if not my_list:` -- "
     "an empty list is falsy, so this reads naturally as \"if the list has nothing in it\"."),
    ("toy-0006", "toy_stackexchange", "code", "en",
     "Q: How do you reverse a singly linked list?\nA: Iterate through it, re-pointing each "
     "node's next pointer to the previous node."),
    ("toy-0007", "toy_math", "math", "en",
     "The derivative of x^2 is 2x. This follows directly from the power rule of differentiation."),
    ("toy-0008", "toy_science", "science", "en",
     "Photosynthesis converts sunlight, water, and carbon dioxide into glucose and oxygen."),
    ("toy-0009", "toy_indic", "web", "hi",
     "यह एक हिंदी वाक्य है जो परीक्षण के लिए लिखा गया है।"),
    ("toy-0010", "toy_indic", "code", "bn",
     "এটি একটি বাংলা বাক্য যা পরীক্ষার জন্য লেখা হয়েছে।"),
    ("toy-0011", "toy_instruction", "instruction", "en",
     "Instruction: Summarize the following paragraph in one sentence.\n"
     "Input: The city council met for three hours to discuss the new budget.\n"
     "Output: The council spent three hours discussing the budget."),
]

# Exact-text duplicate of TOY_DOCUMENTS[4] (toy-0005), added under an
# unrelated source/domain so the two share nothing except content -- this
# is what proves the firewall matches on content hash, not identity.
TOY_DOCUMENTS.append(
    ("toy-0012", "toy_web_mirror", "web", "en", TOY_DOCUMENTS[4][4])
)

# document_id of the toy document configs/eval_registry_toy.yaml holds out.
# Row position determines document_id (see tds.corpus.load_corpus), so this
# only holds as long as TOY_DOCUMENTS' order doesn't change above index 4.
TOY_HELD_OUT_DOCUMENT_ID = "doc-000004"


def write_toy_corpus(out_path: str | Path = DEFAULT_TOY_CORPUS_PATH) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table(
        {
            "id": [d[0] for d in TOY_DOCUMENTS],
            "source": [d[1] for d in TOY_DOCUMENTS],
            "domain": [d[2] for d in TOY_DOCUMENTS],
            "language": [d[3] for d in TOY_DOCUMENTS],
            "text": [d[4] for d in TOY_DOCUMENTS],
        }
    )
    pq.write_table(table, out_path)
    return out_path

"""Document loaders for the cleaning pipeline.

Three sources are supported, selected by args.source:
  - "hf"      : a HuggingFace Hub dataset, optionally streamed (no full download)
  - "hf_disk" : a dataset previously saved locally with datasets.save_to_disk()
  - "local"   : a plain local file: .jsonl / .json / .csv / .parquet / .txt

Every loader yields plain dicts: {"id": str, "text": str}.

Text extraction from a row is controlled by two mutually-exclusive CLI options:
  --text-field NAME        a single column holds the whole document (default: "text")
  --text-fields A,B,C      join multiple columns instead -- for structured datasets
                           that split a document across fields (e.g. a math
                           dataset with separate problem/solution/answer columns).
                           Missing/empty fields are skipped; the rest are joined
                           with a blank line.
"""

import json
from pathlib import Path


def make_text_getter(args):
    """Build a row -> str function from --text-field / --text-fields."""
    fields_spec = getattr(args, "text_fields", None)
    if fields_spec:
        fields = [f.strip() for f in fields_spec.split(",") if f.strip()]

        def getter(row):
            parts = [str(row.get(f, "") or "") for f in fields]
            return "\n\n".join(p for p in parts if p)

        return getter

    field = args.text_field

    def getter(row):
        return row.get(field, "") or ""

    return getter


def _iter_hf(dataset, config, split, streaming, get_text, limit):
    from datasets import load_dataset

    ds = load_dataset(dataset, config, split=split, streaming=streaming)
    for i, row in enumerate(ds):
        if limit is not None and i >= limit:
            break
        yield {"id": str(row.get("id", i)), "text": get_text(row)}


def _iter_hf_disk(path, split, get_text, limit):
    from datasets import load_from_disk

    ds = load_from_disk(path)
    if split is not None and hasattr(ds, "keys") and split in ds:
        ds = ds[split]
    for i, row in enumerate(ds):
        if limit is not None and i >= limit:
            break
        yield {"id": str(row.get("id", i)), "text": get_text(row)}


def _iter_jsonl(path, get_text, limit):
    count = 0
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            if limit is not None and count >= limit:
                break
            obj = json.loads(line)
            yield {"id": str(obj.get("id", i)), "text": get_text(obj)}
            count += 1


def _iter_json(path, get_text, limit):
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data if isinstance(data, list) else data.get("data", [])
    for i, obj in enumerate(rows):
        if limit is not None and i >= limit:
            break
        yield {"id": str(obj.get("id", i)), "text": get_text(obj)}


def _iter_csv(path, get_text, limit):
    import pandas as pd

    df = pd.read_csv(path)
    for i, row in df.iterrows():
        if limit is not None and i >= limit:
            break
        yield {"id": str(row.get("id", i)), "text": get_text(row)}


def _iter_parquet(path, get_text, limit):
    import pandas as pd

    df = pd.read_parquet(path)
    for i, row in df.iterrows():
        if limit is not None and i >= limit:
            break
        yield {"id": str(row.get("id", i)), "text": get_text(row)}


def _iter_txt(path, limit, txt_mode):
    if txt_mode == "whole":
        yield {"id": "0", "text": path.read_text(encoding="utf-8")}
        return
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            yield {"id": str(i), "text": line.rstrip("\n")}


def _iter_local(path, get_text, limit, txt_mode):
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        yield from _iter_jsonl(path, get_text, limit)
    elif suffix == ".json":
        yield from _iter_json(path, get_text, limit)
    elif suffix == ".csv":
        yield from _iter_csv(path, get_text, limit)
    elif suffix == ".parquet":
        yield from _iter_parquet(path, get_text, limit)
    elif suffix == ".txt":
        yield from _iter_txt(path, limit, txt_mode)
    else:
        raise ValueError(
            f"Unsupported local file type: {suffix!r} (expected .jsonl/.json/.csv/.parquet/.txt)"
        )


def load_documents(args):
    get_text = make_text_getter(args)
    if args.source == "hf":
        if not args.dataset:
            raise ValueError("--dataset is required when --source hf")
        yield from _iter_hf(args.dataset, args.config, args.split, args.streaming, get_text, args.limit)
    elif args.source == "hf_disk":
        if not args.path:
            raise ValueError("--path is required when --source hf_disk")
        yield from _iter_hf_disk(args.path, args.split, get_text, args.limit)
    elif args.source == "local":
        if not args.path:
            raise ValueError("--path is required when --source local")
        yield from _iter_local(args.path, get_text, args.limit, args.txt_mode)
    else:
        raise ValueError(f"Unknown source: {args.source!r}")

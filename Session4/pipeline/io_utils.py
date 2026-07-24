"""Document loaders for the cleaning pipeline.

Three sources are supported, selected by args.source:
  - "hf"      : a HuggingFace Hub dataset, optionally streamed (no full download)
  - "hf_disk" : a dataset previously saved locally with datasets.save_to_disk()
  - "local"   : a plain local file: .jsonl / .json / .csv / .parquet / .txt

Every loader yields plain dicts: {"id": str, "text": str}.
"""

import json
from pathlib import Path


def _iter_hf(dataset, config, split, streaming, text_field, limit):
    from datasets import load_dataset

    ds = load_dataset(dataset, config, split=split, streaming=streaming)
    for i, row in enumerate(ds):
        if limit is not None and i >= limit:
            break
        yield {"id": str(row.get("id", i)), "text": row.get(text_field, "") or ""}


def _iter_hf_disk(path, split, text_field, limit):
    from datasets import load_from_disk

    ds = load_from_disk(path)
    if split is not None and hasattr(ds, "keys") and split in ds:
        ds = ds[split]
    for i, row in enumerate(ds):
        if limit is not None and i >= limit:
            break
        yield {"id": str(row.get("id", i)), "text": row.get(text_field, "") or ""}


def _iter_jsonl(path, text_field, limit):
    count = 0
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            if limit is not None and count >= limit:
                break
            obj = json.loads(line)
            yield {"id": str(obj.get("id", i)), "text": obj.get(text_field, "") or ""}
            count += 1


def _iter_json(path, text_field, limit):
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data if isinstance(data, list) else data.get("data", [])
    for i, obj in enumerate(rows):
        if limit is not None and i >= limit:
            break
        yield {"id": str(obj.get("id", i)), "text": obj.get(text_field, "") or ""}


def _iter_csv(path, text_field, limit):
    import pandas as pd

    df = pd.read_csv(path)
    for i, row in df.iterrows():
        if limit is not None and i >= limit:
            break
        yield {"id": str(row.get("id", i)), "text": str(row.get(text_field, "") or "")}


def _iter_parquet(path, text_field, limit):
    import pandas as pd

    df = pd.read_parquet(path)
    for i, row in df.iterrows():
        if limit is not None and i >= limit:
            break
        yield {"id": str(row.get("id", i)), "text": str(row.get(text_field, "") or "")}


def _iter_txt(path, limit, txt_mode):
    if txt_mode == "whole":
        yield {"id": "0", "text": path.read_text(encoding="utf-8")}
        return
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            yield {"id": str(i), "text": line.rstrip("\n")}


def _iter_local(path, text_field, limit, txt_mode):
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        yield from _iter_jsonl(path, text_field, limit)
    elif suffix == ".json":
        yield from _iter_json(path, text_field, limit)
    elif suffix == ".csv":
        yield from _iter_csv(path, text_field, limit)
    elif suffix == ".parquet":
        yield from _iter_parquet(path, text_field, limit)
    elif suffix == ".txt":
        yield from _iter_txt(path, limit, txt_mode)
    else:
        raise ValueError(
            f"Unsupported local file type: {suffix!r} (expected .jsonl/.json/.csv/.parquet/.txt)"
        )


def load_documents(args):
    if args.source == "hf":
        if not args.dataset:
            raise ValueError("--dataset is required when --source hf")
        yield from _iter_hf(
            args.dataset, args.config, args.split, args.streaming, args.text_field, args.limit
        )
    elif args.source == "hf_disk":
        if not args.path:
            raise ValueError("--path is required when --source hf_disk")
        yield from _iter_hf_disk(args.path, args.split, args.text_field, args.limit)
    elif args.source == "local":
        if not args.path:
            raise ValueError("--path is required when --source local")
        yield from _iter_local(args.path, args.text_field, args.limit, args.txt_mode)
    else:
        raise ValueError(f"Unknown source: {args.source!r}")

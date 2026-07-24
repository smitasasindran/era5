"""Output writers for the pipeline scripts.

Format is chosen by the --output file extension: .parquet writes Parquet,
anything else (.jsonl by convention) writes JSON Lines. This mirrors how
pipeline.io_utils already picks a loader by extension, so "the file I point
at tells you the format" stays true on both ends of the pipeline.

Parquet is written incrementally in row-group batches (ParquetWriter) so
memory stays bounded on large corpora -- callers don't need to hold the
whole dataset before writing. Any dict/list field (e.g. ghost_tokens) is
JSON-serialized to a string first, so every batch has the same flat scalar
schema regardless of which keys happen to be present in a given record.
"""

import json
from pathlib import Path


class JsonlWriter:
    def __init__(self, path):
        self._f = open(path, "w", encoding="utf-8")

    def write(self, record):
        self._f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def close(self):
        self._f.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


class ParquetWriter:
    def __init__(self, path, batch_size=2000):
        import pyarrow  # noqa: F401 -- fail fast here if pyarrow is missing
        import pyarrow.parquet  # noqa: F401

        self._pa = pyarrow
        self._pq = pyarrow.parquet
        self._path = str(path)
        self._batch_size = batch_size
        self._buffer = []
        self._writer = None

    @staticmethod
    def _flatten(record):
        return {
            k: (json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v)
            for k, v in record.items()
        }

    def write(self, record):
        self._buffer.append(self._flatten(record))
        if len(self._buffer) >= self._batch_size:
            self._flush()

    def _flush(self):
        if not self._buffer:
            return
        table = self._pa.Table.from_pylist(self._buffer)
        if self._writer is None:
            self._writer = self._pq.ParquetWriter(self._path, table.schema)
        self._writer.write_table(table)
        self._buffer = []

    def close(self):
        self._flush()
        if self._writer is not None:
            self._writer.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


def open_writer(path, batch_size=2000):
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        return ParquetWriter(path, batch_size=batch_size)
    return JsonlWriter(path)

"""Atomic file replacement (director standing rule, Session 22).

Any data file that may be modified WHILE a run reads it must be written atomically: write the
full new content to a temp file in the SAME directory, fsync, then os.replace() onto the target.
os.replace is atomic for same-volume paths (Windows: MoveFileEx REPLACE_EXISTING), so a
concurrent reader sees either the entire old file or the entire new file — never a truncated
half-write. The NCBI column patch used a plain pandas to_csv() overwrite (non-atomic); this
module is the sanctioned replacement.
"""
from __future__ import annotations
import os, tempfile
from pathlib import Path


def atomic_write_bytes(data: bytes, path: str | os.PathLike) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)          # atomic on same volume
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def atomic_write_text(text: str, path: str | os.PathLike, encoding: str = "utf-8") -> None:
    atomic_write_bytes(text.encode(encoding), path)


def atomic_write_csv(df, path: str | os.PathLike, **to_csv_kwargs) -> None:
    """Serialize df to CSV in memory, then atomically replace `path`. kwargs pass to to_csv."""
    to_csv_kwargs.setdefault("index", False)
    encoding = to_csv_kwargs.pop("encoding", "utf-8")
    atomic_write_bytes(df.to_csv(**to_csv_kwargs).encode(encoding), path)

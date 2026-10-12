"""Pick the reader that can parse a given input path."""

from __future__ import annotations

from pathlib import Path

from talent_angels.tagging.models import Document
from talent_angels.tagging.readers.base import Reader, ReaderError
from talent_angels.tagging.readers.neo4j_tree import Neo4jTreeReader
from talent_angels.tagging.readers.table import TableReader

READERS: tuple[Reader, ...] = (Neo4jTreeReader(), TableReader())


def read_document(path: Path) -> Document:
    """The first reader (in `READERS` order) that recognizes `path`."""
    for reader in READERS:
        if reader.can_read(path):
            return reader.read(path)
    raise ReaderError(f"no reader for {path.name}")

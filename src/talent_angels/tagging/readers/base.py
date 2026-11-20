"""The read step's contract: any input format becomes a `Document` of rows.

A new business's export (flat CSV/XLSX table, a different tree shape, ...)
only needs a new `Reader` implementation; nothing downstream of `read()`
knows or cares where the rows came from.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from talent_angels.tagging.models import Document


class ReaderError(ValueError):
    """A reader recognized the file but could not parse it into a `Document`."""


@runtime_checkable
class Reader(Protocol):
    name: str

    def can_read(self, path: Path) -> bool:
        """Cheap, side-effect-free check — never raises on a malformed file."""
        ...

    def read(self, path: Path) -> Document:
        """Parse `path`; raise `ReaderError` with a clear message if it can't."""
        ...

"""Safe writes for the plain-text memory files (USER.md, MEMORY.md).

Both files are updated read-modify-write. Served from the API threadpool, two
unlocked writers interleave and one update is silently lost (measured: 8
concurrent writers, 6 lines survived). Every update therefore runs under
:data:`MEMORY_FILE_LOCK` and lands with an atomic ``os.replace`` so a reader
never sees a half-written file.

The lock is per process. Two processes sharing one home (TUI + API) are still
serialised only by the atomic replace — last writer wins, but never a torn file.
"""

from __future__ import annotations

import os
import tempfile
import threading
from pathlib import Path

MEMORY_FILE_LOCK = threading.RLock()


def atomic_write_text(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` atomically, creating the parent directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def read_text_or_empty(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ""

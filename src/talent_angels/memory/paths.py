"""Where the agent's personal data lives on disk — one home, resolved per call.

Everything that holds a user's words lives under one directory: ``USER.md``,
``MEMORY.md``, ``memory.db``, ``checkpoints.db``, saved sessions, the run-log
and the query-detail dumps. One home means one erase story (ADR-0007).

Resolution order, evaluated on **every call** (never at import):

1. ``TA_AGENTS_HOME`` — the supported override (hosting volume, shared host).
   Because it is read per call, a value loaded from ``.env`` after import is
   honoured.
2. ``<repo>/.ta-agents`` when running from a source checkout (the directory
   holding this project's ``pyproject.toml``), so the memory sits next to the
   code a developer is working on and under the repo's ``.gitignore``.
3. ``~/.ta-agents`` for an installed (non-editable) package or a container —
   never a path inside ``site-packages``.

Nothing is created at import. Writers call :func:`ensure_home` (or create the
parent of the file they write); readers treat a missing home as empty.
"""

from __future__ import annotations

import os
from pathlib import Path

# src/talent_angels/memory/paths.py → candidate repository root.
_CANDIDATE_ROOT = Path(__file__).resolve().parents[3]
_PROJECT_NAME_MARKER = 'name = "talent-angels"'


def _is_source_checkout(root: Path) -> bool:
    pyproject = root / "pyproject.toml"
    try:
        return _PROJECT_NAME_MARKER in pyproject.read_text(encoding="utf-8")
    except OSError:
        return False


def project_root() -> Path | None:
    """The repository root when running from a checkout, else ``None``."""
    return _CANDIDATE_ROOT if _is_source_checkout(_CANDIDATE_ROOT) else None


def home() -> Path:
    """The memory home for this call. Pure: never creates anything."""
    override = os.environ.get("TA_AGENTS_HOME", "").strip()
    if override:
        return Path(override).expanduser()
    root = project_root()
    if root is not None:
        return root / ".ta-agents"
    return Path.home() / ".ta-agents"


def ensure_home() -> Path:
    """The memory home, created if needed. Call from write paths only."""
    path = home()
    path.mkdir(parents=True, exist_ok=True)
    return path


def user_md() -> Path:
    """The human profile (explicit-confirm writes only)."""
    return home() / "USER.md"


def memory_md() -> Path:
    """The agent's notes."""
    return home() / "MEMORY.md"


def db_path() -> Path:
    """``memory.db``: episodes, recall indexes, neighbour cache."""
    return home() / "memory.db"


def checkpoint_db_path() -> Path:
    """LangGraph per-thread checkpoints — personal data (questions, plans)."""
    return home() / "checkpoints.db"


def default_sessions_dir() -> Path:
    return home() / "sessions"


def default_runlog_path() -> Path:
    return home() / "runlog.jsonl"


def default_details_dir() -> Path:
    return home() / "query-details"

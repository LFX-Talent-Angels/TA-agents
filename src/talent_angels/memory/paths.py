"""Where the agent's memory lives on disk.

The home defaults to a directory **inside the project** (``.ta-agents/`` at the
repository root) rather than ``~/.ta-agents``, so the memory — SQLite stores,
``USER.md``, ``MEMORY.md``, checkpoints — sits where a developer already looks,
travels with the checkout, and is covered by the repo's ``.gitignore``
alongside sessions and the run-log. The root is derived from this file's own
path, never from the process working directory, so the answer is the same no
matter where the app is launched from.

Set ``TA_AGENTS_HOME`` to point elsewhere (a shared home, a hosting volume,
…); it is honoured before the default.
"""

import os
from pathlib import Path

# src/talent_angels/memory/paths.py → repository root (three parents up from `memory`)
_PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _default_home() -> Path:
    override = os.environ.get("TA_AGENTS_HOME")
    if override:
        return Path(override).expanduser()
    return _PROJECT_ROOT / ".ta-agents"


TA_HOME = _default_home()
TA_HOME.mkdir(parents=True, exist_ok=True)

USER_MD = TA_HOME / "USER.md"
MEMORY_MD = TA_HOME / "MEMORY.md"
DB_PATH = TA_HOME / "memory.db"

#: LangGraph's per-thread checkpoint state, written by ``assistant.graph`` when a
#: caller passes a ``thread_id``. Personal data on the same terms as the rest of
#: this home — it holds the question, the plan and the cited nodes of every turn
#: on the thread — so it lives under ``TA_HOME`` where ``erase_all`` sweeps and
#: where test isolation redirects, rather than in whatever cwd the API happens
#: to be served from.
CHECKPOINT_DB_PATH = TA_HOME / "checkpoints.db"
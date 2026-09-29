"""Global test isolation for the agent's memory stores.

The memory layer resolves its paths to module-level constants at import time
(``USER_MD``, ``MEMORY_MD``, ``DB_PATH``), defaulting to the real
project-local home (``<repo>/.ta-agents``). Without isolation, any test that writes or erases
memory touches the developer's actual profile and the actual ``memory.db`` —
and with ``/reset`` now purging episodes, a stray test call would silently
delete real history.

This was previously handled by a single module-level fixture in
``tests/test_session_turns.py`` that patched only ``USER_MD``/``MEMORY.md``
and only for that one module. Every other test module that touched memory was
unprotected, and ``DB_PATH`` was never isolated anywhere.

Two things this file is careful about:

- **Every binding site, not just the source.** ``memory.paths`` defines the
  constants, but each consumer does ``from ... import USER_MD``, so patching
  ``memory.paths`` alone reaches nobody. The table below is the real list of
  binding sites.
- **Self-healing.** A consumer that stops binding a constant (because ruff
  removed a now-unused import) is skipped rather than erroring, so a lint fix
  upstream can never turn into 350 test errors downstream.

Note ``memory.paths`` also does ``TA_HOME.mkdir()`` at import. That side
effect is unavoidable without restructuring ``paths`` and only creates an
empty directory, so it is left alone.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path

import pytest


@dataclass(frozen=True, slots=True)
class MemoryHome:
    """The per-test stand-in for the real memory home (``<repo>/.ta-agents``).

    Tests must seed files through these attributes, not through
    ``memory.paths`` — the store modules bind the constants into their own
    namespaces at import time, so ``memory.paths.USER_MD`` is still the *real*
    home path and writing to it would touch the developer's actual profile.
    """

    root: Path
    user_md: Path
    memory_md: Path
    db: Path
    checkpoint_db: Path


@pytest.fixture(autouse=True)
def _isolate_call_time_paths(
    tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Redirect the paths resolved from the environment at *call* time.

    ``session.store.sessions_dir()`` reads ``TA_SESSIONS_DIR`` on every call and
    ``runlog.writer`` reads ``RUNLOG_PATH`` the same way, so monkeypatching the
    module constant cannot reach them — the environment has to be set.

    This is not hygiene, it is safety. ``/reset-all`` sweeps every session
    directory under ``sessions_dir()`` on purpose, and ``erase_all`` deletes
    whatever ``RUNLOG_PATH``/``QUERY_DETAILS_DIR`` point at. Unisolated, a single
    test asserting the forget path would delete the developer's real saved
    sessions and run-log. (It had already been writing ~300 empty session dirs
    into the repo's ``data/local/sessions`` on every run.)
    """
    root = tmp_path_factory.mktemp("ta-sessions")
    monkeypatch.setenv("TA_SESSIONS_DIR", str(root / "sessions"))
    monkeypatch.setenv("RUNLOG_PATH", str(root / "runlog.jsonl"))
    monkeypatch.setenv("QUERY_DETAILS_DIR", str(root / "query-details"))


@pytest.fixture(autouse=True)
def _isolate_memory_home(
    tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> MemoryHome:
    """Point every memory store at a per-test temporary home."""
    root = tmp_path_factory.mktemp("ta-memory")
    home = MemoryHome(
        root=root,
        user_md=root / "USER.md",
        memory_md=root / "MEMORY.md",
        db=root / "memory.db",
        checkpoint_db=root / "checkpoints.db",
    )

    redirects = {
        "talent_angels.memory.profile": {"USER_MD": home.user_md},
        "talent_angels.memory.agent_notes": {"MEMORY_MD": home.memory_md},
        "talent_angels.memory.episodes": {"DB_PATH": home.db},
        "talent_angels.memory.cache": {"DB_PATH": home.db},
        "talent_angels.memory.erase": {"USER_MD": home.user_md, "MEMORY_MD": home.memory_md},
        "talent_angels.assistant.graph": {"CHECKPOINT_DB_PATH": home.checkpoint_db},
        "talent_angels.tui.app": {"MEMORY_MD": home.memory_md},
    }
    for module_name, attributes in redirects.items():
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError:
            # The store being redirected has not been added yet. This fixture is
            # autouse, so an unguarded import here fails *every* test in the
            # suite, not just the ones about memory. Skipping is safe: a module
            # that does not exist cannot read a real path.
            continue
        for name, isolated in attributes.items():
            if not hasattr(module, name):
                continue  # no longer bound here, so it never reads the real path
            monkeypatch.setattr(module, name, isolated)
    return home


@pytest.fixture
def memory_home(_isolate_memory_home: MemoryHome) -> MemoryHome:
    """Request this to seed a memory file; isolation is already applied."""
    return _isolate_memory_home

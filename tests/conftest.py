"""Global test isolation for the agent's memory stores.

Every personal-data store (``USER.md``, ``MEMORY.md``, ``memory.db``,
``checkpoints.db``, sessions, run-log, query details) resolves its path from
``TA_AGENTS_HOME`` / its own env var **at call time** (``memory/paths.py``), so
setting the environment per test is complete isolation. Without it, a test
that writes or erases memory would touch the developer's real profile, history
and saved sessions — ``/reset-all`` deliberately sweeps all of them.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest

# Checkpointed channels must only hold types registered in
# assistant/checkpoint.py; strict mode turns a missing registration (today a
# deprecation warning, later a hard failure in production) into a test failure.
os.environ.setdefault("LANGGRAPH_STRICT_MSGPACK", "true")
# The offline suite never loads an embedding model or calls a provider; tests
# that exercise vector recall pass an explicit (static) embedder.
os.environ.setdefault("TA_EMBEDDING_MODEL", "none")


@dataclass(frozen=True, slots=True)
class MemoryHome:
    """The per-test stand-in for the real memory home (``<repo>/.ta-agents``).

    Seed files through these attributes; they equal the ``memory.paths``
    resolvers for the duration of the test.
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

    # Every store resolves its path from TA_AGENTS_HOME on each call, so one
    # environment variable isolates all of them — no per-module patching.
    monkeypatch.setenv("TA_AGENTS_HOME", str(root))
    return home


@pytest.fixture
def memory_home(_isolate_memory_home: MemoryHome) -> MemoryHome:
    """Request this to seed a memory file; isolation is already applied."""
    return _isolate_memory_home


@pytest.fixture(autouse=True)
def _close_checkpoint_savers() -> Iterator[None]:
    """The durable saver is cached per home; each test has its own home."""
    yield
    from talent_angels.assistant.checkpoint import release_all

    release_all()

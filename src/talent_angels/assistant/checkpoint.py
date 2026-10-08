"""The durable LangGraph checkpointer: one shared saver per database file.

A conversation (``thread_id`` = session id) keeps its state in
``<memory home>/checkpoints.db`` so it survives a process restart. This module
owns that store's lifecycle:

- **One saver per file per process.** The first version opened a new SQLite
  connection for every graph it compiled and never closed it. The saver is now
  cached by resolved path (tests switch homes per test), shared across threads
  (``check_same_thread=False``; ``SqliteSaver`` serialises its own writes) and
  closed at interpreter exit.
- **Explicit serialisation allowlist.** Checkpointed channels hold our Pydantic
  contracts. LangGraph's msgpack serde deserialises unknown classes with a
  deprecation warning today and will refuse them in a future release, so every
  type we persist is registered here. Tests run with
  ``LANGGRAPH_STRICT_MSGPACK=true`` so a new, unregistered type fails loudly.
- **Erasure.** :func:`release` closes and forgets the cached saver *before* the
  eraser unlinks the file — a connection kept open to an unlinked database
  would silently write the next conversation into a file nobody can reach.
  :func:`forget_thread` deletes one conversation and nothing else.
"""

from __future__ import annotations

import atexit
import sqlite3
import threading
from pathlib import Path

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver

from talent_angels.memory.paths import checkpoint_db_path

#: Every class that can appear in a checkpointed channel.
PERSISTED_TYPES: tuple[tuple[str, str], ...] = (
    ("talent_angels.contracts.models", "AgentResult"),
    ("talent_angels.contracts.models", "NodeRef"),
    ("talent_angels.contracts.models", "EdgeRef"),
    ("talent_angels.contracts.models", "EvidencePointer"),
    ("talent_angels.assistant.planning", "ExecutionPlan"),
    ("talent_angels.assistant.planning", "Intent"),
    ("talent_angels.assistant.planning", "PlanStep"),
    ("talent_angels.assistant.llm_plan", "PlanDraft"),
    ("talent_angels.assistant.memo", "TurnMemo"),
    ("talent_angels.assistant.memo", "SuiteMemo"),
    ("talent_angels.runlog.models", "StageUsage"),
    ("talent_angels.runlog.models", "ToolCall"),
    ("talent_angels.llm.protocol", "LLMUsage"),
)

_LOCK = threading.Lock()
_SAVERS: dict[Path, tuple[sqlite3.Connection, SqliteSaver]] = {}


def serializer() -> JsonPlusSerializer:
    return JsonPlusSerializer(allowed_msgpack_modules=list(PERSISTED_TYPES))


def shared_checkpointer() -> SqliteSaver:
    """The process-wide saver for the current memory home."""
    path = checkpoint_db_path().resolve()
    with _LOCK:
        cached = _SAVERS.get(path)
        if cached is not None:
            return cached[1]
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(path, check_same_thread=False)
        saver = SqliteSaver(conn, serde=serializer())
        _SAVERS[path] = (conn, saver)
        return saver


def release(path: Path | None = None) -> None:
    """Close the cached saver for ``path`` (default: current home), if any."""
    target = (path or checkpoint_db_path()).resolve()
    with _LOCK:
        cached = _SAVERS.pop(target, None)
    if cached is not None:
        cached[0].close()


def release_all() -> None:
    with _LOCK:
        cached = list(_SAVERS.values())
        _SAVERS.clear()
    for conn, _saver in cached:
        conn.close()


def forget_thread(thread_id: str) -> bool:
    """Delete one conversation's checkpoints. False when there was no store."""
    if not checkpoint_db_path().exists():
        return False
    shared_checkpointer().delete_thread(thread_id)
    return True


atexit.register(release_all)

"""SQLite index of past turns, queryable by cited node id.

The run-log (``talent_angels.runlog``) already has this data — one JSON
line per turn, question/plan/cited nodes/warnings included. What it does
not have is a way to ask "which turns cited this node" without scanning
and parsing the whole file. This is that index: a small derived
projection of each RunLogRecord, not a second copy of the run-log's
telemetry (cost, tokens, latency stay in the run-log only).

``record_episode`` is called from ``assistant.turn`` next to the existing
``append_record``, so every turn that writes a run-log record also
indexes an episode.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from talent_angels.memory.paths import DB_PATH
from talent_angels.runlog.models import ResultSummary, RunLogRecord

_DB_FILENAME = "memory.db"

# Warnings that mean the turn did not land on a usable answer.
_UNSATISFIED_WARNINGS = frozenset(
    {"not_found", "ambiguous", "unsupported_connect_query", "no_matching_neighbors"}
)


def episodes_db_path() -> Path:
    return DB_PATH


def _connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS episodes (
            run_id TEXT PRIMARY KEY,
            ts TEXT NOT NULL,
            suite TEXT NOT NULL,
            capability TEXT NOT NULL,
            plan TEXT NOT NULL,
            question TEXT NOT NULL,
            node_labels TEXT NOT NULL,
            warnings TEXT NOT NULL,
            satisfied INTEGER NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS episode_nodes (
            run_id TEXT NOT NULL,
            node_id TEXT NOT NULL,
            PRIMARY KEY (run_id, node_id)
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_episode_nodes_node_id ON episode_nodes(node_id)")
    return conn


@dataclass(frozen=True)
class Episode:
    run_id: str
    ts: str
    suite: str
    capability: str
    plan: tuple[str, ...]
    question: str
    node_ids: tuple[str, ...]
    node_labels: tuple[str, ...]
    warnings: tuple[str, ...]
    satisfied: bool


def _is_satisfied(result: ResultSummary) -> bool:
    if not result.node_ids:
        return False
    if any(w.startswith("capability_not_implemented") for w in result.warnings):
        return False
    return not any(w in _UNSATISFIED_WARNINGS for w in result.warnings)


def record_episode(record: RunLogRecord, *, db_path: Path | None = None) -> None:
    """Index one turn. Idempotent — re-recording the same run_id replaces it."""
    path = db_path or episodes_db_path()
    capability = record.plan[-1] if record.plan else ""
    conn = _connect(path)
    try:
        conn.execute(
            "INSERT OR REPLACE INTO episodes "
            "(run_id, ts, suite, capability, plan, question, node_labels, warnings, satisfied) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                record.run_id,
                record.ts,
                record.suite,
                capability,
                json.dumps(record.plan),
                record.question,
                json.dumps(record.result.node_labels),
                json.dumps(record.result.warnings),
                int(_is_satisfied(record.result)),
            ),
        )
        conn.execute("DELETE FROM episode_nodes WHERE run_id = ?", (record.run_id,))
        conn.executemany(
            "INSERT OR IGNORE INTO episode_nodes (run_id, node_id) VALUES (?, ?)",
            [(record.run_id, node_id) for node_id in record.result.node_ids],
        )
        conn.commit()
    finally:
        conn.close()


def _row_to_episode(row: sqlite3.Row, node_ids: tuple[str, ...]) -> Episode:
    return Episode(
        run_id=row["run_id"],
        ts=row["ts"],
        suite=row["suite"],
        capability=row["capability"],
        plan=tuple(json.loads(row["plan"])),
        question=row["question"],
        node_ids=node_ids,
        node_labels=tuple(json.loads(row["node_labels"])),
        warnings=tuple(json.loads(row["warnings"])),
        satisfied=bool(row["satisfied"]),
    )


def _node_ids_for(conn: sqlite3.Connection, run_id: str) -> tuple[str, ...]:
    rows = conn.execute(
        "SELECT node_id FROM episode_nodes WHERE run_id = ? ORDER BY node_id", (run_id,)
    ).fetchall()
    return tuple(r[0] for r in rows)


def recent_episodes(*, limit: int = 20, db_path: Path | None = None) -> list[Episode]:
    """Newest turns first."""
    path = db_path or episodes_db_path()
    if not path.exists():
        return []
    conn = _connect(path)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute("SELECT * FROM episodes ORDER BY ts DESC LIMIT ?", (limit,)).fetchall()
        return [_row_to_episode(row, _node_ids_for(conn, row["run_id"])) for row in rows]
    finally:
        conn.close()


def episodes_citing(node_id: str, *, limit: int = 20, db_path: Path | None = None) -> list[Episode]:
    """Every past turn that cited ``node_id``, newest first."""
    path = db_path or episodes_db_path()
    if not path.exists():
        return []
    conn = _connect(path)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT e.* FROM episodes e "
            "JOIN episode_nodes n ON n.run_id = e.run_id "
            "WHERE n.node_id = ? ORDER BY e.ts DESC LIMIT ?",
            (node_id, limit),
        ).fetchall()
        return [_row_to_episode(row, _node_ids_for(conn, row["run_id"])) for row in rows]
    finally:
        conn.close()


def clear_episodes(*, db_path: Path | None = None) -> int:
    """Delete every recorded episode. Returns how many were removed."""
    path = db_path or episodes_db_path()
    if not path.exists():
        return 0
    conn = _connect(path)
    try:
        cursor = conn.execute("DELETE FROM episodes")
        conn.execute("DELETE FROM episode_nodes")
        conn.commit()
        return cursor.rowcount
    finally:
        conn.close()

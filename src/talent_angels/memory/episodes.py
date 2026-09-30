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

``episodes_fts`` is a lexical index over the same rows, maintained here rather
than by the retriever that reads it (``memory.fts_retriever``) so that recall
is a *derived* store: a database whose FTS half is damaged or absent is
rebuildable with ``sync_fts`` and a query away, and a user who never turns
recall on never pays for it. Writing it on the same transaction as the row is
what keeps the two in step, and the only way to keep them in step *at the byte
level* is to rebuild the index rather than delete rows from it — see
``_rebuild_fts``, and the measured reasons it is not ``DELETE FROM``.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from talent_angels.memory.paths import db_path
from talent_angels.runlog.models import ResultSummary, RunLogRecord

logger = logging.getLogger(__name__)

_DB_FILENAME = "memory.db"

#: The lexical index. ``run_id`` and ``ts`` are carried but not searched: the
#: first to replace a row and report the turn it came from, the second because
#: `EpisodeHit` promises one and a field that is always empty is a field nobody
#: trusts. ``question`` and ``node_labels`` are the searchable text — the label
#: list too, so "what does a nurse need" can match a turn that only ever
#: *answered* with the nurse node and never typed the word.
#:
#: Regular (content-carrying) rather than ``content=''``, on the evidence in
#: ``_rebuild_fts``: a contentless table stores the *same* inverted index, so it
#: leaks the same tokens and buys nothing here.
_FTS_DDL = """
        CREATE VIRTUAL TABLE IF NOT EXISTS episodes_fts USING fts5(
            run_id UNINDEXED,
            ts UNINDEXED,
            question,
            node_labels
        )
        """

# Warnings that mean the turn did not land on a usable answer.
_UNSATISFIED_WARNINGS = frozenset(
    {"not_found", "ambiguous", "unsupported_connect_query", "no_matching_neighbors"}
)


def episodes_db_path() -> Path:
    return db_path()


def _ensure_fts(conn: sqlite3.Connection) -> bool:
    """Create the lexical index if this SQLite can hold one. False if it cannot.

    ``CREATE VIRTUAL TABLE ... USING fts5`` raises on a build compiled without
    FTS5, and it raises inside ``_connect`` — which every episode write goes
    through. Since recall is an enhancement, an *optional index* must not be the
    reason a turn fails, so the absence is a warning and a ``False``, and every
    caller falls back to the plain table. A first-run warning is the honest
    signal that lexical recall is off on this machine.
    """
    try:
        conn.execute(_FTS_DDL)
    except sqlite3.Error:
        logger.warning(
            "this sqlite has no fts5; lexical episode recall is unavailable", exc_info=True
        )
        return False
    return True


def _rebuild_fts(conn: sqlite3.Connection) -> int:
    """Drop the lexical index and refill it from ``episodes``. Returns rows indexed.

    **Drop-and-recreate, not ``DELETE FROM episodes_fts``** — and the reason is
    measured, not theoretical. An FTS5 table stores the question *twice*: once
    in its own content rows, and once tokenised in the ``episodes_fts_data``
    inverted index. ``DELETE FROM episodes_fts`` only rewrites the content rows;
    at the index level it appends a *delete marker* to each term's posting list
    and leaves the postings themselves in place. Those rows are live as far as
    the b-tree is concerned, so ``PRAGMA secure_delete=ON`` never sees them, they
    are not free pages, and ``VACUUM`` has nothing to reclaim. Measured, asking
    for "I am Priya Raman from Bangalore" and then clearing:

    ==========  ==============  ===============  ============  =========
    after        ``episodes``   ``episodes_fts``  ``_fts_data``  PII tokens
    ==========  ==============  ===============  ============  =========
    DELETE       0               0                 **4**         **all 3**
    + VACUUM
    drop and     0               0                 **2**         **none**
    recreate
    ==========  ==============  ===============  ============  =========

    ``2`` is what FTS5 writes for an *empty* table — its structure and averages
    records — and is the floor any scrub can reach. ``4`` is a populated index
    whose postings have not been reclaimed.

    Retrieval was already clean in the leaking case, which is why every
    table-level test passed: `search` returns nothing, and only the file still
    remembers. A ``/reset-all`` that reports "cleared" while ``priya`` is
    readable in ``memory.db`` is not erasure.

    The rejected alternative was a contentless table (``content=''``), the
    obvious shape for a pure index. It does not help, for the same reason: a
    contentless table drops the duplicated *content rows* but keeps the *inverted
    index*, and FTS5's ``'delete'`` command appends exactly the same marker.
    Measured on the same database, contentless + ``'delete'`` + ``VACUUM`` still
    reported ``episodes_fts_data = 4`` with all three tokens in the bytes, and
    FTS5 rejects ``'rebuild'`` on a contentless table outright. Contentless would
    have added a rowid join to the retriever and bought nothing.

    So: DROP and recreate. ``episodes`` is the source of truth, ``fts`` is
    derived, and this makes the derivation total — there is no sequence of
    deletes that can leave residue, because there are no deletes.
    """
    rows = conn.execute(
        "SELECT run_id, ts, question, node_labels FROM episodes ORDER BY rowid"
    ).fetchall()
    conn.execute("DROP TABLE IF EXISTS episodes_fts")
    if not _ensure_fts(conn):
        return 0
    conn.executemany(
        "INSERT INTO episodes_fts (run_id, ts, question, node_labels) VALUES (?, ?, ?, ?)",
        [tuple(row) for row in rows],
    )
    return len(rows)


def _connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    # Overwrite deleted content instead of leaving it in free pages. Without
    # this, a cleared episode's question stays readable in the raw file with
    # `strings`, so "erased" would be true of the table and false of the data.
    # Costs a little write amplification; this database is tiny and local.
    conn.execute("PRAGMA secure_delete=ON")
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
    _ensure_fts(conn)
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
    """Index one turn. Idempotent — re-recording the same run_id replaces it.

    Replacing a turn is the one write path that could leave the *old* wording on
    disk, so it rebuilds the index instead of deleting the old row out of it
    (``_rebuild_fts``). A genuinely new ``run_id`` — every ordinary turn, since a
    run id is minted per turn — takes the cheap path: one insert, no rebuild, and
    no residue because there was nothing there to leak.
    """
    path = db_path or episodes_db_path()
    capability = record.plan[-1] if record.plan else ""
    conn = _connect(path)
    try:
        # Asked *before* the REPLACE, because REPLACE is what makes the question
        # unanswerable. A primary-key lookup on the source of truth, so the
        # branch costs one indexed read and decides whether a rebuild is owed.
        superseded = (
            conn.execute("SELECT 1 FROM episodes WHERE run_id = ?", (record.run_id,)).fetchone()
            is not None
        )
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
        if _ensure_fts(conn):
            if superseded:
                # A retried turn: the table keyed on run_id has already replaced
                # the row above, so rebuilding from it is both the replacement
                # and the scrub. `DELETE FROM episodes_fts WHERE run_id = ?` was
                # the previous mechanism and it left the old wording's tokens in
                # the file — see `_rebuild_fts` for the measurement.
                _rebuild_fts(conn)
            else:
                conn.execute(
                    "INSERT INTO episodes_fts (run_id, ts, question, node_labels) "
                    "VALUES (?, ?, ?, ?)",
                    (
                        record.run_id,
                        record.ts,
                        record.question,
                        # The same encoding the `episodes` row above uses. A second
                        # encoding for the same data would make `sync_fts` a guess.
                        json.dumps(record.result.node_labels),
                    ),
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


def _clear_vector_index(conn: sqlite3.Connection) -> None:
    """Drop the vector index table if this database has one. Never raises.

    The vector index (``episodes_vec``, see ``memory.vector_index``) is derived
    from the same episodes, so an erase that leaves it behind has not erased:
    the question is a function of the text, the vector is a function of the
    question, and "the user's name is gone from the file" is false while its
    embedding is still in it.

    It is dropped rather than emptied, for the same reason the FTS index is
    rebuilt above — and here the reason is stronger still, because the
    behaviour of ``vec0``'s shadow tables on ``DELETE`` is a property of the
    extension rather than of this code, so deleting the rows would be trusting
    it. Dropping removes the shadow tables with it.

    The table name is inlined rather than imported, because ``vector_index``
    imports this module: importing it back would be a cycle. It is also why the
    name is a module constant on both sides rather than a shared one.
    """
    try:
        row = conn.execute("SELECT 1 FROM sqlite_master WHERE name = 'episodes_vec'").fetchone()
        if row is None:
            return
        _load_vec_extension(conn)
        conn.execute("DROP TABLE IF EXISTS episodes_vec")
    except sqlite3.Error:
        # An extension that will not load, or a locked file. The episodes
        # themselves are already deleted by the caller; this is the derived
        # index, and failing the whole erase over it would be worse than
        # reporting it.
        logger.warning("could not drop the vector index; rerun `recall-rebuild`", exc_info=True)


def _load_vec_extension(conn: sqlite3.Connection) -> None:
    """Load ``sqlite-vec`` so ``episodes_vec`` can be dropped. Never raises.

    A ``vec0`` table cannot be dropped without its module, and this install may
    not have one — in which case there is no index to worry about anyway.
    """
    try:
        import sqlite_vec
    except ImportError:
        return
    try:
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
    except (AttributeError, sqlite3.Error):
        logger.debug("this SQLite build refuses to load extensions", exc_info=True)
    finally:
        try:
            conn.enable_load_extension(False)
        except (AttributeError, sqlite3.Error):
            pass


def clear_episodes(*, db_path: Path | None = None) -> int:
    """Delete every recorded episode. Returns how many were removed."""
    path = db_path or episodes_db_path()
    if not path.exists():
        return 0
    conn = _connect(path)
    try:
        cursor = conn.execute("DELETE FROM episodes")
        conn.execute("DELETE FROM episode_nodes")
        # Rebuild rather than delete the index's rows: `DELETE FROM episodes_fts`
        # empties the table and leaves every question's *tokens* in the file, so
        # this erase path reported "cleared" over a database that still contained
        # the user's name. See `_rebuild_fts` for the measurement.
        _rebuild_fts(conn)
        _clear_vector_index(conn)
        conn.commit()
        return cursor.rowcount
    finally:
        conn.close()


def sync_fts(*, db_path: Path | None = None) -> int:
    """Rebuild the lexical index from ``episodes``. Returns rows indexed.

    Needed after an upgrade: episodes recorded before the index existed are
    invisible to recall until this runs. Cheap on a personal corpus (thousands
    of rows), so it runs unconditionally rather than tracking a version.

    Recreates the table rather than deleting its rows, which buys two things at
    once. It repairs a half-built or out-of-date index — a ``CREATE TABLE IF NOT
    EXISTS`` cannot change the columns of a table that already exists, so a
    schema from an earlier build would otherwise leave every query failing with
    "no such column" and recall permanently, silently empty. And it is the only
    way to actually *remove* indexed text rather than unindex it: a row delete
    leaves the tokens in ``episodes_fts_data``, which is what `_rebuild_fts`
    measures. Rebuilding is safe to repeat — ``episodes`` is the source of truth
    and nothing is lost.

    Reachable by hand through ``python -m talent_angels.cli recall-rebuild``,
    which is also the manual scrub for a database written by a build that could
    not erase properly.
    """
    path = db_path or episodes_db_path()
    if not path.exists():
        return 0
    conn = _connect(path)
    try:
        indexed = _rebuild_fts(conn)
        conn.commit()
        return indexed
    finally:
        conn.close()


def vacuum_db(*, db_path: Path | None = None) -> bool:
    """Rewrite the database so free pages are returned to the filesystem.

    `PRAGMA secure_delete=ON` stops *future* deletes from lingering; this
    clears out whatever was already written before that pragma existed, and
    shrinks the file. Returns False when there is nothing to do — no database,
    or no database file on disk yet.
    """
    path = db_path or episodes_db_path()
    if not path.exists():
        return False
    conn = _connect(path)
    try:
        conn.execute("VACUUM")
        return True
    except sqlite3.Error:
        # A vacuum failure must not fail an erase that already deleted every
        # row; report honestly rather than raising.
        return False
    finally:
        conn.close()

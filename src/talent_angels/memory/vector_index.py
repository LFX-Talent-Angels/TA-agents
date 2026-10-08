"""Approximate nearest-neighbour search over episode embeddings, via sqlite-vec.

The second ``Retriever``'s engine, and the only part of the vector path that
touches disk. Kept separate from ``embeddings`` (the network) and
``vector_retriever`` (the ranking) so that this module can be tested with
synthetic vectors and no socket, which is where the bugs that matter live.

**Why the same database as the episodes.** ``memory.db`` already exists, is
already opened by ``memory.episodes``, and is already covered by ``/reset-all``
and the run-scoped erase. A second file would be a second thing to get wrong —
and the vector index is *derived personal data*, exactly the kind that this
repository has now leaked twice (``episodes_fts_data`` surviving ``DELETE``;
query details surviving an env-var path bug). One file is one erasure story.

**Why only ``run_id`` and the vector are stored here.** The question and the
labels stay in ``episodes``, and the rows are joined back on ``run_id``. That is
a privacy decision as much as a normalisation one: the text has exactly one
copy in the file, so the proof that erase removes it is a proof about one table
rather than a search across two. The vector is still derived from that text and
is still personal data — an embedding of "I am Priya Raman in Bangalore" is a
function of a name and a city, and it is deleted on the same paths as
everything else.

**The dimension is part of the schema, so it is part of the identity of this
index.** A ``vec0`` table created at 1536 dimensions cannot accept a 3072-dim
vector, and asking it to is a hard error rather than a silent truncation. So
``dimensions`` is part of ``open``'s signature, a mismatch is refused, and
changing the embedding model means a rebuild — the same bargain
``sync_fts`` already makes.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol, runtime_checkable

from talent_angels.memory.episodes import episodes_db_path
from talent_angels.memory.retrieval import RECALL_LIMIT, EpisodeHit

logger = logging.getLogger(__name__)

#: A local file answering microseconds should never make a turn wait a second.
_BUSY_TIMEOUT_S = 1.0

#: The table name. Not configurable: it is derived state, and a second name
#: would be a second index to drift.
_TABLE = "episodes_vec"

_SELECT = f"""
                SELECT v.run_id AS run_id, e.ts AS ts, e.question AS question,
                       e.node_labels AS node_labels, v.distance AS distance
                FROM {_TABLE} v
                JOIN episodes e ON e.run_id = v.run_id
                WHERE v.embedding MATCH ? AND k = ?
                ORDER BY v.distance
                """

#: Same drift check as the FTS index, and for the same reason: the queries do
#: not fail, so nothing else would say the index is behind the table it is
#: derived from. Counted on every search — it is a `COUNT(*)` on a local file,
#: measured at the same order as the FTS check that is already paid here.
_DRIFT_SQL = f"SELECT (SELECT COUNT(*) FROM {_TABLE}) < (SELECT COUNT(*) FROM episodes)"


@runtime_checkable
class VectorIndex(Protocol):
    """Store vectors, find near ones. What ``vector_retriever`` needs.

    Deliberately narrower than ``Retriever``: this ranks vectors, it does not
    know what an episode question is. The seam that hides embeddings from the
    prompt is ``Retriever``, and this sits underneath it.
    """

    @property
    def dimensions(self) -> int: ...

    @property
    def is_indexed(self) -> bool:
        """Whether an index exists at all, without needing a dimension.

        Askable without a width, and that is the point: a caller asking has no
        vector yet, so it would otherwise have to embed something to find out —
        a network round trip to learn there is nothing to search. Lets an
        unbuilt index be skipped instead of consulted and found empty.
        """
        ...

    def add(self, run_id: str, vector: Sequence[float]) -> None: ...

    def search(self, vector: Sequence[float], *, limit: int = RECALL_LIMIT) -> list[str]:
        """The ``run_id``\\ s of the nearest episodes, nearest first."""
        ...

    def clear(self) -> None: ...


def _load_extension(conn: sqlite3.Connection) -> bool:
    """Load ``sqlite_vec`` into ``conn``. False when the build has no such thing.

    Enabling extension loading requires the SQLite build to permit it, and the
    wheel has to be importable. Both can be absent — a stock ``libsqlite3``
    without loadable extensions, or a platform with no ``sqlite-vec`` wheel —
    and neither is worth an exception on the read path, so this reports rather
    than raises and the caller degrades to "no vector recall".
    """
    try:
        import sqlite_vec
    except ImportError:
        logger.debug("sqlite-vec is not installed; vector recall is unavailable")
        return False
    try:
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
    except (AttributeError, sqlite3.Error):
        logger.debug("this SQLite build refuses to load extensions", exc_info=True)
        return False
    finally:
        try:
            conn.enable_load_extension(False)
        except (AttributeError, sqlite3.Error):
            pass
    return True


class SqliteVecIndex:
    """A ``vec0`` index over episode embeddings, in the episode database."""

    def __init__(self, *, dimensions: int, db_path: Path | None = None) -> None:
        if dimensions <= 0:
            raise ValueError("dimensions must be positive")
        self._dimensions = dimensions
        self._db_path = db_path

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def _connect(self, *, create: bool) -> sqlite3.Connection | None:
        """Open the episode database with ``sqlite-vec`` loaded, or ``None``.

        A missing file is not an error — a first run has no ``memory.db``, and
        "no history yet" is the truth rather than a failure. On the read path
        (``create=False``) nothing is created: a search that has to build a
        table is not a search.
        """
        path = self._db_path or episodes_db_path()
        if not path.exists():
            return None
        try:
            conn = sqlite3.connect(path, timeout=_BUSY_TIMEOUT_S)
        except sqlite3.Error:
            logger.warning("vector recall could not open the episode database", exc_info=True)
            return None
        if not _load_extension(conn):
            conn.close()
            return None
        if create:
            self._ensure_table(conn)
        conn.row_factory = sqlite3.Row
        return conn

    def _existing_dimensions(self, conn: sqlite3.Connection) -> int | None:
        """The declared dimension of an existing table, or ``None``.

        Read from the table's own DDL because ``vec0`` has no portable
        "describe" and a wrong answer here would be a wrong answer everywhere
        else: two indexes over the same rows at different widths are not
        comparable, so this is how a model change is detected rather than
        discovered as inexplicably bad search results.
        """
        row = conn.execute("SELECT sql FROM sqlite_master WHERE name = ?", (_TABLE,)).fetchone()
        if row is None or not row[0]:
            return None
        marker = "float["
        start = row[0].lower().find(marker)
        if start < 0:
            return None
        rest = row[0][start + len(marker) :]
        digits = rest[: rest.find("]")] if "]" in rest else ""
        return int(digits) if digits.isdigit() else None

    def _ensure_table(self, conn: sqlite3.Connection) -> None:
        """Create the index table, or raise if it is a different width.

        Dropping and recreating on a width change is deliberately *not* done
        here. A silent rebuild would delete a user's index — and the embeddings
        in it cost money to recreate — as a side effect of opening a database,
        which is the kind of surprise that belongs behind an explicit command
        instead.
        """
        existing = self._existing_dimensions(conn)
        if existing is not None:
            if existing != self._dimensions:
                raise RuntimeError(
                    f"the existing {_TABLE} index is {existing}-dimensional but this "
                    f"embedder produces {self._dimensions}. They are not comparable. "
                    f"Run `recall-rebuild` to rebuild the index for the current model."
                )
            return
        conn.execute(
            f"CREATE VIRTUAL TABLE {_TABLE} USING vec0("
            f"run_id TEXT PRIMARY KEY, embedding FLOAT[{self._dimensions}])"
        )
        conn.commit()

    def _warn_if_stale(self, conn: sqlite3.Connection) -> None:
        """Say so when the index holds fewer rows than ``episodes``. Never raises."""
        try:
            stale = bool(conn.execute(_DRIFT_SQL).fetchone()[0])
        except sqlite3.Error:
            logger.debug("vector recall could not compare the index against the episodes table")
            return
        if stale:
            logger.warning(
                "vector recall index is out of date; run `recall-rebuild` "
                "(`python -m talent_angels.cli recall-rebuild`) to rebuild it — "
                "recorded turns are missing from vector recall until you do"
            )

    def add(self, run_id: str, vector: Sequence[float]) -> None:
        """Store or replace one episode's vector. Raises on a width mismatch.

        A read-path retriever never calls this, so recall stays a read; this is
        reached only from the explicit rebuild command.
        """
        if len(vector) != self._dimensions:
            raise ValueError(f"vector has {len(vector)} dimensions, index is {self._dimensions}")
        conn = self._connect(create=True)
        if conn is None:
            raise RuntimeError("could not open the episode database for a vector write")
        try:
            conn.execute(f"DELETE FROM {_TABLE} WHERE run_id = ?", (run_id,))
            conn.execute(
                f"INSERT INTO {_TABLE}(run_id, embedding) VALUES (?, ?)",
                (run_id, sqlite_vec_float32(vector)),
            )
            conn.commit()
        except sqlite3.Error:
            logger.warning("vector index write failed", exc_info=True)
            raise
        finally:
            conn.close()

    def search(self, vector: Sequence[float], *, limit: int = RECALL_LIMIT) -> list[str]:
        """Nearest ``run_id``\\ s, nearest first. Never raises.

        A missing index, an absent ``sqlite-vec``, a stale width and a locked
        file are all "nothing to recall" here, exactly as they are on the
        lexical path. Recall is an enhancement and must not be the reason a turn
        fails.
        """
        if limit <= 0 or len(vector) != self._dimensions:
            return []
        conn = self._connect(create=False)
        if conn is None:
            return []
        try:
            exists = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE name = ?", (_TABLE,)
            ).fetchone()
            if exists is None:
                return []
            self._warn_if_stale(conn)
            rows = conn.execute(_SELECT, (sqlite_vec_float32(vector), limit)).fetchall()
        except sqlite3.Error:
            logger.warning("vector recall failed; continuing without recall", exc_info=True)
            return []
        finally:
            conn.close()
        return [row["run_id"] for row in rows]

    def hits(self, vector: Sequence[float], *, limit: int = RECALL_LIMIT) -> list[EpisodeHit]:
        """Nearest episodes as ``EpisodeHit``\\ s, best first. Never raises.

        The join back to ``episodes`` happens here rather than in
        ``vector_retriever`` so that the row-to-hit mapping — including the
        "unreadable labels are dropped, not fatal" behaviour — has exactly one
        implementation, shared with the lexical backend's.
        """
        if limit <= 0 or len(vector) != self._dimensions:
            return []
        conn = self._connect(create=False)
        if conn is None:
            return []
        try:
            exists = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE name = ?", (_TABLE,)
            ).fetchone()
            if exists is None:
                return []
            self._warn_if_stale(conn)
            rows = conn.execute(_SELECT, (sqlite_vec_float32(vector), limit)).fetchall()
        except sqlite3.Error:
            logger.warning("vector recall failed; continuing without recall", exc_info=True)
            return []
        finally:
            conn.close()
        return [_hit(row) for row in rows]

    @property
    def is_indexed(self) -> bool:
        """True when the table is there. Never raises, and never needs a width.

        Checked on every search rather than cached, for the same reason the
        lexical index re-checks its own table each time: this is a `sqlite_master`
        lookup on a local file, and a cached "yes" would outlive the file.
        """
        if not self._dimensions:
            return False
        conn = self._connect(create=False)
        if conn is None:
            return False
        try:
            row = conn.execute("SELECT 1 FROM sqlite_master WHERE name = ?", (_TABLE,)).fetchone()
        except sqlite3.Error:
            return False
        finally:
            conn.close()
        return row is not None

    def count(self) -> int:
        """How many episodes are in the index. Zero when there is no index."""
        conn = self._connect(create=False)
        if conn is None:
            return 0
        try:
            row = conn.execute(f"SELECT COUNT(*) FROM {_TABLE}").fetchone()
        except sqlite3.Error:
            return 0
        finally:
            conn.close()
        return int(row[0]) if row else 0

    def clear(self) -> None:
        """Empty the index. Never raises.

        **Drop and recreate, not ``DELETE``** — the same conclusion reached for
        ``episodes_fts``, and for a different reason. There the inverted index
        kept the tokenised text alive after a delete; here the risk is the
        shadow tables ``vec0`` keeps alongside the virtual one, whose behaviour
        on delete is a property of the extension rather than of this code.
        Dropping the table and recreating it empty removes the shadow tables
        with it, and the measured cost is the same as a delete because neither
        is on the read path.
        """
        conn = self._connect(create=True)
        if conn is None:
            return
        try:
            conn.execute(f"DROP TABLE IF EXISTS {_TABLE}")
            conn.commit()
        except sqlite3.Error:
            logger.warning("vector index clear failed", exc_info=True)
        finally:
            conn.close()
        # Not recreated here: the next `add` calls `_connect(create=True)`, which
        # creates it. Leaving the table absent in between means `search` sees no
        # index and returns nothing, which is the correct answer for "cleared"
        # and is the same state a first run is in.


def sqlite_vec_float32(vector: Sequence[float]) -> bytes:
    """Pack a float vector into the little-endian float32 blob sqlite-vec wants.

    The format is part of the extension's contract, not an implementation
    detail, so it is one function rather than three call sites guessing at
    ``struct`` formats.
    """
    import struct

    return struct.pack(f"{len(vector)}f", *vector)


def _hit(row: sqlite3.Row) -> EpisodeHit:
    """One joined row to one hit. Degrades rather than raising."""
    try:
        labels = tuple(json.loads(row["node_labels"]))
    except (TypeError, ValueError):
        logger.warning("episode %s has unreadable labels; omitting them", row["run_id"])
        labels = ()
    # sqlite-vec's `distance` is smaller-is-better, so negate it to match the
    # seam's "higher is better" contract, exactly as the lexical backend does
    # with bm25. The arithmetic differs — cosine distance in [0, 2] against
    # bm25's unbounded negative — but the direction callers see is the same.
    return EpisodeHit(
        run_id=row["run_id"],
        ts=row["ts"],
        question=row["question"],
        node_labels=labels,
        score=-float(row["distance"] or 0.0),
    )

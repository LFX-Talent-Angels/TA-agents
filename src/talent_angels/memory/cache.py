"""SQLite cache for get_neighbors results, keyed by node id and relation types.

Sits below the Connect skill: caches the raw graph edges/nodes before the
essential/optional filter, so one cached entry serves every relation-kind
query for the same occupation (essential skills, optional skills, and a
plain neighbor dump all read the same row).

Node ids are already suite-prefixed (``esco:occupation:...`` vs
``onet:occupation:...``), so no separate suite column is needed.

The suite contract has no taxonomy-version signal today (checked: neither
ESCO nor O*NET expose one), so this invalidates on a plain TTL rather than
a real version bump. Revisit once TA-taxonomies exposes a version.
"""

from __future__ import annotations

import json
import sqlite3
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from talent_angels.memory.paths import db_path
from talent_angels.skills.connect.reveal import NeighborResult, TaxonomyEdge, TaxonomyNode
from talent_angels.skills.locate.resolve import SearchResult
from talent_angels.suites.protocol import SuiteTools
from talent_angels.suites.schema import SuiteSchema

DEFAULT_TTL_SECONDS = 24 * 60 * 60  # a day; graph facts don't move fast
_UNCACHEABLE_WARNINGS = frozenset({"node_not_found", "no_neighbors", "suite_unavailable"})
_DB_FILENAME = "memory.db"


def cache_db_path() -> Path:
    return db_path()


def _connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    # See the same pragma in memory/episodes.py: a DELETE must not leave the
    # old text readable in the file's free pages.
    conn.execute("PRAGMA secure_delete=ON")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS neighbor_cache (
            node_id TEXT NOT NULL,
            rel_types TEXT NOT NULL,
            payload TEXT NOT NULL,
            cached_at REAL NOT NULL,
            PRIMARY KEY (node_id, rel_types)
        )
        """
    )
    return conn


def _rel_key(rel_types: Sequence[str] | None) -> str:
    return ",".join(sorted(rel_types)) if rel_types else ""


@dataclass(frozen=True)
class _CachedNode:
    id: str
    kind: str
    label: str
    source: str
    source_id: str
    properties: Mapping[str, object]


@dataclass(frozen=True)
class _CachedEdge:
    type: str
    from_id: str
    to_id: str
    properties: Mapping[str, object]


@dataclass(frozen=True)
class _CachedNeighborResult:
    nodes: Sequence[_CachedNode]
    edges: Sequence[_CachedEdge]
    evidence: Sequence[str]
    warnings: Sequence[str]


def _node_to_dict(node: TaxonomyNode) -> dict[str, object]:
    return {
        "id": node.id,
        "kind": node.kind,
        "label": node.label,
        "source": node.source,
        "source_id": node.source_id,
        "properties": dict(node.properties),
    }


def _edge_to_dict(edge: TaxonomyEdge) -> dict[str, object]:
    return {
        "type": edge.type,
        "from_id": edge.from_id,
        "to_id": edge.to_id,
        "properties": dict(edge.properties),
    }


def _node_from_dict(data: dict[str, object]) -> _CachedNode:
    properties = data["properties"]
    return _CachedNode(
        id=str(data["id"]),
        kind=str(data["kind"]),
        label=str(data["label"]),
        source=str(data["source"]),
        source_id=str(data["source_id"]),
        properties=properties if isinstance(properties, dict) else {},
    )


def _edge_from_dict(data: dict[str, object]) -> _CachedEdge:
    properties = data["properties"]
    return _CachedEdge(
        type=str(data["type"]),
        from_id=str(data["from_id"]),
        to_id=str(data["to_id"]),
        properties=properties if isinstance(properties, dict) else {},
    )


def get_cached_neighbors(
    node_id: str,
    rel_types: Sequence[str] | None,
    *,
    ttl_seconds: float = DEFAULT_TTL_SECONDS,
    db_path: Path | None = None,
) -> NeighborResult | None:
    """Return the cached neighbor result, or None on a miss or expiry."""
    path = db_path or cache_db_path()
    if not path.exists():
        return None
    conn = _connect(path)
    try:
        row = conn.execute(
            "SELECT payload, cached_at FROM neighbor_cache WHERE node_id = ? AND rel_types = ?",
            (node_id, _rel_key(rel_types)),
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        return None
    payload_text, cached_at = row
    if time.time() - cached_at > ttl_seconds:
        return None
    payload = json.loads(payload_text)
    return _CachedNeighborResult(
        nodes=[_node_from_dict(n) for n in payload["nodes"]],
        edges=[_edge_from_dict(e) for e in payload["edges"]],
        evidence=list(payload["evidence"]),
        warnings=list(payload["warnings"]),
    )


def set_cached_neighbors(
    node_id: str,
    rel_types: Sequence[str] | None,
    result: NeighborResult,
    *,
    db_path: Path | None = None,
    ttl_seconds: float = DEFAULT_TTL_SECONDS,
) -> bool:
    """Cache a successful neighbour lookup. Returns False when not cached.

    Failures are never cached: a ``node_not_found`` for a malformed id, or an
    empty answer during a partial outage, would otherwise be served for a day
    as if it were a graph fact. Expired rows are pruned on every write, so the
    table cannot grow without bound.
    """
    if not result.nodes or any(w in _UNCACHEABLE_WARNINGS for w in result.warnings):
        return False
    path = db_path or cache_db_path()
    payload = json.dumps(
        {
            "nodes": [_node_to_dict(n) for n in result.nodes],
            "edges": [_edge_to_dict(e) for e in result.edges],
            "evidence": list(result.evidence),
            "warnings": list(result.warnings),
        }
    )
    conn = _connect(path)
    try:
        conn.execute(
            "INSERT OR REPLACE INTO neighbor_cache (node_id, rel_types, payload, cached_at) "
            "VALUES (?, ?, ?, ?)",
            (node_id, _rel_key(rel_types), payload, time.time()),
        )
        conn.execute("DELETE FROM neighbor_cache WHERE cached_at < ?", (time.time() - ttl_seconds,))
        conn.commit()
    finally:
        conn.close()
    return True


def clear_neighbor_cache(*, db_path: Path | None = None) -> int:
    """Delete every cached row. Returns how many were removed."""
    path = db_path or cache_db_path()
    if not path.exists():
        return 0
    conn = _connect(path)
    try:
        cursor = conn.execute("DELETE FROM neighbor_cache")
        conn.commit()
        return cursor.rowcount
    finally:
        conn.close()


class CachedSuite:
    """Wraps a suite; caches get_neighbors, passes search_nodes straight through.

    Locate results stay uncached — ranking/ambiguity is more freshness-
    sensitive than a fixed skill list, and search_nodes is already cheap.

    Implements the full ``SuiteTools`` surface so it is substitutable for a
    bare suite anywhere one is expected. ``suite_schema`` is delegated, not
    cached: the schema is small, static per process, and the planner reads it
    on every dispatch — re-reading it from SQLite each turn would be pure
    overhead. (An earlier version of this class omitted it entirely, which
    meant it could not actually be used as a suite: every dispatch reads
    ``suite.suite_schema``, so wiring it in without this raised
    ``AttributeError`` on the first turn.)
    """

    def __init__(
        self,
        suite: SuiteTools,
        *,
        ttl_seconds: float = DEFAULT_TTL_SECONDS,
        db_path: Path | None = None,
    ) -> None:
        self._suite = suite
        self._ttl = ttl_seconds
        self._db_path = db_path
        self.hits = 0
        self.misses = 0

    @property
    def suite_schema(self) -> SuiteSchema:
        return self._suite.suite_schema

    def search_nodes(self, text: str, kind: str | None = None) -> SearchResult:
        return self._suite.search_nodes(text, kind=kind)

    def search_group(self, text: str, group: str) -> SearchResult:
        """Delegated, uncached: narrowing to one occupation group is rare."""
        inner = getattr(self._suite, "search_group", None)
        if inner is None:
            raise AttributeError("search_group")
        result: SearchResult = inner(text, group)
        return result

    def get_neighbors(self, node_id: str, rel_types: list[str] | None = None) -> NeighborResult:
        # The cache is an optimization, never a dependency. A locked, corrupt,
        # or unreadable memory.db must cost latency, not correctness: fall
        # through to the live graph and answer normally. (Before this, a
        # sqlite3.OperationalError propagated out of the turn and the user got
        # an empty result — the worst possible failure for a local cache.)
        try:
            cached = get_cached_neighbors(
                node_id, rel_types, ttl_seconds=self._ttl, db_path=self._db_path
            )
        except (sqlite3.Error, OSError, ValueError):
            cached = None
        else:
            if cached is not None:
                self.hits += 1
                return cached
        self.misses += 1
        result = self._suite.get_neighbors(node_id, rel_types=rel_types)
        try:
            set_cached_neighbors(
                node_id, rel_types, result, db_path=self._db_path, ttl_seconds=self._ttl
            )
        except (sqlite3.Error, OSError, ValueError):
            pass  # a write failure is invisible to the caller by design
        return result

    # Pathfind is delegated rather than implemented, exactly as MeasuredSuite
    # does. These are not part of SuiteTools today, but the Pathfinder restore
    # needs them to survive the cache wrapper — a wrapper that hides a method
    # the inner suite has would turn "not built yet" into "broken".
    def enumerate_paths(
        self,
        from_id: str,
        to_id: str,
        *,
        max_depth: int = 4,
        max_paths: int = 20,
    ) -> object:
        inner = getattr(self._suite, "enumerate_paths", None)
        if inner is None:
            raise AttributeError("enumerate_paths")
        return inner(from_id, to_id, max_depth=max_depth, max_paths=max_paths)

    def score_paths(self, paths: object, policy: object) -> object:
        inner = getattr(self._suite, "score_paths", None)
        if inner is None:
            raise AttributeError("score_paths")
        return inner(paths, policy)

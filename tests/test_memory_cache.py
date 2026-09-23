"""Offline tests for the SQLite get_neighbors cache."""

from __future__ import annotations

from pathlib import Path

import pytest

from talent_angels.memory.cache import (
    CachedSuite,
    get_cached_neighbors,
    set_cached_neighbors,
)
from tests.fakes.taxonomy import FakeEdge, FakeNode, FakeToolResult


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "memory.db"


def _result() -> FakeToolResult:
    node = FakeNode(
        id="esco:skill:1",
        kind="Skill",
        label="computer programming",
        source="esco",
        source_id="src-1",
        properties={"alt_labels": ["coding"]},
    )
    edge = FakeEdge(
        type="HAS_SKILL",
        from_id="esco:occupation:1",
        to_id="esco:skill:1",
        properties={"relation_type": "essential"},
    )
    return FakeToolResult(nodes=[node], edges=[edge], evidence=["esco:neighbors:1"])


def test_miss_on_an_empty_cache(db_path: Path) -> None:
    assert get_cached_neighbors("esco:occupation:1", ["HAS_SKILL"], db_path=db_path) is None


def test_set_then_get_round_trips_nodes_edges_evidence_warnings(db_path: Path) -> None:
    set_cached_neighbors("esco:occupation:1", ["HAS_SKILL"], _result(), db_path=db_path)

    cached = get_cached_neighbors("esco:occupation:1", ["HAS_SKILL"], db_path=db_path)

    assert cached is not None
    assert [n.label for n in cached.nodes] == ["computer programming"]
    assert cached.nodes[0].properties["alt_labels"] == ["coding"]
    assert [e.type for e in cached.edges] == ["HAS_SKILL"]
    assert cached.edges[0].properties["relation_type"] == "essential"
    assert list(cached.evidence) == ["esco:neighbors:1"]
    assert list(cached.warnings) == []


def test_rel_types_order_does_not_matter_for_the_cache_key(db_path: Path) -> None:
    set_cached_neighbors(
        "onet:occupation:1", ["HAS_SKILL", "USES_SOFTWARE"], _result(), db_path=db_path
    )

    cached = get_cached_neighbors(
        "onet:occupation:1", ["USES_SOFTWARE", "HAS_SKILL"], db_path=db_path
    )

    assert cached is not None


def test_different_node_ids_do_not_collide(db_path: Path) -> None:
    set_cached_neighbors("esco:occupation:1", ["HAS_SKILL"], _result(), db_path=db_path)

    assert get_cached_neighbors("esco:occupation:2", ["HAS_SKILL"], db_path=db_path) is None


def test_entries_expire_after_the_ttl(db_path: Path) -> None:
    set_cached_neighbors("esco:occupation:1", None, _result(), db_path=db_path)

    assert (
        get_cached_neighbors("esco:occupation:1", None, ttl_seconds=10_000, db_path=db_path)
        is not None
    )
    assert get_cached_neighbors("esco:occupation:1", None, ttl_seconds=0, db_path=db_path) is None


class _CountingSuite:
    def __init__(self, result: FakeToolResult) -> None:
        self._result = result
        self.search_calls = 0
        self.neighbor_calls = 0

    def search_nodes(self, text: str, kind: str | None = None) -> FakeToolResult:
        self.search_calls += 1
        return self._result

    def get_neighbors(self, node_id: str, rel_types: list[str] | None = None) -> FakeToolResult:
        self.neighbor_calls += 1
        return self._result


def test_cached_suite_only_calls_the_backend_on_a_miss(db_path: Path) -> None:
    backend = _CountingSuite(_result())
    suite = CachedSuite(backend, db_path=db_path)

    suite.get_neighbors("esco:occupation:1", rel_types=["HAS_SKILL"])
    suite.get_neighbors("esco:occupation:1", rel_types=["HAS_SKILL"])

    assert backend.neighbor_calls == 1
    assert suite.hits == 1
    assert suite.misses == 1


def test_cached_suite_does_not_cache_search_nodes(db_path: Path) -> None:
    backend = _CountingSuite(_result())
    suite = CachedSuite(backend, db_path=db_path)

    suite.search_nodes("software developer")
    suite.search_nodes("software developer")

    assert backend.search_calls == 2

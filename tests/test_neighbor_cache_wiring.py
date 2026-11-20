"""The neighbor cache must fire in production, not just in isolation.

`memory/cache.py` shipped fully built and fully tested — and never ran once for
a real user, because no edge remembered to pass it. Worse, `CachedSuite` did
not implement `suite_schema`, so wiring it in as a suite would have raised
`AttributeError` on the first turn, since every dispatch reads that property.

These tests are deliberately written against the *production path* — a
`SuiteRegistry` built through `with_neighbor_cache`, driven by a real
`run_turn` — rather than against `CachedSuite` in a vacuum. A test that only
exercises the wrapper in isolation would pass just as happily while the cache
stayed dead, which is the exact failure being guarded against.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from talent_angels.assistant import run_turn
from talent_angels.contracts import AgentResult
from talent_angels.llm.stub_client import StubLLMClient
from talent_angels.memory.cache import CachedSuite
from talent_angels.suites import SuiteRegistry
from talent_angels.suites.protocol import SuiteTools
from talent_angels.suites.registry import default_suite_registry, with_neighbor_cache
from talent_angels.suites.schema import SuiteSchema
from tests.fakes.suite import SUITE_NAME, FakeSuite, suite_factory
from tests.fakes.taxonomy import FakeToolResult


class CountingSuite(FakeSuite):
    """FakeSuite that records every get_neighbors call, so a cache hit is
    observable as the absence of a call rather than as a cached value that
    happens to look right."""

    def __init__(self) -> None:
        self.neighbor_calls: list[str] = []

    def get_neighbors(self, node_id: str, rel_types: list[str] | None = None) -> FakeToolResult:
        self.neighbor_calls.append(node_id)
        return super().get_neighbors(node_id, rel_types=rel_types)


def _cached_registry(suite: CountingSuite) -> SuiteRegistry:
    return SuiteRegistry(
        {SUITE_NAME: with_neighbor_cache(suite_factory(SUITE_NAME, suite))},
        default=SUITE_NAME,
    )


def test_repeat_connect_turn_hits_the_graph_once() -> None:
    """The headline claim of Sprint 6 step 3: no repeated graph queries.

    Two identical Connect turns, one graph call. Before the wiring this made
    two, every time, forever.
    """
    suite = CountingSuite()
    registry = _cached_registry(suite)

    def turn() -> AgentResult:
        return run_turn(
            registry=registry,
            llm_client=StubLLMClient(),
            question="what skills does a software developer need",
            force_capability="connect",
            persist=False,
        ).result

    first = turn()
    second = turn()

    assert first.nodes, "fixture should resolve a node to connect from"
    assert second.nodes
    assert [node.pref_label for node in second.nodes] == [
        node.pref_label for node in first.nodes
    ], "cached answer must be identical to the fresh one"
    assert len(suite.neighbor_calls) == 1, (
        f"expected the graph to be queried once and served from cache after, "
        f"got {len(suite.neighbor_calls)} calls: {suite.neighbor_calls}"
    )


def test_a_cache_hit_is_reported_in_the_run_log() -> None:
    """A cache that cannot be seen is a cache that can silently stop working.

    Asserted because the run-log is the only place a regression here would
    surface: without a reported hit, a wrapper that stopped matching would be
    indistinguishable from a cold cache.
    """
    suite = CountingSuite()
    registry = _cached_registry(suite)

    cold = run_turn(
        registry=registry,
        llm_client=StubLLMClient(),
        question="what skills does a software developer need",
        force_capability="connect",
        persist=False,
    )
    warm = run_turn(
        registry=registry,
        llm_client=StubLLMClient(),
        question="what skills does a software developer need",
        force_capability="connect",
        persist=False,
    )

    assert cold.record.efficiency.result_cache_hit is False, "a cold cache is not a hit"
    assert warm.record.efficiency.result_cache_hit is True, (
        "the second turn was served from cache and must say so in the run-log"
    )


def test_distinct_occupations_do_not_collide_in_one_cache() -> None:
    """Cache keys are (node_id, rel_types) — a different node is a different row.

    Guards the failure mode where a too-loose key would serve one occupation's
    skills for another's, which would be a correctness bug, not a slow one.
    """
    from tests.fakes.taxonomy import FakeEdge, FakeNode

    other = FakeNode(
        id="fake:occupation:other",
        kind="Occupation",
        label="accountant",
        source="fake",
        source_id="urn:fake:occupation:other",
        properties={},
    )
    other_skill = FakeNode(
        id="fake:skill:ledger",
        kind="Skill",
        label="Ledger",
        source="fake",
        source_id="urn:fake:skill:ledger",
        properties={},
    )
    other_edge = FakeEdge(
        type="HAS_SKILL",
        from_id=other.id,
        to_id=other_skill.id,
        properties={"relation_type": "essential"},
    )

    class TwoOccupationSuite(CountingSuite):
        def search_nodes(self, text: str, kind: str | None = None) -> FakeToolResult:
            if text == "accountant":
                from tests.fakes.taxonomy import FakeCandidate

                return FakeToolResult(
                    candidates=[FakeCandidate(node=other, confidence=0.95, method="exact_pref")],
                    nodes=[other],
                    evidence=["fake:search:exact_pref:accountant"],
                )
            return super().search_nodes(text, kind=kind)

        def get_neighbors(self, node_id: str, rel_types: list[str] | None = None) -> FakeToolResult:
            self.neighbor_calls.append(node_id)
            if node_id == other.id:
                return FakeToolResult(
                    nodes=[other, other_skill], edges=[other_edge], evidence=["fake:n:other"]
                )
            return super().get_neighbors(node_id, rel_types=rel_types)

    suite = TwoOccupationSuite()
    registry = _cached_registry(suite)

    dev = run_turn(
        registry=registry,
        llm_client=StubLLMClient(),
        question="what skills does a software developer need",
        force_capability="connect",
        persist=False,
    ).result
    accountant = run_turn(
        registry=registry,
        llm_client=StubLLMClient(),
        question="what skills does an accountant need",
        force_capability="connect",
        persist=False,
    ).result

    assert "Python" in [n.pref_label for n in dev.nodes]
    assert "Ledger" in [n.pref_label for n in accountant.nodes]
    assert "Python" not in [n.pref_label for n in accountant.nodes], (
        "the accountant answer must not inherit the developer's skills"
    )


def test_cached_suite_satisfies_the_suite_contract() -> None:
    """The bug that kept this unwired: no `suite_schema`, so no real dispatch.

    Every dispatch path reads `suite.suite_schema` (assistant/graph.py,
    assistant/agent_loop.py), so a wrapper without it raises AttributeError on
    the first turn rather than degrading.
    """
    inner = FakeSuite()
    wrapped: SuiteTools = CachedSuite(inner)

    assert isinstance(wrapped.suite_schema, SuiteSchema)
    assert wrapped.suite_schema.skill_rel_types == inner.suite_schema.skill_rel_types
    assert wrapped.search_nodes("software developer").nodes
    assert wrapped.get_neighbors("fake:occupation:dev").nodes


def test_cached_suite_delegates_the_pathfind_slice() -> None:
    """Pathfind is not built yet, but the wrapper must not hide it either.

    A wrapper that swallowed `enumerate_paths` would convert "not implemented"
    into "broken" the moment Pathfinder is restored underneath it.
    """
    wrapped = CachedSuite(FakeSuite())
    assert isinstance(wrapped, CachedSuite)
    found = wrapped.enumerate_paths("fake:occupation:dev", "fake:skill:python")
    assert getattr(found, "paths", None), "delegate to the inner suite, don't reimplement"

    with pytest.raises(AttributeError):
        CachedSuite(object()).enumerate_paths("a", "b")  # type: ignore[arg-type]


def test_default_registry_wires_the_cache_and_the_opt_out_skips_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`default_suite_registry()` is the one place all edges build a registry.

    That is the only reason the cache can no longer be forgotten by an edge.
    Assert both directions so a future refactor cannot quietly drop the wiring
    (or quietly remove the opt-out the bench depends on).
    """
    assert default_suite_registry.__defaults__ is None  # keyword-only default
    # Offline: stand the fake in for the concrete ESCO adapter, which needs the
    # optional ta-taxonomies package (absent in the CI offline job).
    from talent_angels.suites import registry as registry_module

    monkeypatch.setattr(
        registry_module, "_open_default_esco", suite_factory("esco", CountingSuite())
    )
    with default_suite_registry(neighbor_cache=False).open("esco") as runtime:
        assert not isinstance(runtime.suite, CachedSuite)
    with default_suite_registry().open("esco") as runtime:
        assert isinstance(runtime.suite, CachedSuite), "production edges get the cache by default"


def test_a_failing_cache_never_breaks_a_turn() -> None:
    """The cache is an optimization; a broken one must degrade, not fail.

    A user must never get an error because a local SQLite file was locked or
    corrupt. The turn should fall through to the live graph.
    """
    suite = CountingSuite()
    registry = _cached_registry(suite)

    from talent_angels.memory import cache as cache_module

    def boom(*_args: Any, **_kwargs: Any) -> None:
        raise sqlite_error()

    def sqlite_error() -> Exception:
        import sqlite3

        return sqlite3.OperationalError("database is locked")

    original_get = cache_module.get_cached_neighbors
    original_set = cache_module.set_cached_neighbors
    cache_module.get_cached_neighbors = boom  # type: ignore[assignment]
    cache_module.set_cached_neighbors = boom  # type: ignore[assignment]
    try:
        result = run_turn(
            registry=registry,
            llm_client=StubLLMClient(),
            question="what skills does a software developer need",
            force_capability="connect",
            persist=False,
        ).result
    finally:
        cache_module.get_cached_neighbors = original_get  # type: ignore[assignment]
        cache_module.set_cached_neighbors = original_set  # type: ignore[assignment]

    assert result.nodes, "a broken cache must not cost the user their answer"
    assert len(suite.neighbor_calls) == 1


@pytest.mark.parametrize("command", ["bench", "quality"])
def test_the_bench_and_quality_commands_bypass_the_cache(
    command: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`bench` measures what caching saves; a warm cache would invent a saving.

    `quality` scores live answers against golden values that assume current
    graph data. Both must opt out or their output becomes fiction. Asserted by
    observing what the command actually builds, not by reading its source.
    """
    from talent_angels import cli

    asked: list[bool] = []

    def spy(*, neighbor_cache: bool = True) -> SuiteRegistry:
        asked.append(neighbor_cache)
        return suite_registry_stub()

    monkeypatch.setattr(cli, "default_suite_registry", spy)
    monkeypatch.setattr(cli, "_run_bench", lambda registry: 0)
    monkeypatch.setattr(cli, "_run_quality", lambda args, registry: 0)

    assert cli.main([command]) == 0
    assert asked == [False], f"{command!r} must build its registry with neighbor_cache=False"


def test_a_normal_query_command_keeps_the_cache(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The opt-out is scoped to measurement commands only.

    Guards against over-correcting: an everyday `query` is a real user turn
    and should still benefit from the cache.
    """
    from talent_angels import cli

    asked: list[bool] = []

    def spy(*, neighbor_cache: bool = True) -> SuiteRegistry:
        asked.append(neighbor_cache)
        return suite_registry_stub()

    monkeypatch.setattr(cli, "default_suite_registry", spy)
    monkeypatch.setattr(cli, "get_llm_client", lambda: StubLLMClient())
    monkeypatch.setattr(cli, "run_turn", lambda *a, **k: _null_outcome())
    monkeypatch.setattr(cli, "write_query_details", lambda *a, **k: None)

    assert cli.main(["query", "who is a nurse"]) == 0
    capsys.readouterr()
    assert asked == [True], "an everyday query must keep the cache on"


def suite_registry_stub() -> SuiteRegistry:
    from tests.fakes.suite import fake_registry

    return fake_registry()


def _null_outcome() -> Any:
    """A minimal but *real* TurnOutcome, so the CLI's own field access works."""
    from talent_angels.runlog import RunLogRecord

    result = AgentResult(capability="locate", suite=SUITE_NAME, warnings=["not_found"])
    return SimpleNamespace(
        answer="nothing found",
        capability="locate",
        result=result,
        results=(result,),
        plan=[],
        record=RunLogRecord(suite=SUITE_NAME, plan=["locate"], question="who is a nurse"),
    )


def test_failures_are_never_cached_and_expired_rows_are_pruned(memory_home) -> None:
    """A node_not_found for a malformed id used to be served for 24 hours."""
    import sqlite3
    import time

    from talent_angels.memory.cache import get_cached_neighbors, set_cached_neighbors

    missing = FakeToolResult(warnings=["node_not_found"])
    assert set_cached_neighbors("onet:29-1023.00", None, missing) is False
    assert get_cached_neighbors("onet:29-1023.00", None) is None

    from tests.fakes.suite import DEV

    ok = CountingSuite().get_neighbors(DEV.id)
    assert set_cached_neighbors("a", None, ok) is True
    conn = sqlite3.connect(memory_home.db)
    conn.execute("UPDATE neighbor_cache SET cached_at = ?", (time.time() - 10 * 86400,))
    conn.commit()
    conn.close()

    assert set_cached_neighbors("b", None, ok) is True
    conn = sqlite3.connect(memory_home.db)
    rows = {row[0] for row in conn.execute("SELECT node_id FROM neighbor_cache")}
    conn.close()
    assert rows == {"b"}

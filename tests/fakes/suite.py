"""A whole taxonomy suite, faked — the four contract tools over a tiny graph.

Offline tests exercise the MCP edge against this instead of Neo4j: it answers
in the same shapes the concrete suite does (tiered match methods, warnings
instead of guesses, pruning counts instead of discarded rows) so the edge's
mapping and its failure paths are both testable without a database.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any

from talent_angels.suites import SuiteRegistry, SuiteRuntime
from tests.fakes.taxonomy import (
    FakeCandidate,
    FakeEdge,
    FakeNode,
    FakePath,
    FakePruning,
    FakeToolResult,
)

SUITE_NAME = "fake"

DEV = FakeNode(
    id="fake:occupation:dev",
    kind="Occupation",
    label="software developer",
    source="fake",
    source_id="urn:fake:occupation:dev",
    properties={"alt_labels": ["programmer"]},
)
PYTHON = FakeNode(
    id="fake:skill:python",
    kind="Skill",
    label="Python",
    source="fake",
    source_id="urn:fake:skill:python",
    properties={"alt_labels": ["Python programming"]},
)
HAS_PYTHON = FakeEdge(
    type="HAS_SKILL",
    from_id=DEV.id,
    to_id=PYTHON.id,
    properties={"relation_type": "essential"},
)


class FakeSuite:
    """Implements all four contract tools."""

    name = SUITE_NAME

    def search_nodes(self, text: str, kind: str | None = None) -> FakeToolResult:
        if text == "software developer":
            return FakeToolResult(
                candidates=[FakeCandidate(node=DEV, confidence=0.95, method="exact_pref")],
                nodes=[DEV],
                evidence=["fake:search:exact_pref:software developer"],
            )
        if text == "programmer":
            return FakeToolResult(
                candidates=[FakeCandidate(node=DEV, confidence=0.9, method="exact_alt")],
                nodes=[DEV],
                evidence=["fake:search:exact_alt:programmer"],
            )
        return FakeToolResult(warnings=["not_found"], evidence=[f"fake:search:not_found:{text}"])

    def get_neighbors(self, node_id: str, rel_types: list[str] | None = None) -> FakeToolResult:
        if node_id != DEV.id:
            return FakeToolResult(warnings=["node_not_found"])
        return FakeToolResult(
            nodes=[DEV, PYTHON],
            edges=[HAS_PYTHON],
            evidence=[f"fake:neighbors:{node_id}"],
        )

    def enumerate_paths(
        self,
        from_id: str,
        to_id: str,
        *,
        max_depth: int = 4,
        max_paths: int = 20,
    ) -> FakeToolResult:
        if max_depth < 1:
            return FakeToolResult(warnings=["invalid_max_depth"])
        if max_paths < 1:
            return FakeToolResult(warnings=["invalid_max_paths"])
        if {from_id, to_id} != {DEV.id, PYTHON.id}:
            return FakeToolResult(warnings=["endpoint_not_found"])
        return FakeToolResult(
            nodes=[DEV, PYTHON],
            paths=[FakePath(node_ids=[DEV.id, PYTHON.id], edges=[HAS_PYTHON])],
            # A non-zero `pruned` is the only trace a cut route leaves.
            pruning=FakePruning(considered=3, returned=1, pruned=2),
            evidence=[f"fake:paths:{from_id}->{to_id}"],
        )

    def score_paths(self, paths: list[Any], policy: Any) -> FakeToolResult:
        return FakeToolResult(
            paths=list(paths),
            warnings=[
                "score_paths_not_implemented",
                f"policy={policy.name!r} version={policy.version!r}",
            ],
        )


class LocateOnlySuite:
    """A suite that stops at the Locate/Connect slice, as the registry allows."""

    name = "locate_only"

    def search_nodes(self, text: str, kind: str | None = None) -> FakeToolResult:
        return FakeToolResult(warnings=["not_found"])

    def get_neighbors(self, node_id: str, rel_types: list[str] | None = None) -> FakeToolResult:
        return FakeToolResult(warnings=["node_not_found"])


class FailingSuite(FakeSuite):
    """A suite that opens but fails while servicing a tool call."""

    name = "broken"

    def search_nodes(self, text: str, kind: str | None = None) -> FakeToolResult:
        raise RuntimeError("database details must not reach the MCP client")


def suite_factory(name: str, suite: object, *, reachable: bool = True):
    @contextmanager
    def open_runtime():
        yield SuiteRuntime(name=name, suite=suite, health_check=lambda: reachable)  # type: ignore[arg-type]

    return open_runtime


def unopenable_factory(error: Exception):
    """A registered suite whose infrastructure is down."""

    @contextmanager
    def open_runtime():
        raise error
        yield  # pragma: no cover - unreachable, keeps this a generator

    return open_runtime


def fake_registry() -> SuiteRegistry:
    return SuiteRegistry(
        {
            SUITE_NAME: suite_factory(SUITE_NAME, FakeSuite()),
            "locate_only": suite_factory("locate_only", LocateOnlySuite()),
            "broken": suite_factory("broken", FailingSuite()),
            "down": unopenable_factory(ConnectionRefusedError("neo4j is not running")),
        },
        default=SUITE_NAME,
    )

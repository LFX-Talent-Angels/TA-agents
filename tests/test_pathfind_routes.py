"""Pathfind bridge routes (ADR-0008 decision 4)."""

from __future__ import annotations

from dataclasses import dataclass, field

from talent_angels.mcp.protocol import supports_enumerate_paths
from talent_angels.skills.pathfind.routes import find_routes


@dataclass
class _Node:
    id: str
    kind: str
    label: str
    source: str
    source_id: str
    properties: dict[str, object] = field(default_factory=dict)


@dataclass
class _Edge:
    type: str
    from_id: str
    to_id: str
    properties: dict[str, object] = field(default_factory=dict)


@dataclass
class _Path:
    node_ids: list[str]
    edges: list[_Edge]


@dataclass
class _Pruning:
    considered: int
    returned: int
    pruned: int


@dataclass
class _PathsResult:
    nodes: list[_Node] = field(default_factory=list)
    paths: list[_Path] = field(default_factory=list)
    pruning: _Pruning | None = None
    evidence: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


NURSE = _Node(
    id="esco:occupation:nurse", kind="Occupation", label="nurse", source="esco", source_id="n1"
)
ANALYST = _Node(
    id="esco:occupation:analyst",
    kind="Occupation",
    label="data analyst",
    source="esco",
    source_id="a1",
)
STATISTICS = _Node(
    id="esco:skill:statistics", kind="Skill", label="statistics", source="esco", source_id="s1"
)
NURSE_HAS_STATS = _Edge(type="HAS_SKILL", from_id=NURSE.id, to_id=STATISTICS.id)
ANALYST_HAS_STATS = _Edge(type="HAS_SKILL", from_id=ANALYST.id, to_id=STATISTICS.id)


class _FakeSuite:
    """Bridges nurse <-> data analyst through one shared skill."""

    def enumerate_paths(
        self, from_id: str, to_id: str, *, max_depth: int = 4, max_paths: int = 20
    ) -> _PathsResult:
        if {from_id, to_id} != {NURSE.id, ANALYST.id}:
            return _PathsResult(warnings=["endpoint_not_found"])
        return _PathsResult(
            nodes=[NURSE, STATISTICS, ANALYST],
            paths=[
                _Path(
                    node_ids=[NURSE.id, STATISTICS.id, ANALYST.id],
                    edges=[NURSE_HAS_STATS, ANALYST_HAS_STATS],
                )
            ],
            # A non-zero `pruned` is the only trace a cut route leaves.
            pruning=_Pruning(considered=5, returned=1, pruned=4),
            evidence=[f"fake:paths:{from_id}->{to_id}"],
        )


class _NoRouteSuite:
    def enumerate_paths(
        self, from_id: str, to_id: str, *, max_depth: int = 4, max_paths: int = 20
    ) -> _PathsResult:
        return _PathsResult(warnings=["no_path"])


def test_suite_is_recognised_as_path_enumerating() -> None:
    assert supports_enumerate_paths(_FakeSuite()) is True


def test_a_bridge_route_carries_typed_nodes_and_edges_in_path_order() -> None:
    routes = find_routes(_FakeSuite(), "esco", NURSE.id, ANALYST.id)

    assert len(routes.routes) == 1
    route = routes.routes[0]
    assert [node.pref_label for node in route.nodes] == ["nurse", "statistics", "data analyst"]
    assert [edge.type for edge in route.edges] == ["HAS_SKILL", "HAS_SKILL"]
    assert not routes.is_empty


def test_route_nodes_and_edges_are_suite_scoped() -> None:
    route = find_routes(_FakeSuite(), "esco", NURSE.id, ANALYST.id).routes[0]

    assert all(node.suite == "esco" for node in route.nodes)
    assert all(edge.suite == "esco" for edge in route.edges)


def test_no_path_reports_the_suites_own_warning_rather_than_an_empty_list() -> None:
    routes = find_routes(_NoRouteSuite(), "esco", NURSE.id, ANALYST.id)

    assert routes.is_empty
    assert routes.warnings == ("no_path",)


def test_unreachable_endpoint_is_reported_not_silently_dropped() -> None:
    routes = find_routes(_FakeSuite(), "esco", NURSE.id, "esco:occupation:unknown")

    assert routes.is_empty
    assert routes.warnings == ("endpoint_not_found",)


def test_max_depth_and_max_paths_are_forwarded_to_the_suite() -> None:
    seen: dict[str, int] = {}

    class _RecordingSuite(_FakeSuite):
        def enumerate_paths(
            self, from_id: str, to_id: str, *, max_depth: int = 4, max_paths: int = 20
        ) -> _PathsResult:
            seen["max_depth"] = max_depth
            seen["max_paths"] = max_paths
            return super().enumerate_paths(from_id, to_id, max_depth=max_depth, max_paths=max_paths)

    find_routes(_RecordingSuite(), "esco", NURSE.id, ANALYST.id, max_depth=2, max_paths=5)

    assert seen == {"max_depth": 2, "max_paths": 5}

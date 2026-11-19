"""pathfind_one fills AgentResult.routes as distinct chains (PF-1)."""

from __future__ import annotations

from talent_angels.skills.pathfind.dispatch import pathfind_one
from talent_angels.suites.schema import SuiteSchema
from tests.fakes.taxonomy import (
    FakeCandidate,
    FakeEdge,
    FakeNode,
    FakePath,
    FakeToolResult,
)

_SCHEMA = SuiteSchema(
    skill_rel_types=("HAS_SKILL",),
    optional_rel_values=frozenset({"optional"}),
    group_rel_type=None,
    group_node_kinds=frozenset(),
)

NURSE = FakeNode("esco:occupation:nurse", "Occupation", "nurse", source="esco", source_id="n1")
ANALYST = FakeNode(
    "esco:occupation:analyst", "Occupation", "data analyst", source="esco", source_id="a1"
)
STATISTICS = FakeNode("esco:skill:statistics", "Skill", "statistics", source="esco", source_id="s1")
NURSE_HAS_STATS = FakeEdge(type="HAS_SKILL", from_id=NURSE.id, to_id=STATISTICS.id)
ANALYST_HAS_STATS = FakeEdge(type="HAS_SKILL", from_id=ANALYST.id, to_id=STATISTICS.id)


class _Suite:
    suite_schema = _SCHEMA

    def search_nodes(self, text: str, kind: str | None = None) -> FakeToolResult:
        node = NURSE if text == "nurse" else ANALYST
        return FakeToolResult(candidates=[FakeCandidate(node, 0.95, "exact_pref")])

    def get_neighbors(self, node_id: str, rel_types: list[str] | None = None) -> FakeToolResult:
        if node_id == NURSE.id:
            return FakeToolResult(nodes=[NURSE], edges=[])
        return FakeToolResult(
            nodes=[ANALYST, STATISTICS], edges=[ANALYST_HAS_STATS], evidence=["fake:neighbors"]
        )

    def enumerate_paths(
        self, from_id: str, to_id: str, *, max_depth: int = 4, max_paths: int = 20
    ) -> FakeToolResult:
        bridge = FakePath(
            node_ids=[NURSE.id, STATISTICS.id, ANALYST.id],
            edges=[NURSE_HAS_STATS, ANALYST_HAS_STATS],
        )
        result = FakeToolResult(nodes=[NURSE, STATISTICS, ANALYST])
        result.paths = [bridge]  # type: ignore[attr-defined]
        return result


def test_routes_are_filled_as_distinct_node_id_chains() -> None:
    result, _calls = pathfind_one(_Suite(), "esco", "nurse", "data analyst")

    assert len(result.routes) == 1
    route = result.routes[0]
    assert route.node_ids == [NURSE.id, STATISTICS.id, ANALYST.id]
    assert [edge.type for edge in route.edges] == ["HAS_SKILL", "HAS_SKILL"]


def test_every_route_node_id_also_appears_in_the_result_nodes() -> None:
    result, _calls = pathfind_one(_Suite(), "esco", "nurse", "data analyst")

    result_ids = {node.id for node in result.nodes}
    for route in result.routes:
        assert all(node_id in result_ids for node_id in route.node_ids)

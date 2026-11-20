"""pathfind_one -> render_pathfind never counts a route-only skill as missing (#93).

A route can bridge through a skill both the start and target already share
(so it is not in the real gap, per compute_gap) — rendering must not count
it as essential/optional just because the route's own edge touches target.
"""

from __future__ import annotations

from talent_angels.session.pathfind_view import render_pathfind
from talent_angels.skills.pathfind.dispatch import pathfind_one
from talent_angels.suites.schema import SuiteSchema
from tests.fakes.taxonomy import FakeCandidate, FakeEdge, FakeNode, FakePath, FakeToolResult

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
CARE = FakeNode("esco:skill:care", "Skill", "care", source="esco", source_id="c1")
STATISTICS = FakeNode("esco:skill:statistics", "Skill", "statistics", source="esco", source_id="s1")
SQL = FakeNode("esco:skill:sql", "Skill", "sql", source="esco", source_id="q1")

NURSE_HAS_CARE = FakeEdge(
    type="HAS_SKILL", from_id=NURSE.id, to_id=CARE.id, properties={"relation_type": "essential"}
)
ANALYST_HAS_CARE = FakeEdge(
    type="HAS_SKILL", from_id=ANALYST.id, to_id=CARE.id, properties={"relation_type": "essential"}
)
ANALYST_HAS_STATS = FakeEdge(
    type="HAS_SKILL",
    from_id=ANALYST.id,
    to_id=STATISTICS.id,
    properties={"relation_type": "essential"},
)
ANALYST_HAS_SQL = FakeEdge(
    type="HAS_SKILL", from_id=ANALYST.id, to_id=SQL.id, properties={"relation_type": "optional"}
)


class _Suite:
    """nurse and analyst share "care" (not a gap skill); the route to analyst
    bridges through that same shared skill, exactly the shape that used to
    leak a non-missing skill into the rendered gap (#93)."""

    suite_schema = _SCHEMA

    def search_nodes(self, text: str, kind: str | None = None) -> FakeToolResult:
        node = NURSE if text == "nurse" else ANALYST
        return FakeToolResult(candidates=[FakeCandidate(node, 0.95, "exact_pref")])

    def get_neighbors(self, node_id: str, rel_types: list[str] | None = None) -> FakeToolResult:
        if node_id == NURSE.id:
            return FakeToolResult(nodes=[NURSE, CARE], edges=[NURSE_HAS_CARE])
        return FakeToolResult(
            nodes=[ANALYST, CARE, STATISTICS, SQL],
            edges=[ANALYST_HAS_CARE, ANALYST_HAS_STATS, ANALYST_HAS_SQL],
            evidence=["fake:neighbors"],
        )

    def enumerate_paths(
        self, from_id: str, to_id: str, *, max_depth: int = 4, max_paths: int = 20
    ) -> FakeToolResult:
        bridge = FakePath(
            node_ids=[NURSE.id, CARE.id, ANALYST.id],
            edges=[NURSE_HAS_CARE, ANALYST_HAS_CARE],
        )
        result = FakeToolResult(nodes=[NURSE, CARE, ANALYST])
        result.paths = [bridge]  # type: ignore[attr-defined]
        return result


def test_a_shared_route_bridge_skill_is_not_counted_as_missing() -> None:
    result, _calls = pathfind_one(_Suite(), "esco", "nurse", "data analyst")

    # Ground truth: only statistics (essential) and sql (optional) are
    # actually missing — "care" is shared, even though the route bridges
    # through it and touches target with its own HAS_SKILL edge.
    assert result.routes and result.routes[0].node_ids == [NURSE.id, CARE.id, ANALYST.id]

    text = render_pathfind(result)

    assert "**Essential skills needed** (1): statistics" in text
    assert "**Optional skills needed** (1): sql" in text
    assert "care" not in text.split("Route")[0]  # not in the gap section


def test_result_edges_never_include_a_route_only_edge() -> None:
    """dispatch.py must never extend the flat edges list with route edges —
    that was the actual root cause (#93), not just the renderer's logic."""
    result, _calls = pathfind_one(_Suite(), "esco", "nurse", "data analyst")

    # Every edge in the flat list touches target and a genuinely missing
    # node — never the shared "care" bridge skill.
    missing_ids = {STATISTICS.id, SQL.id}
    for edge in result.edges:
        other = edge.target_node_id if edge.source_node_id == ANALYST.id else edge.source_node_id
        assert other in missing_ids, f"unexpected non-gap edge touching target: {edge}"

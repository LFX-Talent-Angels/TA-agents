"""route-rank-v1: deterministic route ordering on a fake suite."""

from __future__ import annotations

from talent_angels.contracts import AgentResult, EdgeRef, NodeRef, Route
from talent_angels.skills.evaluate import rank_routes
from tests.fakes.taxonomy import FakeEdge, FakeToolResult


class DegreeSuite:
    def __init__(self, degrees: dict[str, int]) -> None:
        self.degrees = degrees

    def get_neighbors(self, node_id: str, rel_types: list[str] | None = None) -> FakeToolResult:
        n = self.degrees.get(node_id, 1)
        return FakeToolResult(edges=[FakeEdge("X", node_id, f"o{i}") for i in range(n)])


def _ref(i: str) -> NodeRef:
    return NodeRef(id=i, suite="s", source="s", source_id=i, kind="Skill", pref_label=i)


def _route(*ids: str, rel: str | None = "essential") -> Route:
    props: dict[str, object] = {"relation_type": rel} if rel else {}
    edges = [
        EdgeRef(type="HAS_SKILL", suite="s", source_node_id=a, target_node_id=b, properties=props)
        for a, b in zip(ids, ids[1:], strict=False)
    ]
    return Route(node_ids=list(ids), edges=edges)


def _result(*routes: Route) -> AgentResult:
    ids = {i for r in routes for i in r.node_ids}
    return AgentResult(
        capability="pathfind", suite="s", nodes=[_ref(i) for i in sorted(ids)], routes=list(routes)
    )


def test_essential_route_beats_optional() -> None:
    res = _result(_route("a", "x", "b", rel="optional"), _route("a", "y", "b"))
    ranked = rank_routes(DegreeSuite({}), res)
    assert ranked[0].route.node_ids == ["a", "y", "b"]
    assert ranked[0].score > ranked[1].score
    assert ranked[0].reasons[0] == "2/2 links essential"


def test_specific_bridge_beats_generic() -> None:
    res = _result(_route("a", "generic", "b"), _route("a", "niche", "b"))
    ranked = rank_routes(DegreeSuite({"generic": 500, "niche": 3}), res)
    assert ranked[0].route.node_ids[1] == "niche"
    assert "bridge used by 3 items" in ranked[0].reasons


def test_ties_are_stable_and_repeatable() -> None:
    res = _result(_route("a", "z", "b"), _route("a", "m", "b"))
    first = rank_routes(DegreeSuite({}), res)
    again = rank_routes(DegreeSuite({}), res)
    assert [r.route.node_ids for r in first] == [["a", "m", "b"], ["a", "z", "b"]]
    assert first == again

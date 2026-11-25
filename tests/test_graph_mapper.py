"""Graph-Mapper core: deterministic neighborhood walk, offline, on a small fake graph."""

from __future__ import annotations

import pytest

from talent_angels.skills.connect.neighbourhood import map_from_text, map_neighbourhood, map_route
from tests.fakes.taxonomy import FakeCandidate, FakeEdge, FakeNode, FakeToolResult


class GraphSuite:
    """Serves one-hop neighbors from a fixed edge list, like a Connect suite."""

    def __init__(self, nodes: list[FakeNode], edges: list[FakeEdge]) -> None:
        self.nodes = {n.id: n for n in nodes}
        self.edges = edges
        self.calls: list[str] = []
        self.searches: list[str] = []

    def search_nodes(self, text: str, kind: str | None = None) -> FakeToolResult:
        self.searches.append(text)
        hits = [n for n in self.nodes.values() if text.casefold() in n.label.casefold()]
        return FakeToolResult(
            candidates=[FakeCandidate(node=n, confidence=0.9, method="contains") for n in hits]
        )

    def get_neighbors(self, node_id: str, rel_types: list[str] | None = None) -> FakeToolResult:
        self.calls.append(node_id)
        incident = [e for e in self.edges if node_id in (e.from_id, e.to_id)]
        ids = {node_id} | {e.from_id for e in incident} | {e.to_id for e in incident}
        return FakeToolResult(
            nodes=[self.nodes[i] for i in sorted(ids) if i in self.nodes],
            edges=incident,
        )


def _node(node_id: str, kind: str = "Skill", label: str | None = None) -> FakeNode:
    return FakeNode(
        id=node_id,
        kind=kind,
        label=label or node_id,
        source="esco",
        source_id=f"https://example.test/{node_id}",
        properties={},
    )


def _edge(kind: str, src: str, dst: str, **props: object) -> FakeEdge:
    return FakeEdge(type=kind, from_id=src, to_id=dst, properties=dict(props))


def _levels(result_dict: dict[str, object]) -> dict[str, int]:
    nodes = result_dict["nodes"]
    assert isinstance(nodes, list)
    return {n["label"]: n["level"] for n in nodes}


def test_related_skill_beats_a_job_and_essential_beats_optional() -> None:
    centre = _node("job", kind="Occupation", label="job")
    suite = GraphSuite(
        [centre, _node("a", label="alpha"), _node("b", label="beta"), _node("c", label="gamma")],
        [
            _edge("HAS_SKILL", "job", "c", relation_type="optional"),
            _edge("HAS_SKILL", "job", "b", relation_type="essential"),
            _edge("HAS_SKILL", "job", "a", relation_type="essential"),
        ],
    )

    result = map_neighbourhood(suite, "esco", "job", depth=1, breadth=2)

    assert set(_levels(result.to_dict())) == {"job", "alpha", "beta"}
    assert result.pruned[0].considered == 3
    assert result.pruned[0].kept == 2


def test_skill_links_come_before_job_links_for_a_skill() -> None:
    skill = _node("py", label="python")
    suite = GraphSuite(
        [
            skill,
            _node("rel", label="scripting"),
            _node("job", kind="Occupation", label="software developer"),
            _node("job2", kind="Occupation", label="web developer"),
        ],
        [
            _edge("HAS_SKILL", "job", "py", relation_type="essential"),
            _edge("HAS_SKILL", "job2", "py", relation_type="essential"),
            _edge("RELATED_TO", "py", "rel", relation_type="optional"),
        ],
    )

    data = map_neighbourhood(suite, "esco", "py", depth=1, breadth=2).to_dict()

    # Breadth 2: the related skill first, then the job with the first label.
    assert set(_levels(data)) == {"python", "scripting", "software developer"}


def test_relation_labels_follow_edge_direction() -> None:
    skill = _node("py", label="python")
    suite = GraphSuite(
        [
            skill,
            _node("parent", label="computer programming"),
            _node("narrow", label="Ruby"),
            _node("rel", label="scripting"),
            _node("job", kind="Occupation", label="software developer"),
        ],
        [
            _edge("BROADER_THAN", "py", "parent"),
            _edge("BROADER_THAN", "narrow", "py"),
            _edge("RELATED_TO", "py", "rel", relation_type="optional"),
            _edge("HAS_SKILL", "job", "py", relation_type="essential"),
        ],
    )

    data = map_neighbourhood(suite, "esco", "py", depth=1, breadth=10).to_dict()

    relations = {(e["relation"], e["from"], e["to"]) for e in data["edges"]}
    assert ("parent", "py", "parent") in relations
    assert ("narrower", "py", "narrow") in relations
    assert ("related", "py", "rel") in relations
    assert ("needed_by", "py", "job") in relations


def test_hub_is_shown_but_not_expanded() -> None:
    suite = GraphSuite(
        [
            _node("py", label="python"),
            _node("group", kind="SkillGroup", label="software development"),
            _node("sibling", label="maltego"),
        ],
        [
            _edge("BROADER_THAN", "py", "group"),
            _edge("BROADER_THAN", "group", "sibling"),
        ],
    )

    data = map_neighbourhood(suite, "esco", "py", depth=2, breadth=5).to_dict()

    assert "software development" in _levels(data)
    assert "maltego" not in _levels(data)
    assert "group" not in suite.calls[1:]


def test_kinds_filter_keeps_only_listed_kinds() -> None:
    suite = GraphSuite(
        [
            _node("py", label="python"),
            _node("skill", label="scripting"),
            _node("job", kind="Occupation", label="software developer"),
        ],
        [
            _edge("RELATED_TO", "py", "skill", relation_type="optional"),
            _edge("HAS_SKILL", "job", "py", relation_type="essential"),
        ],
    )

    data = map_neighbourhood(suite, "esco", "py", depth=1, breadth=5, kinds={"Skill"}).to_dict()

    assert set(_levels(data)) == {"python", "scripting"}


def test_node_reached_twice_is_mapped_once_and_both_links_are_kept() -> None:
    suite = GraphSuite(
        [
            _node("start", label="start"),
            _node("a", label="alpha"),
            _node("b", label="beta"),
            _node("shared", label="shared"),
        ],
        [
            _edge("RELATED_TO", "start", "a", relation_type="optional"),
            _edge("RELATED_TO", "start", "b", relation_type="optional"),
            _edge("RELATED_TO", "a", "shared", relation_type="optional"),
            _edge("RELATED_TO", "b", "shared", relation_type="optional"),
        ],
    )

    data = map_neighbourhood(suite, "esco", "start", depth=2, breadth=5).to_dict()

    labels = [n["label"] for n in data["nodes"]]  # type: ignore[index]
    assert labels.count("shared") == 1
    links = {(e["from"], e["to"]) for e in data["edges"]}  # type: ignore[index]
    assert ("a", "shared") in links
    assert ("b", "shared") in links


def test_same_input_gives_identical_output() -> None:
    suite = GraphSuite(
        [_node("start", label="start")]
        + [_node(f"n{i}", label=f"skill {chr(97 + i)}") for i in range(6)],
        [_edge("RELATED_TO", "start", f"n{i}", relation_type="optional") for i in range(6)],
    )

    first = map_neighbourhood(suite, "esco", "start", depth=1, breadth=3).to_dict()
    second = map_neighbourhood(suite, "esco", "start", depth=1, breadth=3).to_dict()

    assert first == second
    # Ties on every rule fall back to label order, so "skill a" is always first.
    assert [n["label"] for n in first["nodes"]][1:] == ["skill a", "skill b", "skill c"]  # type: ignore[index]


@pytest.mark.parametrize(("depth", "breadth"), [(0, 2), (5, 2), (1, 0), (1, 11)])
def test_depth_and_breadth_are_capped(depth: int, breadth: int) -> None:
    suite = GraphSuite([_node("start")], [])
    with pytest.raises(ValueError):
        map_neighbourhood(suite, "esco", "start", depth=depth, breadth=breadth)


def test_max_nodes_stops_the_walk_and_says_so() -> None:
    suite = GraphSuite(
        [_node("start", label="start")] + [_node(f"n{i}", label=f"n{i}") for i in range(5)],
        [_edge("RELATED_TO", "start", f"n{i}", relation_type="optional") for i in range(5)],
    )

    data = map_neighbourhood(suite, "esco", "start", depth=1, breadth=5, max_nodes=3).to_dict()

    assert len(data["nodes"]) == 3  # type: ignore[arg-type]
    assert "max_nodes:3" in data["warnings"]  # type: ignore[operator]


def test_unknown_start_returns_an_empty_map_with_a_warning() -> None:
    suite = GraphSuite([_node("other")], [])

    data = map_neighbourhood(suite, "esco", "missing", depth=1, breadth=2).to_dict()

    assert data["nodes"] == []
    assert data["warnings"] == ["start_not_found"]


def _text_suite() -> GraphSuite:
    return GraphSuite(
        [
            _node("esco:py", label="Python"),
            _node("esco:py2", label="Python snake"),
            _node("esco:rel", label="scripting"),
        ],
        [_edge("RELATED_TO", "esco:py", "esco:rel", relation_type="optional")],
    )


def test_exact_label_match_picks_the_start() -> None:
    data = map_from_text(_text_suite(), "esco", "python", depth=1, breadth=3).to_dict()

    assert data["start"] == {"id": "esco:py", "label": "Python"}
    assert set(_levels(data)) == {"Python", "scripting"}


def test_ambiguous_name_returns_candidates_and_no_nodes() -> None:
    data = map_from_text(_text_suite(), "esco", "pyth", depth=1, breadth=3).to_dict()

    assert data["nodes"] == []
    assert data["warnings"] == ["ambiguous_start"]
    assert [c["id"] for c in data["candidates"]] == ["esco:py", "esco:py2"]  # type: ignore[index]


def test_unknown_name_warns_with_the_text() -> None:
    data = map_from_text(_text_suite(), "esco", "zzz", depth=1, breadth=3).to_dict()

    assert data["nodes"] == []
    assert data["warnings"] == ["start_not_found:zzz"]


def test_node_id_skips_the_search() -> None:
    suite = _text_suite()

    data = map_from_text(suite, "esco", "esco:py", depth=1, breadth=3).to_dict()

    assert suite.searches == []
    assert data["start"] == {"id": "esco:py", "label": "Python"}


def _route_suite() -> GraphSuite:
    return GraphSuite(
        [_node(i, label=i) for i in ("a", "b", "c", "shared", "only_a")],
        [
            _edge("RELATED_TO", "a", "b", relation_type="optional"),
            _edge("RELATED_TO", "b", "c", relation_type="optional"),
            _edge("RELATED_TO", "a", "shared", relation_type="optional"),
            _edge("RELATED_TO", "c", "shared", relation_type="optional"),
            _edge("RELATED_TO", "a", "only_a", relation_type="optional"),
        ],
    )


def test_route_mode_flags_stops_and_lists_a_shared_neighbour_once() -> None:
    data = map_route(_route_suite(), "esco", ["a", "b", "c"], depth=1, breadth=5).to_dict()

    nodes = {n["id"]: n for n in data["nodes"]}  # type: ignore[attr-defined]
    assert data["route"] == ["a", "b", "c"]
    assert [nodes[i]["stop"] for i in "abc"] == [0, 1, 2]
    assert all(nodes[i]["level"] == 0 and nodes[i]["on_route"] for i in "abc")
    assert nodes["shared"]["on_route"] is False and "stop" not in nodes["shared"]
    assert [n["id"] for n in data["nodes"]].count("shared") == 1  # type: ignore[attr-defined]
    keys = [(e["from"], e["to"], e["relation"]) for e in data["edges"]]  # type: ignore[attr-defined]
    assert len(keys) == len(set(keys))


def test_route_mode_applies_max_nodes_to_the_whole_map() -> None:
    data = map_route(
        _route_suite(), "esco", ["a", "b", "c"], depth=1, breadth=5, max_nodes=4
    ).to_dict()

    assert len(data["nodes"]) == 4  # type: ignore[arg-type]
    assert "max_nodes:4" in data["warnings"]  # type: ignore[operator]
    ids = {n["id"] for n in data["nodes"]}  # type: ignore[attr-defined]
    assert all(e["from"] in ids and e["to"] in ids for e in data["edges"])  # type: ignore[attr-defined]


def test_route_mode_is_deterministic() -> None:
    first = map_route(_route_suite(), "esco", ["a", "b", "c"]).to_dict()
    assert first == map_route(_route_suite(), "esco", ["a", "b", "c"]).to_dict()


# --- Regressions from the validation pass (2026-11-24) ----------------------


def test_unknown_id_is_a_miss_not_a_search() -> None:
    suite = GraphSuite([_node("esco:skill:real", label="python")], [])

    data = map_from_text(suite, "esco", "esco:skill:missing", depth=1, breadth=2).to_dict()

    assert data["nodes"] == []
    assert data["warnings"] == ["start_not_found:esco:skill:missing"]
    assert suite.searches == []


def test_a_link_walked_from_both_sides_is_listed_once() -> None:
    suite = GraphSuite(
        [_node("py", label="python"), _node("cp", label="computer programming")],
        [_edge("BROADER_THAN", "py", "cp")],
    )

    data = map_neighbourhood(suite, "esco", "py", depth=2, breadth=4).to_dict()

    assert len(data["edges"]) == 1  # type: ignore[arg-type]


def test_pruned_counts_what_was_really_kept_when_max_nodes_stops_the_walk() -> None:
    suite = GraphSuite(
        [_node("start", label="start")] + [_node(f"n{i}", label=f"n{i}") for i in range(5)],
        [_edge("RELATED_TO", "start", f"n{i}", relation_type="optional") for i in range(5)],
    )

    result = map_neighbourhood(suite, "esco", "start", depth=1, breadth=5, max_nodes=3)

    assert len(result.nodes) == 3
    assert result.pruned[-1].kept == 2


def test_kinds_filter_ignores_case() -> None:
    suite = GraphSuite(
        [_node("py", label="python"), _node("skill", label="scripting")],
        [_edge("RELATED_TO", "py", "skill", relation_type="optional")],
    )

    data = map_neighbourhood(suite, "esco", "py", depth=1, breadth=5, kinds={"skill"}).to_dict()

    assert set(_levels(data)) == {"python", "scripting"}


def test_link_to_a_neighbour_a_later_sibling_added_is_kept() -> None:
    # a is expanded before b and spends its breadth on a1, a2; b then adds c.
    suite = GraphSuite(
        [_node(i, label=i) for i in ("start", "a", "b", "a1", "a2", "c")],
        [
            _edge("RELATED_TO", "start", "a", relation_type="essential"),
            _edge("RELATED_TO", "start", "b", relation_type="optional"),
            _edge("RELATED_TO", "a", "a1", relation_type="essential"),
            _edge("RELATED_TO", "a", "a2", relation_type="essential"),
            _edge("RELATED_TO", "a", "c", relation_type="optional"),
            _edge("RELATED_TO", "b", "c", relation_type="essential"),
        ],
    )

    data = map_neighbourhood(suite, "esco", "start", depth=2, breadth=2).to_dict()

    links = {frozenset((e["from"], e["to"])) for e in data["edges"]}  # type: ignore[union-attr]
    assert _levels(data)["c"] == 2
    assert frozenset(("b", "c")) in links
    assert frozenset(("a", "c")) in links


def _hub_and_niche_suite() -> GraphSuite:
    ids = ["generic", "niche", "mid"]
    nodes = [_node("job", kind="Occupation", label="job")] + [_node(i, label=i) for i in ids]
    nodes += [_node(f"x{k}", label=f"x{k}") for k in range(6)]
    edges = [_edge("HAS_SKILL", "job", i, relation_type="essential") for i in ids]
    edges += [_edge("HAS_SKILL", f"x{k}", "generic") for k in range(6)]  # generic: degree 7
    edges += [_edge("HAS_SKILL", "x0", "mid")]  # mid: degree 2
    return GraphSuite(nodes, edges)


def test_map_rank_prefers_a_specific_neighbour_to_a_generic_one() -> None:
    # Label order alone would pick "generic" first.
    result = map_neighbourhood(_hub_and_niche_suite(), "esco", "job", depth=1, breadth=2)

    assert set(_levels(result.to_dict())) == {"job", "niche", "mid"}
    assert result.to_dict()["policy"] == "map-rank-v1"


def test_rank_budget_warns_and_stays_deterministic(monkeypatch: pytest.MonkeyPatch) -> None:
    from talent_angels.skills.connect import neighbourhood
    from talent_angels.skills.evaluate.policy import MapRankPolicy

    monkeypatch.setattr(neighbourhood, "MAP_RANK_V1", MapRankPolicy(degree_budget=1))

    first = map_neighbourhood(_hub_and_niche_suite(), "esco", "job", depth=1, breadth=2)
    again = map_neighbourhood(_hub_and_niche_suite(), "esco", "job", depth=1, breadth=2)

    assert "rank_budget:1" in first.warnings
    assert first.to_dict() == again.to_dict()

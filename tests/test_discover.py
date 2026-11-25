"""Discovery map on a small fake graph: links, connectors, routes, orphans, clusters, guesses."""

from __future__ import annotations

import json
import random

import pytest

from talent_angels.skills.connect.discover import MAX_ROUTE_PAIRS, discover
from talent_angels.skills.connect.discover_io import MAX_ITEMS, Item, build_items
from talent_angels.skills.connect.discover_report import render_html, write_outputs
from talent_angels.suites import SuiteRegistry, UnknownSuiteError
from tests.fakes.suite import suite_factory
from tests.fakes.taxonomy import FakeEdge, FakePath, FakeToolResult
from tests.test_graph_mapper import GraphSuite, _edge, _node


class PathSuite(GraphSuite):
    """GraphSuite plus enumerate_paths: every simple path up to ``max_depth`` edges."""

    def enumerate_paths(
        self, from_id: str, to_id: str, *, max_depth: int = 4, max_paths: int = 20
    ) -> FakeToolResult:
        found: list[list[FakeEdge]] = []

        def walk(at: str, trail: list[FakeEdge], seen: set[str]) -> None:
            if len(trail) > max_depth or len(found) >= max_paths:
                return
            if at == to_id and trail:
                found.append(list(trail))
                return
            for e in self.edges:
                other = e.to_id if e.from_id == at else e.from_id if e.to_id == at else None
                if other and other not in seen:
                    walk(other, [*trail, e], seen | {other})

        walk(from_id, [], {from_id})
        ids = {from_id, to_id} | {i for p in found for e in p for i in (e.from_id, e.to_id)}
        paths = []
        for trail in found:
            order = [from_id]
            for e in trail:
                order.append(e.to_id if e.from_id == order[-1] else e.from_id)
            paths.append(FakePath(node_ids=order, edges=trail))
        return FakeToolResult(
            nodes=[self.nodes[i] for i in sorted(ids) if i in self.nodes], paths=paths
        )


def _graph() -> PathSuite:
    nodes = [
        _node("occ:web", "Occupation", "web developer"),
        _node("occ:da", "Occupation", "data analyst"),
        _node("s:js", label="JavaScript"),
        _node("s:py", label="Python"),
        _node("s:sql", label="SQL"),
        _node("s:java", label="Java"),
        _node("s:script", label="JavaScript framework"),
        _node("s:agile", label="agile"),
        _node("s:pm", label="project management"),
        _node("s:ux", label="UX design"),
        _node("s:team", label="teamwork"),
        _node("s:scripting", label="scripting"),
        _node("g:prog", "SkillGroup", "programming"),
        _node("x:a", label="mid a"),
        _node("x:b", label="mid b"),
        _node("y:1", label="far 1"),
        _node("y:2", label="far 2"),
        _node("y:3", label="far 3"),
    ]
    edges = [
        _edge("HAS_SKILL", "occ:web", "s:js", relation_type="essential"),  # direct link
        _edge("HAS_SKILL", "occ:da", "s:sql", relation_type="essential"),  # direct link
        _edge("RELATED_TO", "s:js", "s:scripting", relation_type="optional"),  # connector
        _edge("RELATED_TO", "s:py", "s:scripting", relation_type="optional"),
        _edge("BROADER_THAN", "g:prog", "s:js"),  # broad connector
        _edge("BROADER_THAN", "g:prog", "s:py"),
        _edge("BROADER_THAN", "g:prog", "s:sql"),
        _edge("RELATED_TO", "s:agile", "x:a"),  # agile - a - b - pm: 3 edges
        _edge("RELATED_TO", "x:a", "x:b"),
        _edge("RELATED_TO", "x:b", "s:pm"),
        _edge("RELATED_TO", "s:ux", "y:1"),  # ux - 1 - 2 - 3 - web: 4 edges
        _edge("RELATED_TO", "y:1", "y:2"),
        _edge("RELATED_TO", "y:2", "y:3"),
        _edge("RELATED_TO", "y:3", "occ:web"),
    ]
    return PathSuite(nodes, edges)


def _registry(graph: GraphSuite | None = None) -> SuiteRegistry:
    return SuiteRegistry({"test": suite_factory("test", graph or _graph())}, default="test")


def _items(*texts: str, **guesses: tuple[str, ...]) -> list[Item]:
    return [Item(t, None, guesses.get(t, ())) for t in texts]


def _run(items: list[Item], **kw: object) -> dict:
    return discover(_registry(), items, depth=1, breadth=4, **kw)  # type: ignore[arg-type]


BASE = (
    "web developer", "JavaScript", "Python", "SQL", "data analyst", "teamwork",
    "agile", "project management", "UX design",
)  # fmt: skip


def test_a_direct_edge_between_two_items_is_a_link() -> None:
    data = _run(_items(*BASE))
    pairs = {(x["a"], x["b"]): x for x in data["links"]}
    link = pairs[("JavaScript", "web developer")]
    assert (link["relation"], link["graph_edge"], link["suite"]) == (
        "needed_by",
        "HAS_SKILL",
        "test",
    )
    assert ("data analyst", "SQL") in pairs


def test_a_link_is_found_even_when_breadth_cut_it_from_the_walk() -> None:
    # Breadth 1 keeps only the best neighbour of web developer; the link to UX stays visible
    # because items are checked against each other's full neighbour list.
    graph = _graph()
    graph.edges.append(_edge("RELATED_TO", "occ:web", "s:ux"))
    data = discover(
        _registry(graph), _items("web developer", "UX design", "JavaScript"), depth=1, breadth=1
    )
    assert ("UX design", "web developer") in {(x["a"], x["b"]) for x in data["links"]}


def test_connectors_are_ranked_flagged_and_exclude_items() -> None:
    data = _run(_items(*BASE))
    labels = [(c["label"], c["item_count"], c["broad"]) for c in data["connectors"]]
    # Specific connectors come before broad hubs, whatever their item count.
    assert labels == [("scripting", 2, False), ("programming", 3, True)]
    scripting = data["connectors"][0]
    assert [i["item"] for i in scripting["items"]] == ["JavaScript", "Python"]
    assert scripting["cite"].startswith("https://") and scripting["suite"] == "test"
    assert not any(c["id"] in {"occ:web", "s:js"} for c in data["connectors"])


def test_connectors_tie_break_on_label_then_id() -> None:
    graph = _graph()
    for node_id, label in (("z:2", "same"), ("z:1", "same")):
        graph.nodes[node_id] = _node(node_id, label=label)
        graph.edges += [
            _edge("RELATED_TO", "s:sql", node_id),
            _edge("RELATED_TO", "s:team", node_id),
        ]
    data = discover(_registry(graph), _items("SQL", "teamwork"), depth=1, breadth=4)
    assert [c["id"] for c in data["connectors"]] == ["z:1", "z:2"]


def test_routes_cover_unconnected_pairs_and_say_how_many_edges() -> None:
    data = _run(_items(*BASE))
    routes = {(r["a"], r["b"]): r for r in data["routes"]}
    near = routes[("agile", "project management")]
    assert near["edges"] == 3 and near["policy"] == "route-rank-v1"
    assert [s["label"] for s in near["stops"]] == ["agile", "mid a", "mid b", "project management"]
    assert ("JavaScript", "Python") not in routes  # they share a connector already


def test_the_route_cap_is_reported_and_guess_pairs_are_always_routed() -> None:
    graph = _graph()
    many = [f"extra {n:02d}" for n in range(12)]
    for t in many:
        graph.nodes[f"e:{t}"] = _node(f"e:{t}", label=t)
    items = _items(*many, "UX design", "web developer", **{"UX design": ("web developer",)})
    data = discover(_registry(graph), items, depth=1, breadth=4)
    # 14 items = 91 pairs, one linked-by-guess pair routed anyway, the other 90 capped at 30.
    assert data["summary"]["route_pairs_skipped"] == 90 - MAX_ROUTE_PAIRS
    assert ("UX design", "web developer") in {(r["a"], r["b"]) for r in data["routes"]}
    assert any(not o["routes_fully_checked"] for o in data["orphans"])


def test_orphans_are_resolved_items_connected_to_nothing_and_unfinished_checks_are_marked() -> None:
    data = _run(_items(*BASE))
    assert data["orphans"] == [{"item": "teamwork", "routes_fully_checked": True}]


def test_clusters_are_ordered_by_size_then_first_item() -> None:
    data = _run(_items(*BASE))
    # The 4-edge route UX design - web developer pulls UX design in; agile - project
    # management is a second, smaller group; teamwork is an orphan, in neither.
    assert [(c["name"], c["size"]) for c in data["clusters"]] == [
        ("Cluster 1", 6),
        ("Cluster 2", 2),
    ]
    assert data["clusters"][0]["items"] == [
        "data analyst", "JavaScript", "Python", "SQL", "UX design", "web developer",
    ]  # fmt: skip
    assert data["clusters"][0]["connectors"][:2] == ["programming", "scripting"]
    two = discover(
        _registry(),
        _items("agile", "project management", "data analyst", "SQL"),
        depth=1,
        breadth=4,
    )
    assert [(c["name"], c["items"]) for c in two["clusters"]] == [
        ("Cluster 1", ["agile", "project management"]),
        ("Cluster 2", ["data analyst", "SQL"]),
    ]


def test_every_guess_gets_a_verdict() -> None:
    data = _run(
        _items(
            *BASE,
            "jav",
            **{
                "web developer": ("JavaScript",),  # confirmed: direct link
                "Python": ("SQL", "Rust"),  # indirect via the broad hub; Rust is not an item
                "project management": ("agile",),  # indirect: a 3-edge route
                "UX design": ("web developer",),  # 4-edge route: not_in_graph
                "teamwork": ("SQL",),  # no route at all: not_in_graph
                "jav": ("Python",),  # jav is unclear: unresolved
            },
        )
    )
    got = {(g["item"], g["related_to"]): (g["verdict"], g["detail"]) for g in data["guesses"]}
    assert got[("web developer", "JavaScript")][0] == "confirmed"
    assert got[("Python", "SQL")] == ("indirect", "both reach: programming")
    assert got[("Python", "Rust")][0] == "unresolved"
    assert "not in the item list" in got[("Python", "Rust")][1]
    assert got[("project management", "agile")] == ("indirect", "route of 3 edges")
    assert got[("UX design", "web developer")] == ("not_in_graph", "closest route has 4 edges")
    assert got[("teamwork", "SQL")] == ("not_in_graph", "no route found")
    assert got[("jav", "Python")][0] == "unresolved"
    assert "unclear in test" in got[("jav", "Python")][1]
    assert data["summary"]["guesses"] == {
        "confirmed": 1, "indirect": 2, "not_in_graph": 2, "unresolved": 2,
    }  # fmt: skip


def test_an_unclear_item_is_never_picked_and_is_listed_with_candidates() -> None:
    data = _run(_items("jav", "blorptech", "SQL"))
    unclear = data["gaps"]["test"]["unclear"]
    assert [(u["item"], [c["label"] for c in u["candidates"]]) for u in unclear] == [
        ("jav", ["Java", "JavaScript", "JavaScript framework"])
    ]
    assert unclear[0]["candidates"][0]["confidence"] == 0.9
    assert unclear[0]["candidates"][0]["method"] == "contains"
    assert data["gaps"]["test"]["not_found"] == ["blorptech"]
    status = {i["item"]: (i["status"], i["not_in_any_taxonomy"]) for i in data["items"]}
    assert status == {
        "blorptech": ("not_found", True),
        "jav": ("unclear", False),
        "SQL": ("resolved", False),
    }
    assert not any(r["item"] == "jav" and r["status"] == "resolved" for r in data["resolved"])


def test_an_item_with_only_meaning_guesses_is_in_no_taxonomy_but_stays_a_gap() -> None:
    from tests.fakes.taxonomy import FakeCandidate

    class Meaning(PathSuite):
        def search_nodes(self, text: str, kind: str | None = None) -> FakeToolResult:
            if text != "blorp":
                return super().search_nodes(text, kind)
            hits = [FakeCandidate(self.nodes[i], 0.75, "hybrid_rrf") for i in ("s:py", "s:sql")]
            return FakeToolResult(candidates=hits, warnings=["ambiguous"])

    graph = _graph()
    meaning = Meaning(list(graph.nodes.values()), graph.edges)
    data = discover(_registry(meaning), _items("blorp", "jav"), depth=1, breadth=2)
    flags = {i["item"]: (i["status"], i["not_in_any_taxonomy"]) for i in data["items"]}
    assert flags == {"blorp": ("unclear", True), "jav": ("unclear", False)}
    assert data["summary"]["resolved_items"] == 0


def test_a_failing_suite_is_reported_and_the_others_still_answer() -> None:
    from tests.fakes.suite import FailingSuite

    reg = SuiteRegistry(
        {
            "test": suite_factory("test", _graph()),
            "broken": suite_factory("broken", FailingSuite()),
        },
        default="test",
    )
    data = discover(reg, _items("SQL"), depth=1, breadth=2)
    assert data["suites"] == {"test": {"status": "ok"}, "broken": {"status": "error:RuntimeError"}}
    assert data["summary"]["resolved_items"] == 1


def test_unknown_suite_and_bad_limits_raise() -> None:
    with pytest.raises(UnknownSuiteError):
        discover(_registry(), _items("SQL"), suites=["nope"])
    with pytest.raises(ValueError):
        discover(_registry(), _items("SQL"), depth=9)
    with pytest.raises(ValueError):
        discover(_registry(), _items("SQL"), max_nodes=0)


def test_max_nodes_is_a_hard_cap_on_the_merged_map() -> None:
    data = discover(_registry(), _items(*BASE), depth=2, breadth=4, max_nodes=20)
    assert len(data["nodes"]) <= 20
    assert any(":max_nodes:" in w for w in data["warnings"])


# ---- determinism -------------------------------------------------------------------------


def _dump(data: dict) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2)


def test_row_order_does_not_change_the_bytes() -> None:
    rows = list(BASE) + ["jav", "blorptech"]
    guesses = {"web developer": ("JavaScript",), "project management": ("agile",)}
    first = _dump(_run(_items(*rows, **guesses)))
    for seed in range(5):
        shuffled = rows[:]
        random.Random(seed).shuffle(shuffled)
        assert _dump(_run(_items(*shuffled, **guesses))) == first


def test_running_twice_gives_identical_bytes_and_no_run_specific_fields() -> None:
    items = _items(*BASE)
    first = _dump(_run(items))
    assert _dump(_run(items)) == first
    assert not any(k in first for k in ("timestamp", "run_id", "generated"))


def test_rerunning_from_the_recorded_node_set_is_byte_identical() -> None:
    items = _items(*BASE, "jav", "blorptech", **{"web developer": ("JavaScript",)})
    first = _run(items)
    again = _run(items, recorded=json.loads(json.dumps(first["resolved"])))
    assert _dump(again) == _dump(first)


def test_a_hand_edited_node_set_resolves_an_unclear_item() -> None:
    items = _items("jav", "SQL")
    assert _run(items)["items"][0]["status"] == "unclear"
    data = _run(items, recorded=[{"item": "jav", "suite": "test", "id": "s:java"}])
    row = next(r for r in data["resolved"] if r["item"] == "jav")
    assert (row["status"], row["id"], row["label"]) == ("resolved", "s:java", "Java")
    # SQL was not recorded, so it is not searched for: recorded sets replace resolution.
    assert data["gaps"]["test"]["not_found"] == ["SQL"]


def test_a_recorded_id_that_is_not_in_the_graph_is_a_gap_with_a_warning() -> None:
    data = _run(_items("SQL"), recorded=[{"item": "SQL", "suite": "test", "id": "test:gone"}])
    assert data["gaps"]["test"]["not_found"] == ["SQL"]
    assert data["warnings"] == ["test:SQL:node_not_found"]


def test_two_items_that_resolve_to_one_node_are_linked_as_the_same_node() -> None:
    same = discover(
        _registry(),
        _items("SQL", "database"),
        depth=1,
        breadth=2,
        recorded=[
            {"item": "SQL", "suite": "test", "id": "s:sql"},
            {"item": "database", "suite": "test", "id": "s:sql"},
        ],
    )
    assert [(x["a"], x["b"], x["relation"]) for x in same["links"]] == [
        ("database", "SQL", "same_node")
    ]


def test_a_route_search_error_marks_the_orphan_check_unfinished() -> None:
    class BrokenRoutes(PathSuite):
        def enumerate_paths(self, *args: object, **kwargs: object) -> FakeToolResult:
            raise RuntimeError("down")

    g = _graph()
    data = discover(
        _registry(BrokenRoutes(list(g.nodes.values()), g.edges)),
        _items(*BASE),
        depth=1,
        breadth=4,
    )
    # Items only a route could join are orphans now, and none claims a finished check.
    assert "teamwork" in [o["item"] for o in data["orphans"]]
    assert all(o["routes_fully_checked"] is False for o in data["orphans"])
    assert any(
        w.startswith("route:") and w.endswith(":error:RuntimeError") for w in data["warnings"]
    )


def test_discover_itself_refuses_more_than_a_hundred_items() -> None:
    with pytest.raises(ValueError, match="at most 100"):
        _run(_items(*(f"item {n}" for n in range(MAX_ITEMS + 1))))


def test_hundred_and_one_items_is_an_error() -> None:
    build_items({"item": f"item {n}"} for n in range(MAX_ITEMS))
    with pytest.raises(ValueError, match="at most 100"):
        build_items({"item": f"item {n}"} for n in range(MAX_ITEMS + 1))


# ---- outputs -----------------------------------------------------------------------------


def test_outputs_html_is_static_and_deterministic_and_xlsx_has_the_eight_tabs(tmp_path) -> None:
    from openpyxl import load_workbook

    data = _run(_items(*BASE, "<script>x</script>", **{"web developer": ("JavaScript",)}))
    paths = write_outputs(data, tmp_path)
    html = paths["html"].read_text(encoding="utf-8")
    assert html == render_html(data)
    assert "<script" not in html and "src=" not in html and "href=" not in html
    assert "&lt;script&gt;x&lt;/script&gt;" in html
    assert json.loads(paths["json"].read_text(encoding="utf-8")) == data
    book = load_workbook(paths["xlsx"])
    assert book.sheetnames == [
        "Items", "Connectors", "Links", "Routes", "Orphans", "Clusters", "Guesses", "Gaps",
    ]  # fmt: skip
    assert [c.value for c in book["Guesses"][2]][:3] == [
        "web developer", "JavaScript", "Yes, linked directly",
    ]  # fmt: skip

"""The HTML pages (map, route map, successors, roadmap, discovery): one renderer, offline."""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any

import pytest

from talent_angels.cli import main
from talent_angels.report.pages import map_page, roadmap_page, successors_page
from talent_angels.skills.connect.discover import discover
from talent_angels.skills.connect.discover_report import render_html
from talent_angels.skills.connect.federated import map_all, map_all_route
from talent_angels.skills.roadmap import roadmap, successors
from tests.fakes.order_suite import registry
from tests.fakes.suite import fake_registry
from tests.test_discover import _graph, _items, _node, _registry
from tests.test_graph_mapper import _edge
from tests.test_roadmap import A, B, _esco
from tests.test_successors import X
from tests.test_successors import _esco as _skills

CITE = "http://data.europa.eu/esco/skill/abc"


def _n(i: str, label: str, kind: str, level: int, suite: str = "esco", **more: Any) -> dict:
    return {"id": i, "label": label, "kind": kind, "level": level, "suite": suite, "cite": CITE}


def _e(a: str, b: str, rel: str = "related", edge: str = "RELATED_TO", **more: Any) -> dict:
    return {"from_id": a, "to_id": b, "relation": rel, "graph_edge": edge, "strength": None,
            "suite": "esco", **more}  # fmt: skip


def _map(route: bool = False) -> dict:
    stop = (lambda k: {"on_route": True, "stop": k}) if route else (lambda k: {})
    nodes = [
        _n("esco:a", "web developer", "Occupation", 0) | stop(0),
        _n("esco:b", "software developer", "Occupation", 0) | stop(1)
        if route
        else _n(
            "esco:b", "a very long skill label that must be trimmed with an ellipsis", "Skill", 1
        ),
        _n("esco:c", "domain name service", "Skill", 1),
        _n("esco:g", "Software developers", "ISCOGroup", 1),
        _n("esco:d", "deeper skill", "Skill", 2),
        _n("onet:x", "Web Developers", "Occupation", 0, "onet") | stop(0),
    ]
    edges = [
        _e("esco:a", "esco:c", "needs", "HAS_SKILL"),
        _e("esco:a", "esco:g", "parent", "BROADER_THAN"),
        _e("esco:c", "esco:d"),
        _e("esco:g", "esco:d", "narrower", "BROADER_THAN"),  # a second way in: a faint line
        {"from_id": "esco:a", "to_id": "onet:x", "relation": "same_as", "graph_edge": "CROSSWALK",
         "strength": "soc", "source": "crosswalk:test"},
    ]  # fmt: skip
    if not route:
        edges.insert(0, _e("esco:a", "esco:b", "related"))
    suites = {
        "esco": {"status": "mapped", "start": None, "candidates": [], "via": None, "nodes": 5,
                 "pruned": [], "warnings": []},
        "onet": {"status": "mapped", "start": None, "candidates": [], "via": None, "nodes": 1,
                 "pruned": [], "warnings": []},
        "sfia": {"status": "ambiguous", "start": None, "via": None, "nodes": 0, "pruned": [],
                 "warnings": [],
                 "candidates": [{"id": "sfia:skill:BRMG", "label": "Brand", "kind": "Skill"}]},
    }  # fmt: skip
    if route:
        suites["esco"]["route"] = {"node_ids": ["esco:a", "esco:b"]}
        suites["sfia"]["side"] = "target"
    return {"query": "web developer -> software developer" if route else "web developer",
            "policy": "map-rank-v1", "meaning": "linked in the graph; not a learning order",
            "depth": 2, "breadth": 3, "suites": suites, "nodes": nodes, "edges": edges,
            "summary": {}, "warnings": []}  # fmt: skip


def _succ() -> dict:
    def n(i: str, label: str, depth: int) -> dict:
        return {"id": i, "label": label, "kind": "Skill", "cite": CITE, "depth": depth,
                "reuse_level": "cross-sector"}  # fmt: skip

    return {
        "policy": "learn-order-v1", "meaning": "ordered by learn-order-v1; not a promise",
        "x": "marketing", "depth": 2, "breadth": 4, "max_nodes": 200, "warnings": [],
        "suites": {
            "esco": {"status": "ok", "start": {"id": "esco:s", "label": "marketing"},
                     "candidates": [], "warnings": [],
                     "nodes": [n("esco:s", "marketing", 0), n("esco:t", "plan marketing", 1),
                               n("esco:u", "mobile marketing", 2)],
                     "edges": [{"from_id": "esco:s", "to_id": "esco:t", "kind": "stated",
                                "reason": "esco:essential-for"},
                               {"from_id": "esco:t", "to_id": "esco:u", "kind": "policy",
                                "reason": "esco:narrower"}],
                     "pruned": []},
            "sfia": {"status": "ambiguous", "start": None, "warnings": [],
                     "candidates": [{"id": "sfia:skill:BRMG", "label": "Brand management",
                                     "kind": "Skill"}]},
            "bls": {"status": "no_order_data", "start": None, "candidates": [], "warnings": []},
        },
    }  # fmt: skip


def _road() -> dict:
    def item(suite: str, label: str, step: int, stated: bool, levels: dict | None = None) -> dict:
        out = {
            "suite": suite,
            "id": f"{suite}:{label}",
            "label": label,
            "kind": "Skill",
            "cite": "2.B.3.e",
            "step": step,
            "reasons": [],
            "stated_prerequisite": stated,
        }
        return out | ({"levels": levels} if levels else {})  # fmt: skip

    return {
        "policy": "learn-order-v1",
        "meaning": "ordered by learn-order-v1; only stated ones are prerequisites",
        "from": "A", "to": "B", "per_step": 10, "warnings": [],
        "suites": {
            "onet": {"status": "ok", "start": None, "target": None, "candidates": [], "items": 2,
                     "warnings": []},
            "esco": {"status": "ambiguous", "start": None, "target": None, "items": 0,
                     "warnings": [], "candidates": [{"id": "esco:occupation:1", "label": "analyst",
                                                     "kind": "Occupation"}]},
        },
        "steps": [
            {"step": 2, "name": "Core", "more": 3,
             "items": [item("onet", "Programming", 2, True,
                            {"from": 2.5, "to": 3.88, "jump": 1.38, "importance": 3.75}),
                       item("onet", "Writing", 2, False,
                            {"from": None, "to": 3.0, "jump": 3.0, "importance": 3.1})]},
        ],
        "credentials": [{"suite": "bls", "id": "bls:o:1", "label": "Data Scientists",
                         "kind": "Occupation", "cite": "15-2051",
                         "entry_education": "Master's degree",
                         "work_experience": "None", "job_training": "None"}],
        "job_zone": {"from": {"label": "Job Zone Three"}, "to": {"label": "Job Zone Four"}},
    }  # fmt: skip


PAGES = {
    "map": (map_page, _map()),
    "route": (map_page, _map(route=True)),
    "successors": (successors_page, _succ()),
    "roadmap": (roadmap_page, _road()),
}


@pytest.mark.parametrize("name", PAGES)
def test_each_page_renders_twice_to_the_same_bytes_and_does_not_change_its_input(name: str) -> None:
    render, data = PAGES[name]
    before = copy.deepcopy(data)
    first = render(data)
    assert first == render(data) == render(copy.deepcopy(data))
    assert data == before and first.startswith("<!doctype html>")


@pytest.mark.parametrize("name", PAGES)
def test_pages_follow_the_system_theme_are_static_and_phone_safe(name: str) -> None:
    render, data = PAGES[name]
    html = render(data)
    assert "@media (prefers-color-scheme:dark)" in html and "color-scheme:light dark" in html
    assert re.search(r"body\{background:var\(--bg\);color:var\(--text\)", html)
    assert 'name="viewport"' in html and "overflow-x:auto" in html
    assert not re.search(r"<script|<link|@import|src=|url\(|<iframe|<img", html)
    hrefs = re.findall(r'href="([^"]+)"', html)
    assert all(h.startswith("http://data.europa.eu/") for h in hrefs)  # only the citations
    assert not re.search(r"(Date|date)\b.*\d{4}-\d\d-\d\d|timestamp", html)


@pytest.mark.parametrize("name", ["map", "route", "successors"])
def test_pictures_have_an_aria_label_and_a_legend(name: str) -> None:
    render, data = PAGES[name]
    html = render(data)
    assert re.search(r'<svg class="tree"[^>]* role="img" aria-label="[^"]{20,}"', html)
    assert 'aria-label="Legend"' in html and "skill</li>" in html and "job</li>" in html


def test_the_map_is_a_layered_tree_with_labelled_edges_and_a_table_below() -> None:
    html = map_page(_map())
    for word in ("related", "needs", "broader"):
        assert 'class="el"' in html and f">{word}</text>" in html
    assert html.count('<path class="e-extra"') == 1  # the second way into "deeper skill"
    assert "…" in html and "<title>a very long skill label that must be trimmed" in html
    assert html.count("<section>") == 4  # esco, onet, crosswalk list, every node
    assert "sfia: the name (the second name)" not in html  # the plain map has no sides
    assert "Rerun with one of these ids: Brand = sfia:skill:BRMG" in html
    assert "Same thing in another taxonomy" in html and "soc" in html
    assert "Linked in the graph; not a learning order." in html
    assert html.index("<table>") > html.index("</svg>")
    assert "<th>Taxonomy</th>" in html and f'<a href="{CITE}"' in html
    levels = [int(m) for m in re.findall(r"<tr><td>(\d)</td><td>", html)]
    assert levels == sorted(levels, key=lambda v: 0) and set(levels) == {0, 1, 2}
    assert "e-spine" not in html.split("</style>")[1]


def test_a_route_map_draws_the_route_as_a_spine_with_the_neighbourhood_off_it() -> None:
    html = map_page(_map(route=True)).split("</style>")[1]
    assert html.count('<path class="e-spine"') == 1  # two stops, one spine segment
    assert html.count("n-route") == 3 and "stop 1" in html and "stop 2" in html
    assert 'aria-label="Graph map of web developer -&gt; software developer in esco' in html
    assert "(the second name) is ambiguous" in html
    assert "<th>Route stop</th>" in html


def test_a_map_from_the_real_core_renders(tmp_path: Path) -> None:
    reg = fake_registry()
    page = map_page(map_all(reg, "software developer", suites=["fake"], depth=1).to_dict())
    assert "<svg" in page and "software developer" in page
    route = map_page(
        map_all_route(reg, "software developer", "programmer", suites=["fake"]).to_dict()
    )
    assert "Graph map: software developer" in route


def test_successors_draw_stated_edges_solid_and_policy_edges_dashed() -> None:
    html = successors_page(_succ())
    body = html.split("</style>")[1]
    assert body.count('class="e-stated"') >= 2 and body.count('class="e-policy"') >= 2  # + legend
    assert re.search(r"\.e-policy\{[^}]*stroke-dasharray", html)
    assert not re.search(r"\.e-stated\{[^}]*stroke-dasharray", html)
    assert "stated by the taxonomy" in html and "our reading (policy), dashed" in html
    assert ">needs it</text>" in html and ">narrower</text>" in html
    assert "no order data for this question" in html
    assert "Rerun with one of these ids: Brand management = sfia:skill:BRMG" in html


def test_successors_from_the_real_core_render_with_both_edge_styles() -> None:
    data = successors(registry(esco=_skills()), X.id, depth=2, breadth=4)
    body = successors_page(data).split("</style>")[1]
    assert 'class="e-stated"' in body and 'class="e-policy"' in body


def test_the_roadmap_has_four_step_columns_jumps_badges_and_blocks() -> None:
    html = roadmap_page(_road())
    for title in ("1. Foundations", "2. Core", "3. Field", "4. Job-specific"):
        assert f"<h3>{title}</h3>" in html
    assert "level 2.50→3.88" in html and "level n/a→3.00" in html
    assert html.count('<span class="badge">stated prerequisite</span>') == 1
    assert "+3 more in the JSON" in html and "Nothing here." in html
    assert "Master&#x27;s degree" in html and "Credentials the target job asks for" in html
    assert "Job Zone Three" in html and "Job Zone Four" in html
    assert "Rerun with one of these ids: analyst = esco:occupation:1" in html
    assert "only stated ones are prerequisites" in html.replace("Ordered", "ordered")


def test_a_roadmap_from_the_real_core_renders() -> None:
    page = roadmap_page(roadmap(registry(esco=_esco()), A.id, B.id, per_step=2))
    assert "algebra" in page and "stated prerequisite" in page


# ---- CLI --html ---------------------------------------------------------------------------


def test_cli_html_writes_the_page_next_to_the_json(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    for argv, reg, marker in (
        (["roadmap", A.id, B.id], registry(esco=_esco()), "Foundations"),
        (["successors", X.id], registry(esco=_skills()), "What comes after"),
        (["map", "software developer", "--suites", "fake"], fake_registry(), "Graph map"),
        (["map", "--route", "software developer", "programmer", "--suites", "fake"],
         fake_registry(), "Graph map"),
    ):  # fmt: skip
        out = tmp_path / f"{argv[0]}-{len(list(tmp_path.iterdir()))}" / "page.html"
        code = main([*argv, "--html", str(out)], registry=reg)
        captured = capsys.readouterr()
        assert code in (0, 2) and json.loads(captured.out) and f"html: {out}" in captured.err
        assert marker in out.read_text(encoding="utf-8")


def test_cli_without_html_writes_no_page(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["roadmap", A.id, B.id], registry=registry(esco=_esco())) == 0
    assert "html:" not in capsys.readouterr().err


# ---- discovery: tool-only connectors, labelled dots -----------------------------------------


def _tool_graph():  # noqa: ANN202
    g = _graph()
    for key, label in (("linux", "Linux"), ("git", "git"), ("vim", "vim")):
        g.nodes[f"t:{key}"] = _node(f"t:{key}", "Software", label)
    for occ, label in (("ae", "Aerospace Engineers"), ("av", "Avionics Technicians")):
        g.nodes[f"occ:{occ}"] = _node(f"occ:{occ}", "Occupation", label)
    uses = [("ae", "linux"), ("ae", "git"), ("ae", "vim"), ("av", "linux"), ("av", "git")]
    g.edges += [_edge("USES_SOFTWARE", f"occ:{o}", f"t:{t}") for o, t in uses]
    return g


def _tool_run() -> dict:
    items = _items("JavaScript", "Python", "Linux", "git", "vim")
    return discover(_registry(_tool_graph()), items, depth=1, breadth=4)


def test_tool_only_connectors_are_flagged_and_ranked_after_specific_and_broad() -> None:
    data = _tool_run()
    rows = [(c["label"], c["item_count"], c["broad"], c["tool_only"]) for c in data["connectors"]]
    assert rows == [
        ("scripting", 2, False, False),
        ("programming", 2, True, False),  # broad: after specific ...
        ("Aerospace Engineers", 3, False, True),  # ... tool-only last, though it has 3 items
        ("Avionics Technicians", 2, False, True),
    ]
    order = [(c["tool_only"], c["broad"], -c["item_count"], c["label"].casefold(), c["id"])
             for c in data["connectors"]]  # fmt: skip
    assert order == sorted(order)


def test_the_discovery_page_greys_tool_only_rows_and_labels_the_connector_dots() -> None:
    html = render_html(_tool_run())
    assert "<tr class=dim><td>Aerospace Engineers</td>" in html
    assert "linked only through a tool" in html and "USES_SOFTWARE" in html
    start = html.index('<svg class="ring"')
    svg = html[start : html.index("</svg>", start)]
    assert ">scripting</text>" in svg  # a named connector in the panel
    assert ">1</text>" in svg  # and its number on the dot
    assert "Aerospace Engineers" not in svg  # tool-only connectors stay off the picture
    assert "@media (prefers-color-scheme:dark)" in html and "<script" not in html
    assert 'aria-label="Legend"' in html and re.search(r'role="img" aria-label="Items and', html)
    assert html == render_html(_tool_run())

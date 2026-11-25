"""Successors under learn-order-v1: stated before policy, no repeats, caps, hubs, no order data."""

from __future__ import annotations

import json
from typing import Any

import pytest

from talent_angels.skills.roadmap import successors
from tests.fakes.order_suite import OrderSuite, edge, node, registry


def _skill(key: str, label: str, reuse: str) -> Any:
    return node("esco", key, "Skill", label, skill_type="skill/competence", reuse_level=reuse)


X = _skill("x", "base", "cross-sector")
S = {
    "sa": _skill("sa", "stated sector", "sector-specific"),
    "sb": _skill("sb", "stated transversal", "transversal"),
    "n1": _skill("n1", "narrow job", "occupation-specific"),
    "n2": _skill("n2", "narrow transversal", "transversal"),
    "rl": _skill("rl", "related later", "sector-specific"),
    "rs": _skill("rs", "related same", "cross-sector"),
    "re": _skill("re", "related earlier", "transversal"),
    "pre": _skill("pre", "prerequisite of base", "cross-sector"),
    "gc": _skill("gc", "grandchild", "sector-specific"),
    "uh": _skill("uh", "under the hub", "sector-specific"),
}
HUB = node("esco", "grp", "SkillGroup", "a group")
PARENT = node("esco", "par", "SkillGroup", "parent group")
OCC = node("esco", "occ", "Occupation", "some job")


def _esco(shuffle: int = 0) -> OrderSuite:
    need = lambda a, b: edge("RELATED_TO", a, b, relation_type="essential")  # noqa: E731
    edges = [
        need(S["sa"], X),
        need(S["sb"], X),  # stated: these skills need X
        need(X, S["pre"]),  # X needs this: it comes before X, never after
        need(S["gc"], S["sa"]),  # a stated grandchild
        edge("BROADER_THAN", S["n1"], X),  # incoming = narrower, as the mapper reads it
        edge("BROADER_THAN", S["n2"], X),
        edge("BROADER_THAN", S["n1"], S["sa"]),  # n1 is also under sa: it must not repeat
        edge("BROADER_THAN", HUB, X),
        edge("BROADER_THAN", S["uh"], HUB),  # behind the hub: never reached
        edge("BROADER_THAN", X, PARENT),  # outgoing = parent
        edge("RELATED_TO", X, S["rl"], relation_type="optional"),
        edge("RELATED_TO", S["rs"], X, relation_type="optional"),
        edge("RELATED_TO", X, S["re"], relation_type="optional"),
        edge("HAS_SKILL", OCC, X, relation_type="essential"),  # a job: not a successor
    ]
    return OrderSuite([X, HUB, PARENT, OCC, *S.values()], edges, shuffle)


def _ids(data: dict[str, Any], suite: str = "esco") -> list[str]:
    return [n["id"].split(":")[1] for n in data["suites"][suite]["nodes"]]


def test_stated_children_come_first_then_reuse_level_label_and_id() -> None:
    data = successors(registry(esco=_esco()), X.id, depth=1, breadth=10)

    part = data["suites"]["esco"]
    assert part["status"] == "ok"
    assert _ids(data) == ["x", "sb", "sa", "n2", "rl", "n1", "grp"]
    assert [(e["to_id"].split(":")[1], e["kind"], e["reason"]) for e in part["edges"]] == [
        ("sb", "stated", "esco:essential-for"),
        ("sa", "stated", "esco:essential-for"),
        ("n2", "policy", "esco:narrower"),
        ("rl", "policy", "esco:related-later-reuse"),
        ("n1", "policy", "esco:narrower"),
        ("grp", "policy", "esco:narrower"),
    ]


def test_a_prerequisite_a_parent_a_job_and_an_earlier_related_skill_are_not_successors() -> None:
    data = successors(registry(esco=_esco()), X.id, depth=2, breadth=10)

    reached = set(_ids(data))
    assert reached.isdisjoint({"pre", "par", "occ", "rs", "re"})
    assert "uh" not in reached  # a hub is shown but never expanded


def test_a_node_is_never_repeated() -> None:
    data = successors(registry(esco=_esco()), X.id, depth=3, breadth=10)

    ids = _ids(data)
    assert len(ids) == len(set(ids))
    assert [e["from_id"] for e in data["suites"]["esco"]["edges"] if e["to_id"] == "esco:n1"] == [
        "esco:x"
    ]  # reached once, from the first parent
    assert ("esco:sa", "esco:gc") in {
        (e["from_id"], e["to_id"]) for e in data["suites"]["esco"]["edges"]
    }


def test_depth_breadth_and_max_nodes_are_hard_caps() -> None:
    reg = registry(esco=_esco())

    assert _ids(successors(reg, X.id, depth=1, breadth=2)) == ["x", "sb", "sa"]
    assert {n["depth"] for n in successors(reg, X.id, depth=1)["suites"]["esco"]["nodes"]} <= {0, 1}
    capped = successors(reg, X.id, depth=2, breadth=10, max_nodes=3)
    part = capped["suites"]["esco"]
    assert _ids(capped) == ["x", "sb", "sa"]
    assert part["warnings"] == ["max_nodes:3"]
    assert part["pruned"][-1] == {"node": "esco:x", "considered": 6, "kept": 2}
    for bad in ({"depth": 0}, {"depth": 5}, {"breadth": 11}, {"max_nodes": 0}):
        with pytest.raises(ValueError):
            successors(reg, X.id, **bad)


def test_the_result_is_byte_identical_and_independent_of_graph_order() -> None:
    def run(shuffle: int) -> str:
        return json.dumps(successors(registry(esco=_esco(shuffle)), X.id, depth=3, breadth=10))

    assert run(0) == run(0) == run(1) == run(2)


def test_a_name_resolves_by_the_mappers_rule_and_an_unclear_one_returns_candidates() -> None:
    reg = registry(esco=_esco())

    assert successors(reg, "base")["suites"]["esco"]["start"]["id"] == "esco:x"
    unclear = successors(reg, "narrow")["suites"]["esco"]
    assert unclear["status"] == "ambiguous"
    assert [c["id"] for c in unclear["candidates"]] == ["esco:n1", "esco:n2"]
    assert successors(reg, "zzz")["suites"]["esco"]["status"] == "not_found"


def test_a_job_is_not_a_skill_so_it_has_no_successors() -> None:
    part = successors(registry(esco=_esco()), OCC.id)["suites"]["esco"]

    assert part["status"] == "no_order_data"
    assert part["warnings"] == ["not_a_skill:Occupation"]


# ---- SFIA: the next level, stated; taxonomies without order data -----------------------------


def _sfia() -> OrderSuite:
    skill = node("sfia", "prog", "Skill", "Programming")
    levels = [
        node("sfia", f"level:prog:{n}", "SkillLevel", f"Programming (level {n})") for n in (4, 2, 3)
    ]
    return OrderSuite([skill, *levels], [edge("HAS_LEVEL", skill, lv) for lv in levels])


def test_sfia_gives_the_levels_of_a_skill_in_order_as_stated_successors() -> None:
    data = successors(registry(sfia=_sfia()), "sfia:prog", depth=4)

    part = data["suites"]["sfia"]
    assert [n["id"] for n in part["nodes"]] == [
        "sfia:prog",
        "sfia:level:prog:2",
        "sfia:level:prog:3",
        "sfia:level:prog:4",
    ]
    assert [(e["kind"], e["reason"]) for e in part["edges"]] == [
        ("stated", "sfia:first-level"),
        ("stated", "sfia:next-level"),
        ("stated", "sfia:next-level"),
    ]
    from_three = successors(registry(sfia=_sfia()), "sfia:level:prog:3")["suites"]["sfia"]
    assert [n["id"] for n in from_three["nodes"]] == ["sfia:level:prog:3", "sfia:level:prog:4"]


def test_other_taxonomies_report_no_order_data_without_being_opened() -> None:
    onet = OrderSuite([], [])
    data = successors(registry(esco=_esco(), onet=onet), X.id)

    assert data["suites"]["onet"] == {
        "status": "no_order_data",
        "start": None,
        "candidates": [],
        "warnings": [],
    }
    assert onet.calls == []


def test_a_broken_taxonomy_is_an_error_status_and_the_meaning_is_printed() -> None:
    data = successors(registry(esco=_esco(), sfia=ConnectionRefusedError("down")), X.id)

    assert data["suites"]["sfia"]["status"] == "error:ConnectionRefusedError"
    assert data["suites"]["esco"]["status"] == "ok"
    assert data["policy"] == "learn-order-v1"
    assert "only items marked stated_prerequisite" in data["meaning"]

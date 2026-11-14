"""Rendering a pathfind result: gap buckets, routes, retrain line (PF-4)."""

from __future__ import annotations

from talent_angels.contracts import AgentResult, EdgeRef, NodeRef, Route
from talent_angels.session.pathfind_view import render_pathfind


def _node(label: str, kind: str = "Skill") -> NodeRef:
    return NodeRef(
        id=f"esco:{kind.casefold()}:{label}",
        suite="esco",
        source="esco",
        source_id=label,
        kind=kind,
        pref_label=label,
    )


START = _node("software developer", "Occupation")
TARGET = _node("web developer", "Occupation")
STATS = _node("statistics")
SQL = _node("sql")
PYTHON = _node("python", "Software")
BRIDGE = _node("computer programming")


def _edge(source: NodeRef, target: NodeRef, edge_type: str, relation: str | None) -> EdgeRef:
    props = {"relation_type": relation} if relation is not None else {}
    return EdgeRef(
        type=edge_type,
        suite="esco",
        source_node_id=target.id,
        target_node_id=source.id,
        properties=props,
    )


def test_gap_buckets_split_essential_optional_and_software() -> None:
    result = AgentResult(
        capability="pathfind",
        suite="esco",
        nodes=[START, TARGET, STATS, SQL, PYTHON],
        edges=[
            _edge(STATS, TARGET, "HAS_SKILL", "essential"),
            _edge(SQL, TARGET, "HAS_SKILL", "optional"),
            _edge(PYTHON, TARGET, "USES_SOFTWARE", None),
        ],
    )

    text = render_pathfind(result)

    assert "**Essential skills needed** (1): statistics" in text
    assert "**Optional skills needed** (1): sql" in text
    assert "**Software/tools needed** (1): python" in text


def test_a_route_bridge_node_is_not_counted_as_a_gap_skill() -> None:
    result = AgentResult(
        capability="pathfind",
        suite="esco",
        nodes=[START, TARGET, STATS, BRIDGE],
        edges=[
            _edge(STATS, TARGET, "HAS_SKILL", "essential"),
            EdgeRef(
                type="HAS_SKILL", suite="esco", source_node_id=START.id, target_node_id=BRIDGE.id
            ),
        ],
        routes=[Route(node_ids=[START.id, BRIDGE.id, TARGET.id], edges=[])],
    )

    text = render_pathfind(result)

    assert "**Essential skills needed** (1): statistics" in text
    assert "Route 1: software developer → *computer programming* → web developer" in text


def test_gap_too_large_adds_the_full_retrain_line() -> None:
    result = AgentResult(
        capability="pathfind",
        suite="esco",
        nodes=[START, TARGET, STATS],
        edges=[_edge(STATS, TARGET, "HAS_SKILL", "essential")],
        warnings=["gap_too_large"],
    )

    text = render_pathfind(result)

    assert "full retrain" in text


def test_no_route_found_still_shows_the_gap() -> None:
    result = AgentResult(
        capability="pathfind",
        suite="esco",
        nodes=[START, TARGET, STATS],
        edges=[_edge(STATS, TARGET, "HAS_SKILL", "essential")],
        warnings=["no_path"],
    )

    text = render_pathfind(result)

    assert "**Essential skills needed** (1): statistics" in text
    assert "No route found within 4 steps" in text


def test_more_than_the_preview_cap_is_counted_not_listed() -> None:
    skills = [_node(f"skill-{i}") for i in range(7)]
    result = AgentResult(
        capability="pathfind",
        suite="esco",
        nodes=[START, TARGET, *skills],
        edges=[_edge(s, TARGET, "HAS_SKILL", "essential") for s in skills],
    )

    text = render_pathfind(result)

    assert "**Essential skills needed** (7):" in text
    assert "(+2 more)" in text
    assert "skill-6" not in text

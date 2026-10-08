"""Shared and own skills of two occupations, from two Connect results."""

from __future__ import annotations

import pytest

from talent_angels.contracts import AgentResult, EdgeRef, NodeRef
from talent_angels.skills.connect.compare import compare_result, skill_overlap


def _node(label: str, kind: str = "Skill", suite: str = "esco") -> NodeRef:
    return NodeRef(
        id=f"{suite}:{kind.casefold()}:{label}",
        suite=suite,
        source=suite,
        source_id=label,
        kind=kind,
        pref_label=label,
    )


def _connect(occupation: str, skills: list[str], suite: str = "esco") -> AgentResult:
    centre = _node(occupation, "Occupation", suite)
    neighbours = [_node(skill, suite=suite) for skill in skills]
    edges = [
        EdgeRef(
            type="HAS_SKILL",
            suite=suite,
            source_node_id=centre.id,
            target_node_id=node.id,
            properties={"relation_type": "essential"},
        )
        for node in neighbours
    ]
    return AgentResult(capability="connect", suite=suite, nodes=[centre, *neighbours], edges=edges)


def _labels(nodes: tuple[NodeRef, ...]) -> list[str]:
    return [node.pref_label for node in nodes]


def test_splits_shared_and_own_skills_in_neighbour_order() -> None:
    accountant = _connect("accountant", ["accounting", "use spreadsheets", "auditing"])
    developer = _connect("software developer", ["programming", "use spreadsheets", "debug"])

    overlap = skill_overlap(compare_result(accountant, developer))

    assert overlap.a.pref_label == "accountant"
    assert overlap.b.pref_label == "software developer"
    assert _labels(overlap.shared) == ["use spreadsheets"]
    assert _labels(overlap.only_a) == ["accounting", "auditing"]
    assert _labels(overlap.only_b) == ["programming", "debug"]


def test_compare_result_lists_each_node_once_with_both_centres_first() -> None:
    result = compare_result(_connect("a", ["x", "y"]), _connect("b", ["y", "z"]))

    assert result.capability == "compare"
    assert [node.pref_label for node in result.nodes] == ["a", "b", "x", "y", "z"]
    assert len(result.edges) == 4


def test_identical_occupations_share_everything() -> None:
    overlap = skill_overlap(compare_result(_connect("a", ["x"]), _connect("b", ["x"])))
    assert _labels(overlap.shared) == ["x"]
    assert overlap.only_a == overlap.only_b == ()


def test_refuses_to_compare_across_suites() -> None:
    with pytest.raises(ValueError, match="across suites"):
        compare_result(_connect("a", ["x"]), _connect("b", ["x"], suite="onet"))


def test_refuses_a_side_without_a_centre() -> None:
    empty = AgentResult(capability="connect", suite="esco", warnings=["not_found"])
    with pytest.raises(ValueError, match="centre"):
        compare_result(_connect("a", ["x"]), empty)


def test_skill_overlap_rejects_a_plain_connect_result() -> None:
    with pytest.raises(ValueError, match="not a compare result"):
        skill_overlap(_connect("a", ["x"]))

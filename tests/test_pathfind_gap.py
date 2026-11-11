"""Skill gap computation (ADR-0008 decisions 3 and 5)."""

from __future__ import annotations

from talent_angels.contracts import AgentResult, EdgeRef, NodeRef
from talent_angels.skills.pathfind.gap import MAX_REALISTIC_GAP_SKILLS, compute_gap


def _node(label: str, kind: str = "Skill", suite: str = "esco") -> NodeRef:
    return NodeRef(
        id=f"{suite}:{kind.casefold()}:{label}",
        suite=suite,
        source=suite,
        source_id=label,
        kind=kind,
        pref_label=label,
    )


def _connect(
    occupation: str, skills: list[tuple[str, str | None]], suite: str = "esco"
) -> AgentResult:
    """skills: (label, relation_type) pairs; relation_type None = no signal."""
    centre = _node(occupation, "Occupation", suite)
    neighbours = [_node(label, suite=suite) for label, _ in skills]
    edges = [
        EdgeRef(
            type="HAS_SKILL",
            suite=suite,
            source_node_id=centre.id,
            target_node_id=node.id,
            properties={"relation_type": relation} if relation is not None else {},
        )
        for node, (_, relation) in zip(neighbours, skills, strict=True)
    ]
    return AgentResult(capability="connect", suite=suite, nodes=[centre, *neighbours], edges=edges)


def _labels(nodes: tuple[NodeRef, ...]) -> list[str]:
    return [node.pref_label for node in nodes]


def test_shared_skills_are_excluded_from_the_gap() -> None:
    start = _connect("nurse", [("care", "essential"), ("hygiene", "essential")])
    target = _connect(
        "data analyst", [("care", "essential"), ("statistics", "essential"), ("sql", "essential")]
    )

    gap = compute_gap(start, target)

    assert _labels(gap.essential_missing) == ["statistics", "sql"]
    assert gap.optional_missing == ()
    assert not gap.is_empty


def test_essential_and_optional_are_split_and_essential_comes_first() -> None:
    start = _connect("nurse", [])
    target = _connect(
        "data analyst",
        [("statistics", "essential"), ("r language", "optional"), ("sql", "essential")],
    )

    gap = compute_gap(start, target)

    assert _labels(gap.essential_missing) == ["statistics", "sql"]
    assert _labels(gap.optional_missing) == ["r language"]


def test_a_relation_with_no_signal_counts_as_essential() -> None:
    """Matches connect()'s own rule: no relation_type survives every filter."""
    start = _connect("nurse", [])
    target = _connect("web developer", [("javascript", None)])

    gap = compute_gap(start, target)

    assert _labels(gap.essential_missing) == ["javascript"]
    assert gap.optional_missing == ()


def test_custom_optional_rel_values_match_the_suite_schema() -> None:
    start = _connect("nurse", [])
    target = _connect("data scientist", [("python", "transferable")])

    default_split = compute_gap(start, target)
    assert _labels(default_split.optional_missing) == ["python"]

    narrower_split = compute_gap(start, target, optional_rel_values=frozenset({"optional"}))
    assert _labels(narrower_split.essential_missing) == ["python"]


def test_gap_is_empty_when_start_already_covers_the_target() -> None:
    start = _connect("software developer", [("programming", "essential")])
    target = _connect("web developer", [("programming", "essential")])

    gap = compute_gap(start, target)

    assert gap.is_empty
    assert gap.gap_too_large is False


def test_gap_too_large_flags_past_the_threshold_but_still_returns_the_list() -> None:
    skills = [(f"skill-{i}", "essential") for i in range(MAX_REALISTIC_GAP_SKILLS + 1)]
    start = _connect("nurse", [])
    target = _connect("data scientist", skills)

    gap = compute_gap(start, target)

    assert gap.gap_too_large is True
    assert len(gap.essential_missing) == MAX_REALISTIC_GAP_SKILLS + 1


def test_gap_at_exactly_the_threshold_is_not_flagged() -> None:
    skills = [(f"skill-{i}", "essential") for i in range(MAX_REALISTIC_GAP_SKILLS)]
    start = _connect("nurse", [])
    target = _connect("data scientist", skills)

    gap = compute_gap(start, target)

    assert gap.gap_too_large is False
    assert len(gap.essential_missing) == MAX_REALISTIC_GAP_SKILLS


def test_onet_uses_its_own_higher_threshold_than_esco() -> None:
    """O*NET counts software tools as skills, so its essential-missing counts
    run far higher than ESCO's and are not on the same scale (measured
    against live ESCO/O*NET in PR 6's golden pairs set)."""
    skill_count = MAX_REALISTIC_GAP_SKILLS + 10
    skills = [(f"skill-{i}", "essential") for i in range(skill_count)]
    start = _connect("nurse", [], suite="onet")
    target = _connect("data scientist", skills, suite="onet")

    gap = compute_gap(start, target)

    assert gap.gap_too_large is False
    assert len(gap.essential_missing) == skill_count

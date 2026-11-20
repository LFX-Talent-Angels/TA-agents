"""Compare two Connect results in one suite: shared skills, and each side's own.

Pure set arithmetic over graph neighbours — no model call, no traversal. The
compare result keeps both centres first and every edge of both sides, so the
split is recomputed from the typed result rather than carried as prose.
Pathfinder's skill gap from A to B is ``only_b``.
"""

from __future__ import annotations

from dataclasses import dataclass

from talent_angels.contracts import AgentResult, NodeRef

CAPABILITY_COMPARE = "compare"


@dataclass(frozen=True)
class SkillOverlap:
    a: NodeRef
    b: NodeRef
    shared: tuple[NodeRef, ...]
    only_a: tuple[NodeRef, ...]
    only_b: tuple[NodeRef, ...]


def _neighbours(result: AgentResult) -> list[NodeRef]:
    return result.nodes[1:]


def compare_result(a: AgentResult, b: AgentResult) -> AgentResult:
    """One typed result from two Connect results of the same suite."""
    if a.suite != b.suite:
        raise ValueError(f"cannot compare across suites: {a.suite} vs {b.suite}")
    if not a.nodes or not b.nodes:
        raise ValueError("both sides need a resolved centre node")
    seen = {a.nodes[0].id, b.nodes[0].id}
    neighbours: list[NodeRef] = []
    for node in [*_neighbours(a), *_neighbours(b)]:
        if node.id not in seen:
            seen.add(node.id)
            neighbours.append(node)
    return AgentResult(
        capability=CAPABILITY_COMPARE,
        suite=a.suite,
        nodes=[a.nodes[0], b.nodes[0], *neighbours],
        edges=[*a.edges, *b.edges],
        evidence=[*a.evidence, *b.evidence],
        warnings=list(dict.fromkeys([*a.warnings, *b.warnings])),
    )


def skill_overlap(result: AgentResult) -> SkillOverlap:
    """Split a compare result into shared / only-A / only-B, in neighbour order."""
    if result.capability != CAPABILITY_COMPARE or len(result.nodes) < 2:
        raise ValueError("not a compare result")
    a, b = result.nodes[0], result.nodes[1]
    linked: dict[str, set[str]] = {a.id: set(), b.id: set()}
    for edge in result.edges:
        for centre, other in (
            (edge.source_node_id, edge.target_node_id),
            (edge.target_node_id, edge.source_node_id),
        ):
            if centre in linked:
                linked[centre].add(other)
    shared: list[NodeRef] = []
    only_a: list[NodeRef] = []
    only_b: list[NodeRef] = []
    for node in result.nodes[2:]:
        in_a, in_b = node.id in linked[a.id], node.id in linked[b.id]
        if in_a and in_b:
            shared.append(node)
        elif in_a:
            only_a.append(node)
        elif in_b:
            only_b.append(node)
    return SkillOverlap(a, b, tuple(shared), tuple(only_a), tuple(only_b))

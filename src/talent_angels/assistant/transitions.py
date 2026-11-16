"""'What can a X move to?': the job-to-job moves a taxonomy itself publishes.

Only graph data. JobTech publishes substitutability between occupations,
`(X)-[:SUBSTITUTES {substitutability_pct}]->(Y)`, read as "a person in X can
do Y's job to that degree" (inferred from the data: a basic nurse substitutes
a geriatric nurse at 75% and a children's nurse at 25%; see ADR-0009). O*NET
publishes related occupations. ESCO, BLS and SFIA publish none, and say so.
This is not Pathfinder: one step, no skill gap, no ranking.
"""

from __future__ import annotations

import re

from talent_angels.contracts import AgentResult
from talent_angels.skills.connect import ConnectRequest, connect
from talent_angels.suites import SuiteRegistry

#: The relationship that carries a suite's own job-to-job moves, outgoing only.
TRANSITION_RELS: dict[str, tuple[str, ...]] = {
    "jobtech": ("SUBSTITUTES",),
    "onet": ("RELATED_TO",),
}
WARNING = "transitions"
NO_DATA = "no_transition_data"

_QUESTION = re.compile(
    r"\b(move|moving|switch|switching|transition|transfer|change)\s+(in)?to\b|"
    r"\b(similar|related|other|alternative|nearby)\s+(jobs|occupations|roles|careers)\b|"
    r"\bwhat\s+else\s+can\b|"
    r"\bbyta\s+till\b|\bbyta\s+yrke\b|\bliknande\s+yrken\b|\bkan\s+.+\s+(byta|gå)\s+till\b",
    re.IGNORECASE,
)


def is_transition_question(question: str) -> bool:
    return bool(_QUESTION.search(question))


def _strength(edge: object) -> float:
    """JobTech's percentage when there is one; otherwise keep the suite's order."""
    pct = getattr(edge, "properties", {}).get("substitutability_pct")
    return float(pct) if isinstance(pct, (int, float)) else 0.0


def transitions_for(registry: SuiteRegistry, result: AgentResult) -> AgentResult | None:
    """The moves from ``result``'s one clear occupation, or None to leave it."""
    if not result.nodes or "ambiguous" in result.warnings:
        return None
    center = result.nodes[0]
    if center.kind != "Occupation":
        return None
    rels = TRANSITION_RELS.get(result.suite)
    if not rels:
        return AgentResult(
            capability="connect",
            suite=result.suite,
            nodes=[center],
            confidence=result.confidence,
            evidence=result.evidence,
            warnings=[NO_DATA],
        )
    with registry.open(result.suite) as runtime:
        found = connect(
            runtime.suite,
            result.suite,
            center,
            request=ConnectRequest(subject=center.pref_label, rel_types=rels),
            confidence=result.confidence,
            locate_evidence=result.evidence,
        )
    # Outgoing only: who X can move to, not who can move into X.
    edges = [e for e in found.edges if e.source_node_id == center.id and e.type in rels]
    edges.sort(key=_strength, reverse=True)
    keep = {e.target_node_id for e in edges}
    order = {e.target_node_id: i for i, e in enumerate(edges)}
    neighbors = sorted(
        (n for n in found.nodes[1:] if n.id in keep), key=lambda n: order.get(n.id, 0)
    )
    return found.model_copy(
        update={
            "nodes": [center, *neighbors],
            "edges": edges,
            "warnings": [*[w for w in found.warnings if w != "no_neighbors"], WARNING],
        }
    )

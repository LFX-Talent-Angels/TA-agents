"""Skill gap: what a target occupation needs beyond a start's skills (ADR-0008).

Reuses ``skills/connect/compare.py`` rather than a second set-difference
implementation — its own docstring already names the reuse: "Pathfinder's
skill gap from A to B is ``only_b``". This module adds the one thing
``compare.py`` does not do: splitting that gap into essential-missing and
optional-missing (ADR-0008 decision 3), and flagging a gap too large to call
a realistic route (decision 5) against a per-suite threshold measured from
real occupation pairs (PR 6's golden pairs set).
"""

from __future__ import annotations

from dataclasses import dataclass

from talent_angels.contracts import AgentResult, NodeRef
from talent_angels.skills.connect.compare import compare_result, skill_overlap

#: Measured against live ESCO v1.2.1 and O*NET 31.0 (ADR-0008 decision 5
#: amendment, PR 6's golden pairs set — see
#: docs/decisions/0008-pathfind-contract.md). ESCO's number rests on 6 real
#: pairs with a clean split between same-family title variants (10-11
#: essential-missing) and genuine transitions (19+); O*NET's rests on only
#: 4, noisier pairs and should be revisited first if more O*NET golden
#: pairs land. The two are not on the same scale: O*NET's HAS_SKILL and
#: USES_SOFTWARE edges include specific software tools ESCO does not model
#: as skills, so its essential-missing counts run far higher.
MAX_REALISTIC_GAP_SKILLS = 15
_MAX_REALISTIC_GAP_SKILLS_BY_SUITE: dict[str, int] = {
    "esco": MAX_REALISTIC_GAP_SKILLS,
    "onet": 60,
}

GAP_TOO_LARGE_WARNING = "gap_too_large"


@dataclass(frozen=True)
class SkillGap:
    """A target occupation's missing skills, essential-missing first."""

    essential_missing: tuple[NodeRef, ...]
    optional_missing: tuple[NodeRef, ...]
    gap_too_large: bool

    @property
    def is_empty(self) -> bool:
        return not self.essential_missing and not self.optional_missing


def _relation_by_skill_id(target: AgentResult, center_id: str) -> dict[str, object]:
    """target's HAS_SKILL edges, keyed by the skill id on the other end."""
    relation: dict[str, object] = {}
    for edge in target.edges:
        skill_id = edge.target_node_id if edge.source_node_id == center_id else edge.source_node_id
        relation[skill_id] = edge.properties.get("relation_type")
    return relation


def compute_gap(
    start: AgentResult,
    target: AgentResult,
    *,
    optional_rel_values: frozenset[str] = frozenset({"optional", "transferable"}),
) -> SkillGap:
    """The skills ``target``'s occupation needs that ``start``'s does not have.

    ``start`` and ``target`` are Connect results (HAS_SKILL) for two
    occupations in the same suite — node 0 of each is the occupation itself.
    Pass the suite's own ``SuiteSchema.optional_rel_values`` so the
    essential/optional split matches what Connect itself would call optional
    for this suite, rather than a second hardcoded default drifting from it.

    An edge with no ``relation_type`` (e.g. O*NET ``USES_SOFTWARE``) carries
    no essential/optional signal; ``connect()`` already treats that as
    "always relevant" (it survives every relation_kind filter), so this
    counts it as essential-missing rather than inventing a third bucket
    ADR-0008 does not call for.

    Ordering within each bucket is inherited from ``target``'s own node
    order, which ``connect()`` already ranks by O*NET importance where
    present — nothing here re-sorts.
    """
    overlap = skill_overlap(compare_result(start, target))
    target_center_id = target.nodes[0].id
    relation = _relation_by_skill_id(target, target_center_id)

    essential: list[NodeRef] = []
    optional: list[NodeRef] = []
    for node in overlap.only_b:
        if relation.get(node.id) in optional_rel_values:
            optional.append(node)
        else:
            essential.append(node)

    threshold = _MAX_REALISTIC_GAP_SKILLS_BY_SUITE.get(target.suite, MAX_REALISTIC_GAP_SKILLS)
    return SkillGap(
        essential_missing=tuple(essential),
        optional_missing=tuple(optional),
        gap_too_large=len(essential) > threshold,
    )

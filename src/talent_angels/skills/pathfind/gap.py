"""Skill gap: what a target occupation needs beyond a start's skills (ADR-0008).

Reuses ``skills/connect/compare.py`` rather than a second set-difference
implementation — its own docstring already names the reuse: "Pathfinder's
skill gap from A to B is ``only_b``". This module adds the things
``compare.py`` does not do: splitting that gap into essential-missing,
optional-missing and software-missing (ADR-0008 decisions 3 and PF-2), and
flagging a gap too large to call a realistic route (decision 5) two ways —
a per-suite essential-count threshold, and a coverage-ratio floor for the
cases essential-count alone misses (PF-2 amendment).
"""

from __future__ import annotations

from dataclasses import dataclass

from talent_angels.contracts import AgentResult, NodeRef
from talent_angels.skills.connect.compare import compare_result, skill_overlap

#: Edge types naming a specific tool, not a skill or knowledge area — kept
#: out of the essential/optional split (PF-2 amendment). O*NET's
#: USES_SOFTWARE routinely adds dozens to hundreds of named tools per
#: occupation (GAP F), which drowned out the real skill/knowledge gap under
#: one combined essential count: measuring live, every tested O*NET pair's
#: non-software essential/optional gap was zero, the entire count was
#: software. No other suite currently uses a rel type in this set.
SOFTWARE_REL_TYPES = frozenset({"USES_SOFTWARE"})

#: Measured against live ESCO v1.2.1 and O*NET 31.0 (ADR-0008 decision 5
#: amendment, PR 6's golden pairs set — see
#: docs/decisions/0008-pathfind-contract.md). ESCO's number rests on 6 real
#: pairs with a clean split between same-family title variants (10-11
#: essential-missing) and genuine transitions (19+); O*NET's 60 predates the
#: software split above and in practice rarely fires now — O*NET's generic
#: skill/knowledge taxonomy barely differentiates between occupations at
#: all once software is excluded (PF-2 finding), so ``gap_too_large`` for
#: O*NET is carried almost entirely by the coverage floor below, not this
#: per-suite count. Revisit both with a wider O*NET golden pairs set.
MAX_REALISTIC_GAP_SKILLS = 15
_MAX_REALISTIC_GAP_SKILLS_BY_SUITE: dict[str, int] = {
    "esco": MAX_REALISTIC_GAP_SKILLS,
    "onet": 60,
    # BLS has 17 broad skills, essential when BLS scores them 3.5+ (a median
    # of 5 per occupation). Missing 8 or more is most of an occupation's
    # profile. Provisional: re-measure with golden pairs (PF-6).
    "bls": 8,
    # JobTech marks few skills essential (direct links on 144 of 2,105
    # occupations); its group skills count as optional. Provisional, as ESCO's,
    # until golden pairs measure it (PF-6).
    "jobtech": MAX_REALISTIC_GAP_SKILLS,
}

#: Share of the target's skills (essential + optional + software) the start
#: already has, below which the gap is a full retrain regardless of the
#: essential-only count (PF-2 amendment) — catches a target with few
#: essential links but a huge optional gap (ESCO "data analyst" ->
#: "general practitioner": 10 essential-missing, under threshold, but 0%
#: coverage once optional is counted too). Deliberately conservative: only
#: the clear 0-15% cases measured so far are below it. One live pair
#: ("civil engineer" -> "construction engineer", 27%) sits close above this
#: line and was deliberately left unflagged pending a wider golden pairs set
#: — not yet confirmed either way.
MIN_REALISTIC_COVERAGE = 0.20

GAP_TOO_LARGE_WARNING = "gap_too_large"


@dataclass(frozen=True)
class SkillGap:
    """A target occupation's missing skills, essential-missing first."""

    essential_missing: tuple[NodeRef, ...]
    optional_missing: tuple[NodeRef, ...]
    #: Named tools, not skills or knowledge (SOFTWARE_REL_TYPES) — listed
    #: separately, never mixed into essential/optional (PF-2).
    software_missing: tuple[NodeRef, ...]
    #: Share of target's skills the start already has; 1.0 when the target
    #: has none to begin with (nothing to be missing).
    coverage: float
    gap_too_large: bool

    @property
    def is_empty(self) -> bool:
        return not (self.essential_missing or self.optional_missing or self.software_missing)


def _target_edges_by_skill_id(target: AgentResult, center_id: str) -> dict[str, tuple[str, object]]:
    """target's skill edges, keyed by the skill id: (edge type, relation_type)."""
    by_id: dict[str, tuple[str, object]] = {}
    for edge in target.edges:
        skill_id = edge.target_node_id if edge.source_node_id == center_id else edge.source_node_id
        by_id[skill_id] = (edge.type, edge.properties.get("relation_type"))
    return by_id


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

    A software edge (``SOFTWARE_REL_TYPES``) always goes to
    ``software_missing``, regardless of ``relation_type`` — it never carried
    essential/optional signal, just a list of named tools (PF-2). An edge
    with no ``relation_type`` that is *not* software carries no
    essential/optional signal either; ``connect()`` already treats that as
    "always relevant" (it survives every relation_kind filter), so this
    counts it as essential-missing rather than inventing a fourth bucket.

    Ordering within each bucket is inherited from ``target``'s own node
    order, which ``connect()`` already ranks by O*NET importance where
    present — nothing here re-sorts.
    """
    overlap = skill_overlap(compare_result(start, target))
    target_center_id = target.nodes[0].id
    target_edges = _target_edges_by_skill_id(target, target_center_id)

    essential: list[NodeRef] = []
    optional: list[NodeRef] = []
    software: list[NodeRef] = []
    for node in overlap.only_b:
        edge_type, relation_type = target_edges.get(node.id, (None, None))
        if edge_type in SOFTWARE_REL_TYPES:
            software.append(node)
        elif relation_type in optional_rel_values:
            optional.append(node)
        else:
            essential.append(node)

    total = len(overlap.shared) + len(overlap.only_b)
    coverage = len(overlap.shared) / total if total else 1.0

    threshold = _MAX_REALISTIC_GAP_SKILLS_BY_SUITE.get(target.suite, MAX_REALISTIC_GAP_SKILLS)
    return SkillGap(
        essential_missing=tuple(essential),
        optional_missing=tuple(optional),
        software_missing=tuple(software),
        coverage=coverage,
        gap_too_large=len(essential) > threshold or coverage < MIN_REALISTIC_COVERAGE,
    )

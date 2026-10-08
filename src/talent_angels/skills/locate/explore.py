"""Explore: turn a vague request into a pick list of titles the map confirms.

The planner turns "an engineer who builds buildings" into a search word
("engineer") and a few titles that may fit ("civil engineer", "construction
engineer", "structural engineer"). Those titles are the model's guesses, so
each is looked up: only a title the map has (by name or alias) is offered, and
the whole list is always the user's to choose from — never an automatic pick.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Protocol

from talent_angels.contracts import AgentResult, EdgeRef, NodeRef
from talent_angels.skills.connect.reveal import ConnectableSuite
from talent_angels.skills.locate.areas import Area, areas_from
from talent_angels.skills.locate.rank import classified_under_parent, group_and_sort_locate
from talent_angels.skills.locate.resolve import SearchableSuite, SearchResult, locate

#: Planner titles checked per turn; each costs one search.
MAX_CANDIDATES = 5


class ExploreSuite(SearchableSuite, ConnectableSuite, Protocol):
    """Search to find titles, neighbours to name their groups."""


class _Recording:
    """Keeps the suite's last raw answer, whose group counts locate() drops."""

    def __init__(self, suite: ExploreSuite) -> None:
        self._suite = suite
        self.last: SearchResult | None = None

    def search_nodes(self, text: str, kind: str | None = None) -> SearchResult:
        self.last = self._suite.search_nodes(text, kind=kind)
        return self.last


def _same_title(candidate: str, node: NodeRef) -> bool:
    """The map's own name or alias for ``candidate``, singular or plural."""
    wanted = candidate.casefold().strip()
    for name in (node.pref_label, *node.alt_labels):
        name = name.casefold().strip()
        if name in (wanted, f"{wanted}s") or f"{name}s" == wanted:
            return True
    return False


def _names_subject(subject: str, node: NodeRef) -> bool:
    """``subject`` as a whole word or phrase of the title or an alias.

    "SWE" names "SWE" but not "chimney sweep"; "engineer" names "civil
    engineer". A plural still counts.
    """
    pattern = re.compile(rf"(?<![\w]){re.escape(subject.casefold().strip())}s?(?![\w])")
    return any(pattern.search(name.casefold()) for name in (node.pref_label, *node.alt_labels))


def _confirmed(
    suite: ExploreSuite, suite_name: str, candidate: str, kind: str | None
) -> tuple[NodeRef, AgentResult] | None:
    found = locate(suite, suite_name, candidate, kind=kind)
    node = next((node for node in found.nodes if _same_title(candidate, node)), None)
    return (node, found) if node is not None else None


def explore(
    suite: ExploreSuite,
    suite_name: str,
    subject: str,
    candidates: Sequence[str],
    *,
    kind: str | None,
    subject_is_users: bool = True,
    group_rel_type: str | None = "CLASSIFIED_UNDER",
    group_node_kinds: frozenset[str] = frozenset({"ISCOGroup", "isco group"}),
) -> tuple[AgentResult, list[Area]] | None:
    """A pick list for a vague subject, or None when the subject is one clear title.

    ``subject_is_users`` is False when the planner wrote a title the user did
    not say ("civil engineer" for "an engineer who builds buildings"): its
    exact match is then offered first, never bound.

    The list holds the confirmed candidates first, then the subject's own
    matches; the areas are the subject's group counts, for "which area?".
    """
    recording = _Recording(suite)
    located = locate(recording, suite_name, subject, kind=kind)
    located = group_and_sort_locate(
        suite,
        located,
        subject,
        suite_name=suite_name,
        group_rel_type=group_rel_type,
        group_node_kinds=group_node_kinds,
    )
    if located.nodes and "ambiguous" not in located.warnings:
        if subject_is_users:
            return None
        candidates = [subject, *candidates]
    areas = areas_from(recording.last, suite_name)

    confirmed: list[NodeRef] = []
    evidence = list(located.evidence)
    for candidate in list(dict.fromkeys(c.strip() for c in candidates if c.strip()))[
        :MAX_CANDIDATES
    ]:
        hit = _confirmed(suite, suite_name, candidate, kind)
        if hit is None or any(node.id == hit[0].id for node in confirmed):
            continue
        confirmed.append(hit[0])
        evidence.extend(hit[1].evidence)

    if not confirmed:
        if not subject_is_users and located.nodes and "ambiguous" not in located.warnings:
            # The planner wrote this subject; its one hit is offered, not bound.
            located = located.model_copy(update={"warnings": [*located.warnings, "ambiguous"]})
        return located, areas
    # Each confirmed title under its own group heading, not the one above it.
    grouped = {edge.source_node_id for edge in located.edges}
    edges: list[EdgeRef] = []
    for node in confirmed:
        if node.id in grouped:
            continue
        parent = classified_under_parent(
            suite,
            node,
            suite_name=suite_name,
            group_rel_type=group_rel_type,
            group_node_kinds=group_node_kinds,
        )
        if parent is not None:
            edges.append(
                EdgeRef(
                    type=group_rel_type or "CLASSIFIED_UNDER",
                    suite=suite_name,
                    source_node_id=node.id,
                    target_node_id=parent.id,
                    properties={"group_label": parent.pref_label},
                )
            )
    seen = {node.id for node in confirmed}
    # With confirmed titles in hand, the subject's own matches stay only when
    # they really name it: "SWE" word-start hits ("chimney sweep") are noise.
    rest = [node for node in located.nodes if node.id not in seen and _names_subject(subject, node)]
    nodes = [*confirmed, *rest]
    kept = [w for w in located.warnings if w in ("truncated", "match_count_capped")]
    return (
        AgentResult(
            capability="locate",
            suite=suite_name,
            nodes=nodes,
            edges=[*edges, *located.edges],
            evidence=evidence,
            confidence=located.confidence,
            warnings=["ambiguous", "guided", *kept],
        ),
        areas,
    )

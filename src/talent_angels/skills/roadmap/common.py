"""Shared by roadmap and successors: cached neighbours, a lookup budget, start resolution."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from talent_angels.skills.connect.neighbourhood import resolve_start
from talent_angels.skills.connect.reveal import NeighborResult, TaxonomyNode
from talent_angels.skills.evaluate.policy import LEARN_ORDER_V1
from talent_angels.suites import SuiteRegistry

MAX_NODES = 500
MAX_CANDIDATES = 3
Json = dict[str, Any]


class Graph:
    """``get_neighbors`` with a cache. Only a call's own node carries its full properties
    (skill type, reuse level), so reading them costs one call, counted against the budget."""

    def __init__(self, suite: Any) -> None:
        self.suite = suite
        self._seen: dict[str, NeighborResult] = {}
        self._lookups = 0
        self.exhausted = False
        self.notes: set[str] = set()

    def neighbors(self, node_id: str) -> NeighborResult:
        if node_id not in self._seen:
            self._seen[node_id] = self.suite.get_neighbors(node_id)
        return self._seen[node_id]

    def node(self, node_id: str) -> TaxonomyNode | None:
        return next((n for n in self.neighbors(node_id).nodes if n.id == node_id), None)

    def props(self, node_id: str) -> Mapping[str, object] | None:
        """The node's own properties, or None once the lookup budget is spent."""
        if node_id not in self._seen:
            if self._lookups >= LEARN_ORDER_V1.lookup_budget:
                self.exhausted = True
                return None
            self._lookups += 1
        node = self.node(node_id)
        return node.properties if node is not None else {}

    @property
    def warnings(self) -> list[str]:
        spent = [f"lookup_budget:{LEARN_ORDER_V1.lookup_budget}"] if self.exhausted else []
        return [*sorted(self.notes), *spent]


@dataclass
class Side:
    """How one end (a name or node ID) resolved in one suite."""

    status: str = "not_found"  # ok | ambiguous | not_found
    node: TaxonomyNode | None = None
    candidates: list[Json] = field(default_factory=list)


def id_suite(registry: SuiteRegistry, text: str) -> str | None:
    """The suite a node ID belongs to; its text is never searched in another suite."""
    prefix = text.split(":", 1)[0] if ":" in text else ""
    return prefix if prefix in registry.available else None


def resolve_side(graph: Graph, name: str, text: str, owner: str | None, label: str = "") -> Side:
    """Resolve with the mapper's rule: a clear match only, otherwise candidates."""
    if owner is not None and owner != name:
        return Side()
    match = resolve_start(graph.suite, name, text)
    if match.id is None:
        hit = match.candidate_hits
        found = sorted(
            match.candidates,
            key=lambda c: (-hit.get(c.id, (0.0, ""))[0], c.pref_label.casefold(), c.id),
        )[:MAX_CANDIDATES]
        return Side(
            "ambiguous" if found else "not_found",
            candidates=[
                {"side": label, "id": c.id, "label": c.pref_label, "kind": c.kind} for c in found
            ],
        )
    node = graph.node(match.id)
    return Side("ok", node) if node is not None else Side()


def ref(node: TaxonomyNode) -> Json:
    return {"id": node.id, "label": node.label, "kind": node.kind, "cite": node.source_id}

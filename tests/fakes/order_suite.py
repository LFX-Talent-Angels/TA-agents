"""A graph suite that behaves like the real ones for learning order.

Like a concrete suite, only the *called* node comes back with its properties (skill type,
reuse level); its neighbours come back bare. Registered under the names ``esco``, ``onet`` ...
so the per-taxonomy rules of learn-order-v1 apply.
"""

from __future__ import annotations

import random
from typing import Any

from talent_angels.suites import SuiteRegistry
from talent_angels.suites.schema import SuiteSchema
from tests.fakes.suite import suite_factory, unopenable_factory
from tests.fakes.taxonomy import FakeCandidate, FakeEdge, FakeNode, FakeToolResult

SCHEMA = SuiteSchema(
    skill_rel_types=("HAS_SKILL",),
    optional_rel_values=frozenset({"optional", "group"}),
    group_rel_type=None,
    group_node_kinds=frozenset(),
)


def node(suite: str, key: str, kind: str, label: str, **props: object) -> FakeNode:
    return FakeNode(
        id=f"{suite}:{key}",
        kind=kind,
        label=label,
        source=suite,
        source_id=f"urn:{suite}:{key}",
        properties=dict(props),
    )


def edge(kind: str, a: FakeNode, b: FakeNode, **props: object) -> FakeEdge:
    return FakeEdge(type=kind, from_id=a.id, to_id=b.id, properties=dict(props))


class OrderSuite:
    """Serves one-hop neighbours from fixed lists; ``shuffle`` proves input-order independence."""

    suite_schema = SCHEMA

    def __init__(self, nodes: list[FakeNode], edges: list[FakeEdge], shuffle: int = 0) -> None:
        self.nodes = {n.id: n for n in nodes}
        self.edges = list(edges)
        if shuffle:
            random.Random(shuffle).shuffle(self.edges)
        self.calls: list[str] = []

    def search_nodes(self, text: str, kind: str | None = None) -> FakeToolResult:
        hits = [n for n in self.nodes.values() if text.casefold() in n.label.casefold()]
        return FakeToolResult(
            candidates=[FakeCandidate(node=n, confidence=0.9, method="contains") for n in hits]
        )

    def get_neighbors(self, node_id: str, rel_types: list[str] | None = None) -> FakeToolResult:
        self.calls.append(node_id)
        incident = [
            e
            for e in self.edges
            if node_id in (e.from_id, e.to_id) and (not rel_types or e.type in rel_types)
        ]
        ids = {node_id} | {e.from_id for e in incident} | {e.to_id for e in incident}
        bare = [
            n if n.id == node_id else FakeNode(n.id, n.kind, n.label, n.source, n.source_id)
            for n in (self.nodes[i] for i in sorted(ids) if i in self.nodes)
        ]
        return FakeToolResult(nodes=bare, edges=incident)

    def enumerate_paths(self, from_id: str, to_id: str, **_: Any) -> FakeToolResult:
        return FakeToolResult(warnings=["no_path"])  # Pathfind asks; the card needs no route


def registry(**suites: Any) -> SuiteRegistry:
    """``registry(esco=OrderSuite(...), ...)``; an Exception value is a suite that is down."""
    made = {
        name: unopenable_factory(s) if isinstance(s, Exception) else suite_factory(name, s)
        for name, s in suites.items()
    }
    return SuiteRegistry(made, default=next(iter(made)))

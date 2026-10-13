"""Pathfind routes: bridge paths between a start and a target occupation.

Wraps the suite's own ``enumerate_paths`` (ADR-0008 decision 4) — deterministic
graph traversal, no model call. Routes are returned in whatever order
``enumerate_paths`` produces them: unscored, because the only scorer that
exists (``score_paths``) is an unfinished stub, and showing a stub's number as
a real rank is exactly the invented-fact failure ``ARCHITECTURE.md`` rule 1
exists to prevent. ``score_paths`` is never called here. Ranking routes under
a named policy is Evaluate's job, a separate, later skill.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

from talent_angels.contracts import EdgeRef, NodeRef
from talent_angels.mcp.protocol import PathEnumeratingSuite, supports_enumerate_paths
from talent_angels.skills.connect.reveal import TaxonomyEdge
from talent_angels.skills.locate.resolve import TaxonomyNode

__all__ = ["BridgeRoute", "BridgeRoutes", "find_routes", "supports_enumerate_paths"]


@dataclass(frozen=True)
class BridgeRoute:
    """One start-to-target path, in ``enumerate_paths``'s own node order."""

    nodes: tuple[NodeRef, ...]
    edges: tuple[EdgeRef, ...]


@dataclass(frozen=True)
class BridgeRoutes:
    """Every route ``enumerate_paths`` returned, unscored (ADR-0008 decision 4)."""

    routes: tuple[BridgeRoute, ...]
    warnings: tuple[str, ...]

    @property
    def is_empty(self) -> bool:
        return not self.routes


def _node_ref(node: TaxonomyNode, suite: str) -> NodeRef:
    alt_labels = cast(Sequence[str], node.properties.get("alt_labels") or ())
    raw_description = node.properties.get("description")
    description = raw_description.strip() if isinstance(raw_description, str) else None
    return NodeRef(
        id=node.id,
        suite=suite,
        source=node.source,
        source_id=node.source_id,
        kind=node.kind,
        pref_label=node.label,
        alt_labels=list(alt_labels),
        description=description or None,
    )


def _edge_ref(edge: TaxonomyEdge, suite: str) -> EdgeRef:
    return EdgeRef(
        type=edge.type,
        suite=suite,
        source_node_id=edge.from_id,
        target_node_id=edge.to_id,
        properties=dict(edge.properties),
    )


def find_routes(
    suite: PathEnumeratingSuite,
    suite_name: str,
    start_occupation_id: str,
    target_occupation_id: str,
    *,
    max_depth: int = 4,
    max_paths: int = 20,
) -> BridgeRoutes:
    """Bridge routes between two occupation nodes in the same suite.

    Only call this when the start names an occupation (ADR-0008 decision 1) —
    a bare skill-ID start has no graph node to pathfind from, so routes are
    skipped upstream of this function rather than synthesizing one here.
    Check ``supports_enumerate_paths(suite)`` before calling, same as any
    other Pathfind/Evaluate suite-tool slice.

    ``max_depth``/``max_paths`` default to the suite's own values and are not
    re-tuned here. A suite warning such as ``no_path`` or
    ``endpoint_not_found`` comes back as an empty ``routes`` tuple plus that
    warning, not a silently empty list.
    """
    raw = suite.enumerate_paths(
        start_occupation_id, target_occupation_id, max_depth=max_depth, max_paths=max_paths
    )
    node_by_id = {node.id: _node_ref(node, suite_name) for node in raw.nodes}
    routes = tuple(
        BridgeRoute(
            nodes=tuple(node_by_id[node_id] for node_id in path.node_ids),
            edges=tuple(_edge_ref(edge, suite_name) for edge in path.edges),
        )
        for path in raw.paths
    )
    return BridgeRoutes(routes=routes, warnings=tuple(raw.warnings))

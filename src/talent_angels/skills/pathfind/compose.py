"""Pathfind: enumerate routes between two already-resolved nodes in one suite."""

from __future__ import annotations

from typing import Any, Protocol

from talent_angels.contracts import AgentResult, EdgeRef, EvidencePointer, NodeRef
from talent_angels.skills.connect.reveal import ConnectableSuite, _node_ref
from talent_angels.skills.locate.resolve import SearchableSuite

PATHFIND_UNIMPLEMENTED = "capability_not_implemented:pathfind"


class PathfindSuite(SearchableSuite, ConnectableSuite, Protocol):
    def enumerate_paths(
        self,
        from_id: str,
        to_id: str,
        *,
        max_depth: int = 4,
        max_paths: int = 20,
    ) -> Any: ...


def pathfind(
    suite: SearchableSuite,
    suite_name: str,
    from_node: NodeRef,
    to_node: NodeRef,
    *,
    max_depth: int = 4,
    max_paths: int = 20,
) -> AgentResult:
    """Call enumerate_paths. Endpoints must already be the same suite."""
    enumerate_paths = getattr(suite, "enumerate_paths", None)
    if enumerate_paths is None:
        return AgentResult(
            capability="pathfind",
            suite=suite_name,
            warnings=[PATHFIND_UNIMPLEMENTED],
        )

    try:
        raw = enumerate_paths(
            from_node.id,
            to_node.id,
            max_depth=max_depth,
            max_paths=max_paths,
        )
    except AttributeError:
        return AgentResult(
            capability="pathfind",
            suite=suite_name,
            warnings=[PATHFIND_UNIMPLEMENTED],
        )
    warnings = list(getattr(raw, "warnings", ()) or ())
    nodes = [_node_ref(node, suite_name) for node in getattr(raw, "nodes", ()) or ()]
    if not nodes:
        nodes = [from_node, to_node]
    edges: list[EdgeRef] = []
    for path in getattr(raw, "paths", ()) or ():
        for edge in getattr(path, "edges", ()) or ():
            edges.append(
                EdgeRef(
                    type=str(getattr(edge, "type", "") or "RELATED"),
                    suite=suite_name,
                    source_node_id=str(getattr(edge, "from_id", "")),
                    target_node_id=str(getattr(edge, "to_id", "")),
                    properties=dict(getattr(edge, "properties", {}) or {}),
                )
            )
    evidence = [
        EvidencePointer(suite=suite_name, pointer=str(item))
        for item in getattr(raw, "evidence", ()) or ()
    ]
    if not getattr(raw, "paths", None) and "no_path" not in warnings:
        if "endpoint_not_found" not in warnings and "invalid_max_depth" not in warnings:
            warnings.append("no_path")

    gap_nodes, gap_warning = _skill_gap(suite, suite_name, from_node, to_node)
    if gap_warning:
        warnings.append(gap_warning)
    # Derived gap skills are extra nodes; they are not a new taxonomy identity.
    existing = {node.id for node in nodes}
    for extra in gap_nodes:
        if extra.id not in existing:
            nodes.append(extra)
            existing.add(extra.id)

    return AgentResult(
        capability="pathfind",
        suite=suite_name,
        nodes=nodes,
        edges=edges,
        evidence=evidence,
        warnings=warnings,
    )


def _skill_gap(
    suite: SearchableSuite,
    suite_name: str,
    from_node: NodeRef,
    to_node: NodeRef,
) -> tuple[list[NodeRef], str | None]:
    if from_node.kind.casefold() != "occupation" or to_node.kind.casefold() != "occupation":
        return [], None
    get_neighbors = getattr(suite, "get_neighbors", None)
    if get_neighbors is None:
        return [], None
    start = get_neighbors(from_node.id, rel_types=["HAS_SKILL"])
    dest = get_neighbors(to_node.id, rel_types=["HAS_SKILL"])
    start_ids = {
        node.id
        for node in getattr(start, "nodes", ()) or ()
        if str(getattr(node, "kind", "")).casefold() == "skill"
    }
    extras: list[NodeRef] = []
    for node in getattr(dest, "nodes", ()) or ():
        if str(getattr(node, "kind", "")).casefold() != "skill":
            continue
        if node.id in start_ids:
            continue
        extras.append(_node_ref(node, suite_name))
    if not extras:
        return [], None
    return extras, "derived_skill_gap"

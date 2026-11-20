"""Translate suite ``ToolResult`` values into the typed MCP payloads.

One direction only: suite shapes in, ``talent_angels.mcp.models`` out. Nothing
here interprets, ranks, or summarises — the suite already decided all of that
in code (ARCHITECTURE.md: determinism is pushed down), and the client decides
what to say about it.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import cast

from talent_angels.contracts import EdgeRef, EvidencePointer, NodeRef
from talent_angels.mcp.models import (
    CandidateRef,
    NeighborsPayload,
    PathRef,
    PathsPayload,
    PolicyRef,
    PruningRef,
    ScoredPathRef,
    ScoredPathsPayload,
    SearchPayload,
)
from talent_angels.mcp.protocol import (
    PathsResult,
    ScoredPathsResult,
    TaxonomyEdge,
    TaxonomyNode,
    TaxonomyPath,
)
from talent_angels.skills.connect.reveal import NeighborResult
from talent_angels.skills.locate.resolve import SearchResult

__all__ = [
    "edge_ref",
    "neighbors_payload",
    "node_ref",
    "path_ref",
    "paths_payload",
    "scored_paths_payload",
    "search_payload",
    "suite_path_arg",
]


def node_ref(node: TaxonomyNode, suite: str) -> NodeRef:
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


def edge_ref(edge: TaxonomyEdge, suite: str) -> EdgeRef:
    return EdgeRef(
        type=edge.type,
        suite=suite,
        source_node_id=edge.from_id,
        target_node_id=edge.to_id,
        properties=dict(edge.properties),
    )


def path_ref(path: TaxonomyPath, suite: str) -> PathRef:
    return PathRef(
        node_ids=list(path.node_ids),
        edges=[edge_ref(edge, suite) for edge in path.edges],
    )


def suite_path_arg(path: PathRef) -> dict[str, object]:
    """Turn a route back into the contract's own path shape.

    ``score_paths`` is the one tool whose input is another tool's output, so
    the client round-trips ``enumerate_paths`` results through us verbatim.
    Plain dicts keep this edge free of a concrete-library import; the suite
    validates them into its own model.
    """
    return {
        "node_ids": list(path.node_ids),
        "edges": [
            {
                "type": edge.type,
                "from_id": edge.source_node_id,
                "to_id": edge.target_node_id,
                "properties": dict(edge.properties),
            }
            for edge in path.edges
        ],
    }


def _evidence(pointers: Iterable[str], suite: str) -> list[EvidencePointer]:
    return [EvidencePointer(suite=suite, pointer=pointer) for pointer in pointers]


def search_payload(
    result: SearchResult, *, suite: str, query: str, kind: str | None
) -> SearchPayload:
    return SearchPayload(
        suite=suite,
        query=query,
        kind=kind,
        candidates=[
            CandidateRef(
                node=node_ref(candidate.node, suite),
                confidence=candidate.confidence,
                method=candidate.method,
            )
            for candidate in result.candidates
        ],
        evidence=_evidence(result.evidence, suite),
        warnings=list(result.warnings),
    )


def _edge_rank(edge: object) -> tuple[float, int]:
    props = getattr(edge, "properties", None) or {}
    importance = props.get("importance")
    weight = -float(importance) if isinstance(importance, int | float) else 0.0
    flagged = props.get("hot_technology") == "Y" or props.get("in_demand") == "Y"
    return (weight, 0 if flagged else 1)


def neighbors_payload(
    result: NeighborResult, *, suite: str, center_id: str, limit: int | None = None
) -> NeighborsPayload:
    """Map one hop; with ``limit``, keep the top neighbours and count the rest.

    Ranking is deterministic (O*NET importance, then hot/in-demand technology,
    then suite order), and what is cut is reported as counts, not dropped
    silently — ARCHITECTURE "truncate at the tool edge". Unbounded, an O*NET
    occupation's hop was ~1.4 MB in one tool result.
    """
    edges = list(result.edges)
    neighbour_ids: list[str] = []
    for edge in sorted(edges, key=_edge_rank):
        other = edge.to_id if edge.from_id == center_id else edge.from_id
        if other not in neighbour_ids:
            neighbour_ids.append(other)
    pruning = None
    warnings = list(result.warnings)
    if limit is not None and len(neighbour_ids) > limit:
        kept = set(neighbour_ids[:limit])
        edges = [e for e in edges if (e.to_id if e.from_id == center_id else e.from_id) in kept]
        pruning = PruningRef(
            considered=len(neighbour_ids), returned=limit, pruned=len(neighbour_ids) - limit
        )
        warnings.append(f"truncated:{len(neighbour_ids) - limit}")
    keep_nodes = {center_id} | {e.to_id for e in edges} | {e.from_id for e in edges}
    nodes = [n for n in result.nodes if n.id in keep_nodes or pruning is None]
    order = {nid: i for i, nid in enumerate(neighbour_ids)}
    nodes.sort(key=lambda n: -1 if n.id == center_id else order.get(n.id, len(order)))
    return NeighborsPayload(
        suite=suite,
        center_id=center_id,
        nodes=[node_ref(node, suite) for node in nodes],
        edges=[edge_ref(edge, suite) for edge in sorted(edges, key=_edge_rank)],
        evidence=_evidence(result.evidence, suite),
        warnings=warnings,
        pruning=pruning,
    )


def paths_payload(result: PathsResult, *, suite: str, from_id: str, to_id: str) -> PathsPayload:
    pruning = result.pruning
    return PathsPayload(
        suite=suite,
        from_id=from_id,
        to_id=to_id,
        nodes=[node_ref(node, suite) for node in result.nodes],
        paths=[path_ref(path, suite) for path in result.paths],
        pruning=(
            None
            if pruning is None
            else PruningRef(
                considered=pruning.considered,
                returned=pruning.returned,
                pruned=pruning.pruned,
            )
        ),
        evidence=_evidence(result.evidence, suite),
        warnings=list(result.warnings),
    )


def scored_paths_payload(
    result: ScoredPathsResult, *, suite: str, policy: PolicyRef
) -> ScoredPathsPayload:
    return ScoredPathsPayload(
        suite=suite,
        policy=policy,
        scored_paths=[
            ScoredPathRef(
                path=path_ref(scored.path, suite),
                score=scored.score,
                # The policy the suite reports back, not the one we asked for:
                # a suite may answer under a different declared version.
                policy=PolicyRef.model_validate(scored.policy, from_attributes=True),
            )
            for scored in result.scored_paths
        ],
        evidence=_evidence(result.evidence, suite),
        warnings=list(result.warnings),
    )

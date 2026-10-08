"""Map a suite's deterministic one-hop result to the agent contract."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol, cast

from talent_angels.contracts import AgentResult, EdgeRef, EvidencePointer, NodeRef
from talent_angels.skills.connect.models import ConnectRequest


class TaxonomyNode(Protocol):
    @property
    def id(self) -> str: ...

    @property
    def kind(self) -> str: ...

    @property
    def label(self) -> str: ...

    @property
    def source(self) -> str: ...

    @property
    def source_id(self) -> str: ...

    @property
    def properties(self) -> Mapping[str, object]: ...


class TaxonomyEdge(Protocol):
    @property
    def type(self) -> str: ...

    @property
    def from_id(self) -> str: ...

    @property
    def to_id(self) -> str: ...

    @property
    def properties(self) -> Mapping[str, object]: ...


class NeighborResult(Protocol):
    @property
    def nodes(self) -> Sequence[TaxonomyNode]: ...

    @property
    def edges(self) -> Sequence[TaxonomyEdge]: ...

    @property
    def evidence(self) -> Sequence[str]: ...

    @property
    def warnings(self) -> Sequence[str]: ...


class ConnectableSuite(Protocol):
    """The suite-contract slice required by Connect."""

    def get_neighbors(self, node_id: str, rel_types: list[str] | None = None) -> NeighborResult: ...


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


def connect(
    suite: ConnectableSuite,
    suite_name: str,
    center: NodeRef,
    *,
    request: ConnectRequest,
    confidence: float | None,
    locate_evidence: Sequence[EvidencePointer],
    optional_rel_values: frozenset[str] = frozenset({"optional", "transferable"}),
) -> AgentResult:
    """Return direct graph neighbors of an already-resolved center node."""
    rel_types = list(request.rel_types) or None
    raw = suite.get_neighbors(center.id, rel_types=rel_types)
    edges = list(raw.edges)
    if request.relation_kind is not None:
        edges = [
            edge
            for edge in edges
            if edge.properties.get("relation_type") == request.relation_kind
            or (
                request.relation_kind == "optional"
                and edge.properties.get("relation_type") in optional_rel_values
            )
            # Edges with no relation_type carry no essential/optional signal
            # (e.g. O*NET USES_SOFTWARE) — always include them in any filter.
            or edge.properties.get("relation_type") is None
        ]

    kept_ids = {center.id}
    for edge in edges:
        kept_ids.update((edge.from_id, edge.to_id))

    # Order skill nodes so tech-tool rel_types (e.g. USES_SOFTWARE) appear before
    # generic ones (e.g. HAS_SKILL). The last rel_type in request.rel_types gets
    # priority 0, so its skills surface first in the compacted LLM context window.
    _rel_priority: dict[str, int] = {rt: i for i, rt in enumerate(reversed(request.rel_types))}
    node_rel_priority: dict[str, int] = {}
    for edge in edges:
        nid = edge.to_id if edge.from_id == center.id else edge.from_id
        pri = _rel_priority.get(edge.type, len(request.rel_types))
        if nid not in node_rel_priority or pri < node_rel_priority[nid]:
            node_rel_priority[nid] = pri

    nodes = [center]
    skill_nodes = sorted(
        (
            _node_ref(node, suite_name)
            for node in raw.nodes
            if node.id != center.id and node.id in kept_ids
        ),
        key=lambda n: node_rel_priority.get(n.id, len(request.rel_types)),
    )
    nodes.extend(skill_nodes)

    mapped_edges = [
        EdgeRef(
            type=edge.type,
            suite=suite_name,
            source_node_id=edge.from_id,
            target_node_id=edge.to_id,
            properties=dict(edge.properties),
        )
        for edge in edges
    ]
    warnings = list(raw.warnings)
    if request.relation_kind is not None and raw.edges and not mapped_edges:
        warnings.append("no_matching_neighbors")
    evidence = list(locate_evidence)
    evidence.extend(EvidencePointer(suite=suite_name, pointer=pointer) for pointer in raw.evidence)

    return AgentResult(
        capability="connect",
        suite=suite_name,
        nodes=nodes,
        edges=mapped_edges,
        evidence=evidence,
        confidence=confidence,
        warnings=warnings,
    )

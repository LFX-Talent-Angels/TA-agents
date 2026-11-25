"""Typed payloads the MCP tools return — data only, never prose.

The MCP client (Claude Code) plays the part the main assistant plays inside
this repo: it owns the goal and writes the answer. So this edge asserts facts
and nothing else — IDs, labels, relationship types, confidence, warnings —
and leaves every sentence to the client (ARCHITECTURE.md rules #2 and #8).

Node and edge shapes are reused from ``talent_angels.contracts`` so the MCP
edge and the skills describe a graph the same way. ``CandidateRef`` is the one
addition: it keeps the suite's ``method`` next to the confidence, which
``AgentResult`` drops because it carries a single turn-level confidence.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_serializer

from talent_angels.contracts import EdgeRef, EvidencePointer, NodeRef

__all__ = [
    "CandidateRef",
    "DiscoverPayload",
    "MapPayload",
    "NeighborsPayload",
    "OrderPayload",
    "PathRef",
    "PathsPayload",
    "PolicyRef",
    "PruningRef",
    "ScoredPathRef",
    "ScoredPathsPayload",
    "SearchPayload",
]


class CandidateRef(BaseModel):
    """One search hit: the node, how confident, and how it was matched."""

    node: NodeRef
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description=(
            "How the match was made, expressed as a number — not a probability "
            "that the node is the right answer. The suite assigns one fixed "
            "value per match method."
        ),
    )
    method: str = Field(
        ...,
        description="Match method: exact_pref | exact_alt | casefold_pref | contains | ...",
    )


class PathRef(BaseModel):
    """An ordered route: node IDs plus the edges that connect them."""

    node_ids: list[str] = Field(..., min_length=1)
    edges: list[EdgeRef] = Field(default_factory=list)


class PruningRef(BaseModel):
    """What bounded enumeration cut, as counts — never the discarded rows."""

    considered: int = Field(..., ge=0)
    returned: int = Field(..., ge=0)
    pruned: int = Field(..., ge=0)


class PolicyRef(BaseModel):
    """Stable identity of a declared scoring policy (never a free invention)."""

    name: str = Field(..., min_length=1)
    version: str = Field(..., min_length=1)


class ScoredPathRef(BaseModel):
    """A path score together with the named policy that produced it."""

    path: PathRef
    score: float
    policy: PolicyRef


class _Payload(BaseModel):
    """Fields every tool answer carries, whatever the tool did."""

    suite: str = Field(..., description="Suite the IDs below are scoped to.")
    evidence: list[EvidencePointer] = Field(
        default_factory=list,
        description="Citation pointers to the graph facts — pointers, not licensed payloads.",
    )
    warnings: list[str] = Field(
        default_factory=list,
        description=(
            "Machine-readable conditions the client must not paper over: "
            "not_found, ambiguous, no_neighbors, node_not_found, no_path, "
            "suite_unavailable:<error>, capability_unavailable:<tool>, ..."
        ),
    )


class _Sparse(BaseModel):
    """Optional fields that do not apply to this answer are left out, not sent as null."""

    @model_serializer(mode="wrap")
    def _drop_none(self, handler: Any) -> dict[str, Any]:
        data: dict[str, Any] = handler(self)
        return {k: v for k, v in data.items() if v is not None or k == "strength"}


class MapNodeRef(_Sparse):
    id: str
    label: str
    kind: str
    level: int = Field(..., description="Hops from the start; 0 is the start itself.")
    on_route: bool | None = Field(default=None, description="Route mode only.")
    stop: int | None = Field(default=None, description="Route mode: index along the route.")
    suite: str | None = Field(default=None, description="All-taxonomy maps: the node's suite.")
    cite: str | None = Field(default=None, description="All-taxonomy maps: its source id.")
    confirmed_by: list[str] | None = Field(
        default=None, description="Suites an official crosswalk links this node to."
    )


class MapEdgeRef(_Sparse):
    from_id: str
    to_id: str
    relation: str = Field(..., description="parent | narrower | related | level | needs | ...")
    graph_edge: str
    strength: str | None = None
    suite: str | None = Field(default=None, description="All-taxonomy maps: owning suite.")
    source: str | None = Field(default=None, description="Crosswalk links: the code source.")


class MapPrunedRef(BaseModel):
    node: str
    considered: int
    kept: int


class MapCandidateRef(BaseModel):
    id: str
    label: str
    kind: str


class MapStartRef(BaseModel):
    id: str
    label: str


class MapPayload(_Payload):
    """map_neighbors: a deterministic multi-hop neighbourhood of one node."""

    start: MapStartRef | None = Field(default=None, description="Resolved start: id, label.")
    policy: str | None = Field(default=None, description="Ranking policy, e.g. map-rank-v1.")
    query: str | None = Field(default=None, description="All-taxonomy maps: the question asked.")
    suites: dict[str, Any] | None = Field(
        default=None, description="All-taxonomy maps: per-suite status, start, candidates, via."
    )
    summary: dict[str, int] | None = Field(default=None, description="All-taxonomy map totals.")
    route: list[str] = Field(default_factory=list, description="Route mode: stop IDs in order.")
    depth: int
    breadth: int
    meaning: str = "linked in the graph; not a learning order"
    nodes: list[MapNodeRef] = Field(default_factory=list)
    edges: list[MapEdgeRef] = Field(default_factory=list)
    pruned: list[MapPrunedRef] = Field(default_factory=list)
    candidates: list[MapCandidateRef] = Field(
        default_factory=list, description="Pick list, only with `ambiguous_start`."
    )


class DiscoverPayload(BaseModel):
    """discover_map: the discovery map of a list of items (the ``discover-v1`` JSON, as is)."""

    model_config = ConfigDict(extra="allow")

    policy: str = "discover-v1"
    meaning: str = "linked in the graph; not a learning order"
    warnings: list[str] = Field(default_factory=list)


class OrderPayload(BaseModel):
    """roadmap and successors: the ``learn-order-v1`` JSON, as is."""

    model_config = ConfigDict(extra="allow")

    policy: str = "learn-order-v1"
    meaning: str = (
        "ordered by learn-order-v1 over official taxonomy data; only items marked "
        "stated_prerequisite are prerequisites stated by a taxonomy"
    )
    warnings: list[str] = Field(default_factory=list)


class PathfindPayload(BaseModel):
    """pathfind: skill gap and ranked routes between two occupations, per taxonomy, as is."""

    model_config = ConfigDict(extra="allow")

    meaning: str = "skill gap and routes from the taxonomy graph; not advice"
    warnings: list[str] = Field(default_factory=list)


class SearchPayload(_Payload):
    """search_nodes: free text resolved to candidate nodes."""

    query: str
    kind: str | None = None
    candidates: list[CandidateRef] = Field(default_factory=list)


class NeighborsPayload(_Payload):
    """get_neighbors: one deterministic hop around a resolved node."""

    center_id: str
    nodes: list[NodeRef] = Field(default_factory=list)
    edges: list[EdgeRef] = Field(default_factory=list)
    #: considered / returned / pruned neighbour counts when ``limit`` cut the hop.
    pruning: PruningRef | None = None


class PathsPayload(_Payload):
    """enumerate_paths: depth-capped, cycle-free routes between two nodes."""

    from_id: str
    to_id: str
    nodes: list[NodeRef] = Field(default_factory=list)
    paths: list[PathRef] = Field(default_factory=list)
    pruning: PruningRef | None = None


class ScoredPathsPayload(_Payload):
    """score_paths: routes ranked under a named, versioned policy."""

    policy: PolicyRef
    scored_paths: list[ScoredPathRef] = Field(default_factory=list)

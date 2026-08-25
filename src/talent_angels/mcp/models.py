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

from pydantic import BaseModel, Field

from talent_angels.contracts import EdgeRef, EvidencePointer, NodeRef

__all__ = [
    "CandidateRef",
    "NeighborsPayload",
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

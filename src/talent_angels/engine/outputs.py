"""Typed output models: they describe each capability's JSON, they never produce it.

`run()` returns the skill's own dict, so these models sit on no serialization path and the bytes
cannot change. They give the manifest its output JSON Schemas, and the tests validate real
outputs against them with `extra="forbid"`, so a key nobody described fails a test (drift guard).
shortcut: outputs are validated in tests only; add opt-in runtime validation if an integrator
reports drift.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class _Out(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


# --- map (FederatedMap.to_dict) ---------------------------------------------------------
class StartRef(_Out):
    id: str
    label: str


class MapCandidate(_Out):
    id: str
    label: str
    kind: str


class Pruned(_Out):
    node: str
    considered: int
    kept: int


class MapRoute(_Out):
    node_ids: list[str]
    score: float
    reasons: list[str]


class MapSuite(_Out):
    status: str = Field(
        description="mapped | ambiguous | not_found | not_searched | via_crosswalk | no_route | "
        "error"
    )
    start: StartRef | None
    candidates: list[MapCandidate]
    via: str | None
    nodes: int
    pruned: list[Pruned]
    warnings: list[str]
    route: MapRoute | None = None
    side: str | None = None


class MapNode(_Out):
    id: str
    label: str
    kind: str
    level: int
    suite: str
    cite: str
    on_route: bool | None = None
    stop: int | None = None
    confirmed_by: list[str] | None = None


class MapEdge(_Out):
    from_id: str
    to_id: str
    relation: str
    graph_edge: str
    strength: str | None
    suite: str | None = None
    source: str | None = None


class MapSummary(_Out):
    suites_mapped: int
    nodes: int
    edges: int
    crosswalk_links: int
    confirmed_nodes: int


class MapOutput(_Out):
    query: str
    policy: str
    meaning: str
    depth: int
    breadth: int
    suites: dict[str, MapSuite]
    nodes: list[MapNode]
    edges: list[MapEdge]
    summary: MapSummary
    warnings: list[str]


# --- discover ---------------------------------------------------------------------------
class DiscoverParameters(_Out):
    depth: int
    breadth: int
    kinds: list[str] | None
    max_nodes: int
    nodes_per_item: int
    suites: list[str]
    max_route_pairs: int
    indirect_max_edges: int


class DiscoverCandidate(_Out):
    id: str
    label: str
    kind: str
    confidence: float | None
    method: str | None


class DiscoverItemOut(_Out):
    item: str
    kind: str | None
    related_to: list[str]
    status: Literal["resolved", "unclear", "not_found"]
    resolved_in: list[str]
    unclear_in: list[str]
    not_in_any_taxonomy: bool
    connected: bool


class ResolvedEntry(_Out):
    item: str
    suite: str
    status: Literal["resolved", "unclear"]
    id: str | None = None
    label: str | None = None
    kind: str | None = None
    confidence: float | None = None
    cite: str | None = None
    candidates: list[DiscoverCandidate] | None = None


class GuessCounts(_Out):
    confirmed: int
    indirect: int
    not_in_graph: int
    unresolved: int


class DiscoverSummary(_Out):
    items: int
    resolved_items: int
    not_in_any_taxonomy: int
    nodes: int
    edges: int
    links: int
    connectors: int
    broad_connectors: int
    routes: int
    route_pairs_skipped: int
    orphans: int
    clusters: int
    guesses: GuessCounts


class SuiteState(_Out):
    status: str = Field(description="ok | error:<Type>")


class Link(_Out):
    a: str
    b: str
    a_id: str
    b_id: str
    relation: str
    graph_edge: str
    strength: str | None
    suite: str | None


class ConnectorItem(_Out):
    item: str
    level: int


class Connector(_Out):
    id: str
    label: str
    kind: str
    suite: str
    cite: str
    broad: bool
    item_count: int
    items: list[ConnectorItem]
    tool_only: bool  # reached only through tool edges (ADR-0012); ranked after the rest


class Stop(_Out):
    id: str
    label: str
    kind: str


class DiscoverRoute(_Out):
    a: str
    b: str
    suite: str
    edges: int
    score: float | None
    reasons: list[str]
    stops: list[Stop]
    policy: str


class Orphan(_Out):
    item: str
    routes_fully_checked: bool


class Cluster(_Out):
    name: str
    size: int
    items: list[str]
    connectors: list[str]


class Guess(_Out):
    item: str
    related_to: str
    verdict: Literal["confirmed", "indirect", "not_in_graph", "unresolved"]
    detail: str


class UnclearGap(_Out):
    item: str
    candidates: list[DiscoverCandidate]


class Gap(_Out):
    unclear: list[UnclearGap]
    not_found: list[str]
    status: str


class DiscoverNode(_Out):
    id: str
    label: str
    kind: str
    suite: str
    cite: str
    items: list[str]


class DiscoverOutput(_Out):
    policy: str
    map_policy: str
    route_policy: str
    meaning: str
    parameters: DiscoverParameters
    versions: dict[str, str]
    input_notes: list[str]
    items: list[DiscoverItemOut]
    resolved: list[ResolvedEntry]
    summary: DiscoverSummary
    suites: dict[str, SuiteState]
    links: list[Link]
    connectors: list[Connector]
    routes: list[DiscoverRoute]
    orphans: list[Orphan]
    clusters: list[Cluster]
    guesses: list[Guess]
    gaps: dict[str, Gap]
    nodes: list[DiscoverNode]
    edges: list[MapEdge]
    warnings: list[str]


# --- roadmap ----------------------------------------------------------------------------
class NodeCite(_Out):
    id: str
    label: str
    kind: str
    cite: str


class SideCandidate(_Out):
    side: str = Field(description="from | to | start")
    id: str
    label: str
    kind: str


class RoadmapSuite(_Out):
    status: str = Field(description="ok | ambiguous | not_found | no_order_data | error:<Type>")
    start: NodeCite | None
    target: NodeCite | None
    candidates: list[SideCandidate]
    items: int
    warnings: list[str]


class Levels(_Out):
    from_: float | None = Field(alias="from")
    to: float
    jump: float
    importance: float


class RoadmapItem(_Out):
    suite: str
    id: str
    label: str
    kind: str
    cite: str
    step: int
    reasons: list[str]
    stated_prerequisite: bool
    levels: Levels | None = None


class Step(_Out):
    step: int
    name: str
    items: list[RoadmapItem]
    more: int


class Credentials(_Out):
    suite: str
    id: str
    label: str
    kind: str
    cite: str
    entry_education: Any = None
    work_experience: Any = None
    job_training: Any = None


class ZoneRef(NodeCite):
    zone: int | None


class JobZone(_Out):
    from_: ZoneRef | None = Field(alias="from")
    to: ZoneRef | None


class RoadmapOutput(_Out):
    policy: str
    meaning: str
    from_: str = Field(alias="from")
    to: str
    per_step: int
    suites: dict[str, RoadmapSuite]
    steps: list[Step]
    credentials: list[Credentials]
    job_zone: JobZone | None
    warnings: list[str]


# --- successors -------------------------------------------------------------------------
class TreeNode(_Out):
    id: str
    label: str
    kind: str
    cite: str
    depth: int
    reuse_level: str | None


class TreeEdge(_Out):
    from_id: str
    to_id: str
    kind: Literal["stated", "policy"]
    reason: str


class SuccessorsSuite(_Out):
    status: str
    start: StartRef | None
    candidates: list[SideCandidate]
    warnings: list[str]
    nodes: list[TreeNode] | None = None
    edges: list[TreeEdge] | None = None
    pruned: list[Pruned] | None = None


class SuccessorsOutput(_Out):
    policy: str
    meaning: str
    x: str
    depth: int
    breadth: int
    max_nodes: int
    suites: dict[str, SuccessorsSuite]
    warnings: list[str]


# --- pathfind ---------------------------------------------------------------------------
class Brief(_Out):
    id: str
    label: str


class GapOut(_Out):
    essential: list[Brief]
    optional: list[Brief]
    software: list[Brief]


class PathRoute(_Out):
    node_ids: list[str]
    labels: list[str]
    score: float | None


class OrderItem(_Out):
    label: str
    stated: bool


class OrderStep(_Out):
    step: int
    items: list[OrderItem]
    more: int


class PathfindSuite(_Out):
    status: str = Field(
        description="ok | ambiguous | not_found | same_start_and_target | error:<Type>"
    )
    warnings: list[str]
    side: str | None = None
    candidates: list[Brief] | None = None
    start: Brief | None = None
    target: Brief | None = None
    confidence: float | None = None
    gap: GapOut | None = None
    gap_too_large: bool | None = None
    routes: list[PathRoute] | None = None
    learning_order: list[OrderStep] | None = None


class PathfindOutput(_Out):
    start: str
    target: str
    suites: dict[str, PathfindSuite]
    meaning: str

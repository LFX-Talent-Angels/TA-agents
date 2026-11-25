"""Graph-Mapper core: a deterministic multi-hop walk over Connect's one-hop neighbors.

Each expanded node asks the suite for its direct neighbors (one Connect call),
ranks them by a fixed rule, keeps the top ``breadth`` not yet mapped, and repeats
from those for ``depth`` levels. No model call: same suite data in, same bytes out.
Hub kinds (groups, levels, categories) are shown but never expanded, so the walk
does not drift from a skill out to its whole group.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from talent_angels.contracts import EdgeRef, NodeRef
from talent_angels.skills.connect.crosswalk import crosswalk_props
from talent_angels.skills.connect.reveal import NeighborResult, TaxonomyEdge, TaxonomyNode
from talent_angels.skills.evaluate.neighbours import rerank_neighbours
from talent_angels.skills.evaluate.policy import MAP_RANK_V1
from talent_angels.skills.locate import SearchableSuite, locate

MAX_DEPTH = 4
# Locate's casefold-unique tier; exact preferred/alternative labels score higher.
AUTO_PICK_CONFIDENCE = 0.85
MAX_BREADTH = 10
DEFAULT_MAX_NODES = 500
MEANING = "linked in the graph; not a learning order"

# Shown in the map, never expanded further.
HUB_KINDS: frozenset[str] = frozenset(
    {
        "SkillGroup",
        "SkillLevel",
        "Subcategory",
        "Category",
        "ISCOGroup",
        "SSYKGroup",
        "SkillHeadline",
    }
)

# Plain-language relation labels, from the edge type and its direction.
_RELATION_RANK = {"related": 0, "narrower": 1, "level": 2, "parent": 3, "needs": 4, "needed_by": 4}
_STRENGTH = {"essential": 0, "optional": 1, "transferable": 1}
_UNTYPED = 2


class MapperNeighbors(Protocol):
    """The one-hop slice the walker needs from a suite."""

    def get_neighbors(self, node_id: str, rel_types: list[str] | None = None) -> NeighborResult: ...


@dataclass(frozen=True)
class MapNode:
    ref: NodeRef
    level: int


@dataclass(frozen=True)
class MapEdge:
    ref: EdgeRef
    relation: str  # parent | narrower | related | level | needs | needed_by | group | ...
    strength: str | None  # essential | optional | None when the graph does not say


@dataclass(frozen=True)
class Pruned:
    node_id: str
    considered: int
    kept: int


def _node_dict(n: MapNode) -> dict[str, object]:
    return {"id": n.ref.id, "label": n.ref.pref_label, "kind": n.ref.kind, "level": n.level}


def _edge_dict(e: MapEdge) -> dict[str, object]:
    return {
        "from_id": e.ref.source_node_id,
        "to_id": e.ref.target_node_id,
        "relation": e.relation,
        "graph_edge": e.ref.type,
        "strength": e.strength,
    }


def _pruned_dict(p: Pruned) -> dict[str, object]:
    return {"node": p.node_id, "considered": p.considered, "kept": p.kept}


@dataclass
class Neighbourhood:
    suite: str
    start_id: str
    depth: int
    breadth: int
    start_label: str = ""
    candidates: list[NodeRef] = field(default_factory=list)  # set only when the start is ambiguous
    nodes: list[MapNode] = field(default_factory=list)
    edges: list[MapEdge] = field(default_factory=list)
    pruned: list[Pruned] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    xprops: dict[str, dict[str, object]] = field(default_factory=dict)  # crosswalk inputs

    def to_dict(self) -> dict[str, object]:
        """Plain JSON-ready form; field order and list order are fixed."""
        return {
            "suite": self.suite,
            "policy": MAP_RANK_V1.name,
            "start": {"id": self.start_id, "label": self.start_label},
            "depth": self.depth,
            "breadth": self.breadth,
            "meaning": MEANING,
            "nodes": [_node_dict(n) for n in self.nodes],
            "edges": [_edge_dict(e) for e in self.edges],
            "pruned": [_pruned_dict(p) for p in self.pruned],
            "warnings": list(self.warnings),
            "candidates": [
                {"id": c.id, "label": c.pref_label, "kind": c.kind} for c in self.candidates
            ],
        }


@dataclass(frozen=True)
class _Candidate:
    node: TaxonomyNode
    relation: str
    strength: str | None
    key: tuple[int, int, float, int, str, str]


def _relation(edge: TaxonomyEdge, centre_id: str) -> str:
    outgoing = edge.from_id == centre_id
    if edge.type == "BROADER_THAN":
        return "parent" if outgoing else "narrower"
    if edge.type == "RELATED_TO":
        return "related"
    if edge.type == "HAS_LEVEL":
        return "level"
    if edge.type == "HAS_SKILL":
        return "needs" if outgoing else "needed_by"
    if edge.type == "CLASSIFIED_UNDER":
        return "group"
    return edge.type.lower()


def _strength(edge: TaxonomyEdge) -> str | None:
    value = edge.properties.get("relation_type")
    return value if isinstance(value, str) else None


def _rank_key(
    edge: TaxonomyEdge, node: TaxonomyNode, relation: str
) -> tuple[int, int, float, int, str, str]:
    """Lower sorts first: relation, strength, O*NET weight, hot tech, label, id.

    O*NET weight is `importance`, else the related-occupation `index` (1 = closest).

    Relation comes first so skill-to-skill links beat job links. Strength first
    would fill the map with jobs (every job is essential or optional for a skill).
    """
    strength = _strength(edge)
    importance = edge.properties.get("importance")
    index = edge.properties.get("index")  # O*NET related occupations: 1 = most related
    if isinstance(importance, int | float):
        weight = -float(importance)
    elif isinstance(index, int | float):
        weight = float(index) - 1000.0  # any ranked link before unranked ones
    else:
        weight = 0.0
    flagged = (
        edge.properties.get("hot_technology") == "Y" or edge.properties.get("in_demand") == "Y"
    )
    return (
        _RELATION_RANK.get(relation, 9),
        _STRENGTH.get(strength or "", _UNTYPED),
        weight,
        0 if flagged else 1,
        node.label.casefold(),
        node.id,
    )


def _candidates(raw: NeighborResult, centre_id: str) -> dict[str, _Candidate]:
    """Best edge per neighbor, so a node reached two ways is ranked once."""
    labels = {n.id: n for n in raw.nodes}
    best: dict[str, _Candidate] = {}
    for edge in raw.edges:
        other_id = edge.to_id if edge.from_id == centre_id else edge.from_id
        node = labels.get(other_id)
        if node is None or other_id == centre_id:
            continue
        relation = _relation(edge, centre_id)
        cand = _Candidate(
            node=node,
            relation=relation,
            strength=_strength(edge),
            key=_rank_key(edge, node, relation),
        )
        current = best.get(other_id)
        if current is None or cand.key < current.key:
            best[other_id] = cand
    return best


def _node_ref(node: TaxonomyNode, suite: str) -> NodeRef:
    return NodeRef(
        id=node.id,
        suite=suite,
        source=node.source,
        source_id=node.source_id,
        kind=node.kind,
        pref_label=node.label,
    )


def _check_limits(depth: int, breadth: int) -> None:
    if not 1 <= depth <= MAX_DEPTH:
        raise ValueError(f"depth must be 1..{MAX_DEPTH}, got {depth}")
    if not 1 <= breadth <= MAX_BREADTH:
        raise ValueError(f"breadth must be 1..{MAX_BREADTH}, got {breadth}")


def map_neighbourhood(
    suite: MapperNeighbors,
    suite_name: str,
    start_id: str,
    *,
    depth: int,
    breadth: int,
    kinds: Iterable[str] | None = None,
    max_nodes: int = DEFAULT_MAX_NODES,
) -> Neighbourhood:
    """Walk ``depth`` levels from ``start_id``, keeping the top ``breadth`` per node."""
    _check_limits(depth, breadth)
    allowed = frozenset(k.casefold() for k in kinds) if kinds is not None else None

    result = Neighbourhood(suite=suite_name, start_id=start_id, depth=depth, breadth=breadth)
    fetched: dict[str, NeighborResult] = {}  # shared by degree lookups and expansion
    degree_lookups = 0

    def neighbors(node_id: str) -> NeighborResult:
        if node_id not in fetched:
            fetched[node_id] = suite.get_neighbors(node_id)
        return fetched[node_id]

    def degree_of(node_id: str) -> int | None:
        nonlocal degree_lookups
        if node_id not in fetched:
            if degree_lookups >= MAP_RANK_V1.degree_budget:
                if f"rank_budget:{MAP_RANK_V1.degree_budget}" not in result.warnings:
                    result.warnings.append(f"rank_budget:{MAP_RANK_V1.degree_budget}")
                return None
            degree_lookups += 1
        return len(neighbors(node_id).edges)

    first = neighbors(start_id)
    centre = next((n for n in first.nodes if n.id == start_id), None)
    if centre is None:
        result.warnings.append("start_not_found")
        return result
    result.start_label = centre.label
    mapped: dict[str, MapNode] = {start_id: MapNode(_node_ref(centre, suite_name), 0)}
    result.nodes.append(mapped[start_id])
    seen_edges: set[tuple[str, str, str]] = set()
    frontier: list[tuple[str, NeighborResult]] = [(start_id, first)]
    expanded: list[tuple[str, list[tuple[str, _Candidate]]]] = []

    def finish() -> Neighbourhood:
        # Links between mapped nodes (a neighbor a sibling added later, too) are
        # kept as edges, never as extra nodes.
        for node_id, ranked in expanded:
            for other_id, cand in ranked:
                if other_id in mapped and other_id != node_id:
                    _add_edge(result, seen_edges, node_id, other_id, cand, suite_name)
        # Only expanded or degree-checked nodes come back with full properties.
        for raw in fetched.values():
            for n in raw.nodes:
                if n.id in mapped and (xp := crosswalk_props(n.properties)):
                    result.xprops[n.id] = xp
        return result

    for level in range(1, depth + 1):
        next_frontier: list[str] = []
        for node_id, raw in frontier:
            ranked = sorted(_candidates(raw, node_id).items(), key=lambda kv: kv[1].key)
            expanded.append((node_id, ranked))
            fresh = [
                (other_id, cand)
                for other_id, cand in ranked
                if other_id not in mapped
                and (allowed is None or cand.node.kind.casefold() in allowed)
            ]
            kept = 0
            by_id = dict(fresh)
            hubs = frozenset(i for i, c in fresh if c.node.kind in HUB_KINDS)

            def degree(i: str, hubs: frozenset[str] = hubs) -> int | None:
                # Hubs are never expanded, so their degree is not looked up.
                return None if i in hubs else degree_of(i)

            order = rerank_neighbours([(i, c.key) for i, c in fresh], breadth, degree)
            for other_id in order:
                cand = by_id[other_id]
                if len(result.nodes) >= max_nodes:
                    result.pruned.append(Pruned(node_id, considered=len(fresh), kept=kept))
                    result.warnings.append(f"max_nodes:{max_nodes}")
                    return finish()
                mapped[other_id] = MapNode(_node_ref(cand.node, suite_name), level)
                result.nodes.append(mapped[other_id])
                _add_edge(result, seen_edges, node_id, other_id, cand, suite_name)
                kept += 1
                if cand.node.kind not in HUB_KINDS:
                    next_frontier.append(other_id)
            result.pruned.append(Pruned(node_id, considered=len(fresh), kept=kept))
        if not next_frontier:
            break
        frontier = [(nid, neighbors(nid)) for nid in next_frontier]
    return finish()


class MapperSuite(MapperNeighbors, SearchableSuite, Protocol):
    """A suite that can both resolve a name and list neighbors."""


@dataclass(frozen=True)
class StartMatch:
    """How a start name resolved: one node ID, or (when unsure) a pick list, or neither."""

    id: str | None = None
    confidence: float | None = None  # Locate's confidence of the best hit
    candidates: tuple[NodeRef, ...] = ()
    # per candidate ID: (confidence, match method), as the suite reported them
    candidate_hits: dict[str, tuple[float, str]] = field(default_factory=dict)


class _Spy:
    """Passes search through and keeps the raw result, so per-candidate confidence survives."""

    def __init__(self, suite: SearchableSuite) -> None:
        self.suite = suite
        self.last: Any = None

    def search_nodes(self, text: str, kind: str | None = None) -> Any:
        self.last = self.suite.search_nodes(text, kind=kind)
        return self.last


def resolve_start(suite: SearchableSuite, suite_name: str, text: str) -> StartMatch:
    """Resolve a node ID or a name; only a clear match is taken, never a guess."""
    if text.startswith(f"{suite_name}:"):
        return StartMatch(id=text)
    spy = _Spy(suite)
    located = locate(spy, suite_name, text)  # type: ignore[arg-type]
    found = located.nodes
    if not found:
        return StartMatch()
    # A lone hit is only taken when it is a label match (exact, alternative or
    # case-insensitive, the tiers Locate itself auto-selects). A substring or
    # meaning hit ("data analyst" -> "Data scientist") is shown, never picked.
    exact = found[0].pref_label.casefold() == text.casefold()
    sure = (
        len(found) == 1
        and "ambiguous" not in located.warnings
        and (located.confidence or 0.0) >= AUTO_PICK_CONFIDENCE
    )
    if exact or sure:
        return StartMatch(id=found[0].id, confidence=located.confidence)
    hits = {c.node.id: (c.confidence, c.method) for c in spy.last.candidates} if spy.last else {}
    return StartMatch(
        confidence=located.confidence,
        candidates=tuple(found[:10]),
        candidate_hits=hits,
    )


def map_from_text(
    suite: MapperSuite,
    suite_name: str,
    text: str,
    *,
    depth: int,
    breadth: int,
    kinds: Iterable[str] | None = None,
    max_nodes: int = DEFAULT_MAX_NODES,
) -> Neighbourhood:
    """Map from a node ID or a name; an ambiguous name returns candidates, never a guess."""
    _check_limits(depth, breadth)
    empty = Neighbourhood(suite=suite_name, start_id="", depth=depth, breadth=breadth)
    match = resolve_start(suite, suite_name, text)
    if match.id is None:
        empty.warnings.append("ambiguous_start" if match.candidates else f"start_not_found:{text}")
        empty.candidates = list(match.candidates)
        return empty
    walked = map_neighbourhood(
        suite, suite_name, match.id, depth=depth, breadth=breadth, kinds=kinds, max_nodes=max_nodes
    )
    if text.startswith(f"{suite_name}:") and not walked.nodes:
        # An ID that does not exist is a miss; searching its text would guess.
        empty.warnings.append(f"start_not_found:{text}")
        return empty
    return walked


@dataclass
class RouteMap:
    """One merged graph: a Pathfind route plus the neighbourhood of each stop."""

    suite: str
    route: list[str]
    depth: int
    breadth: int
    nodes: list[MapNode] = field(default_factory=list)
    edges: list[MapEdge] = field(default_factory=list)
    pruned: list[Pruned] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    xprops: dict[str, dict[str, object]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        stops = {node_id: i for i, node_id in enumerate(self.route)}
        return {
            "suite": self.suite,
            "policy": MAP_RANK_V1.name,
            "route": list(self.route),
            "depth": self.depth,
            "breadth": self.breadth,
            "meaning": MEANING,
            "nodes": [
                {**_node_dict(n), "on_route": n.ref.id in stops}
                | ({"stop": stops[n.ref.id]} if n.ref.id in stops else {})
                for n in self.nodes
            ],
            "edges": [_edge_dict(e) for e in self.edges],
            "pruned": [_pruned_dict(p) for p in self.pruned],
            "warnings": list(self.warnings),
        }


def map_route(
    suite: MapperNeighbors,
    suite_name: str,
    route_node_ids: Sequence[str],
    *,
    depth: int = 1,
    breadth: int = 3,
    kinds: Iterable[str] | None = None,
    max_nodes: int = DEFAULT_MAX_NODES,
) -> RouteMap:
    """Walk around every stop of a route and merge into one graph (first stop wins)."""
    walks = [
        map_neighbourhood(
            suite,
            suite_name,
            node_id,
            depth=depth,
            breadth=breadth,
            kinds=kinds,
            max_nodes=max_nodes,
        )
        for node_id in route_node_ids
    ]
    merged = RouteMap(suite_name, [w.start_id for w in walks if w.nodes], depth, breadth)
    for node_id, walk in zip(route_node_ids, walks, strict=True):
        if not walk.nodes:
            merged.warnings.append(f"start_not_found:{node_id}")
    # Route stops first, all at level 0, so a stop is never filed as someone's neighbour.
    seen = {w.start_id for w in walks if w.nodes}
    merged.nodes = [MapNode(w.nodes[0].ref, 0) for w in walks if w.nodes]
    seen_edges: set[tuple[str, str, str]] = set()
    seen_pruned: set[str] = set()
    for walk in walks:
        for node in walk.nodes:
            if node.ref.id in seen:
                continue
            if len(seen) >= max_nodes:
                if f"max_nodes:{max_nodes}" not in merged.warnings:
                    merged.warnings.append(f"max_nodes:{max_nodes}")
                break
            seen.add(node.ref.id)
            merged.nodes.append(node)
        merged.warnings += [w for w in walk.warnings if w not in merged.warnings]
        merged.xprops |= walk.xprops
        for edge in walk.edges:
            key = _edge_key(edge.ref.source_node_id, edge.ref.target_node_id, edge.ref.type)
            if key not in seen_edges and {key[1], key[2]} <= seen:
                seen_edges.add(key)
                merged.edges.append(edge)
        for p in walk.pruned:
            if p.node_id not in seen_pruned:
                seen_pruned.add(p.node_id)
                merged.pruned.append(p)
    return merged


def _add_edge(
    result: Neighbourhood,
    seen: set[tuple[str, str, str]],
    from_id: str,
    to_id: str,
    cand: _Candidate,
    suite_name: str,
) -> None:
    graph_type = _graph_type(cand.relation)
    key = _edge_key(from_id, to_id, graph_type)
    if key in seen:
        return
    seen.add(key)
    ref = EdgeRef(
        type=graph_type,
        suite=suite_name,
        source_node_id=from_id,
        target_node_id=to_id,
        properties={},
    )
    result.edges.append(MapEdge(ref=ref, relation=cand.relation, strength=cand.strength))


def _edge_key(a: str, b: str, graph_type: str) -> tuple[str, str, str]:
    """One graph link, whichever side the walk reached it from."""
    low, high = sorted((a, b))
    return (graph_type, low, high)


def _graph_type(relation: str) -> str:
    return {
        "parent": "BROADER_THAN",
        "narrower": "BROADER_THAN",
        "related": "RELATED_TO",
        "level": "HAS_LEVEL",
        "needs": "HAS_SKILL",
        "needed_by": "HAS_SKILL",
        "group": "CLASSIFIED_UNDER",
    }.get(relation, relation.upper())


__all__ = [
    "DEFAULT_MAX_NODES",
    "HUB_KINDS",
    "MAX_BREADTH",
    "MAX_DEPTH",
    "MapEdge",
    "MapNode",
    "Neighbourhood",
    "MapperSuite",
    "RouteMap",
    "Pruned",
    "StartMatch",
    "map_from_text",
    "map_neighbourhood",
    "map_route",
    "resolve_start",
]

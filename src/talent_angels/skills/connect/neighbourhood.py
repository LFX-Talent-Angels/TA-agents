"""Graph-Mapper core: a deterministic multi-hop walk over Connect's one-hop neighbors.

Each expanded node asks the suite for its direct neighbors (one Connect call),
ranks them by a fixed rule, keeps the top ``breadth`` not yet mapped, and repeats
from those for ``depth`` levels. No model call: same suite data in, same bytes out.
Hub kinds (groups, levels, categories) are shown but never expanded, so the walk
does not drift from a skill out to its whole group.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Protocol

from talent_angels.contracts import EdgeRef, NodeRef
from talent_angels.skills.connect.reveal import NeighborResult, TaxonomyEdge, TaxonomyNode

MAX_DEPTH = 4
MAX_BREADTH = 10
DEFAULT_MAX_NODES = 500

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


@dataclass
class Neighbourhood:
    suite: str
    start_id: str
    depth: int
    breadth: int
    nodes: list[MapNode] = field(default_factory=list)
    edges: list[MapEdge] = field(default_factory=list)
    pruned: list[Pruned] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        """Plain JSON-ready form; field order and list order are fixed."""
        return {
            "suite": self.suite,
            "start": self.start_id,
            "depth": self.depth,
            "breadth": self.breadth,
            "meaning": "linked in the graph; not a learning order",
            "nodes": [
                {"id": n.ref.id, "label": n.ref.pref_label, "kind": n.ref.kind, "level": n.level}
                for n in self.nodes
            ],
            "edges": [
                {
                    "from": e.ref.source_node_id,
                    "to": e.ref.target_node_id,
                    "relation": e.relation,
                    "graph_edge": e.ref.type,
                    "strength": e.strength,
                }
                for e in self.edges
            ],
            "pruned": [
                {"node": p.node_id, "considered": p.considered, "kept": p.kept} for p in self.pruned
            ],
            "warnings": list(self.warnings),
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
    """Lower sorts first: relation, strength, O*NET importance, hot tech, label, id.

    Relation comes first so skill-to-skill links beat job links. Strength first
    would fill the map with jobs (every job is essential or optional for a skill).
    """
    strength = _strength(edge)
    importance = edge.properties.get("importance")
    weight = -float(importance) if isinstance(importance, int | float) else 0.0
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
    if not 1 <= depth <= MAX_DEPTH:
        raise ValueError(f"depth must be 1..{MAX_DEPTH}, got {depth}")
    if not 1 <= breadth <= MAX_BREADTH:
        raise ValueError(f"breadth must be 1..{MAX_BREADTH}, got {breadth}")
    allowed = frozenset(kinds) if kinds is not None else None

    result = Neighbourhood(suite=suite_name, start_id=start_id, depth=depth, breadth=breadth)
    first = suite.get_neighbors(start_id)
    centre = next((n for n in first.nodes if n.id == start_id), None)
    if centre is None:
        result.warnings.append("start_not_found")
        return result
    mapped: dict[str, MapNode] = {start_id: MapNode(_node_ref(centre, suite_name), 0)}
    result.nodes.append(mapped[start_id])
    seen_edges: set[tuple[str, str, str]] = set()
    frontier: list[tuple[str, NeighborResult]] = [(start_id, first)]

    for level in range(1, depth + 1):
        next_frontier: list[str] = []
        for node_id, raw in frontier:
            ranked = sorted(_candidates(raw, node_id).items(), key=lambda kv: kv[1].key)
            # Links to nodes already mapped are kept as edges, not as new nodes.
            for other_id, cand in ranked:
                if other_id in mapped and other_id != node_id:
                    _add_edge(result, seen_edges, node_id, other_id, cand, suite_name)
            fresh = [
                (other_id, cand)
                for other_id, cand in ranked
                if other_id not in mapped and (allowed is None or cand.node.kind in allowed)
            ]
            keep = fresh[:breadth]
            result.pruned.append(Pruned(node_id, considered=len(fresh), kept=len(keep)))
            for other_id, cand in keep:
                if len(result.nodes) >= max_nodes:
                    result.warnings.append(f"max_nodes:{max_nodes}")
                    return result
                mapped[other_id] = MapNode(_node_ref(cand.node, suite_name), level)
                result.nodes.append(mapped[other_id])
                _add_edge(result, seen_edges, node_id, other_id, cand, suite_name)
                if cand.node.kind not in HUB_KINDS:
                    next_frontier.append(other_id)
        if not next_frontier:
            break
        frontier = [(nid, suite.get_neighbors(nid)) for nid in next_frontier]
    return result


def _add_edge(
    result: Neighbourhood,
    seen: set[tuple[str, str, str]],
    from_id: str,
    to_id: str,
    cand: _Candidate,
    suite_name: str,
) -> None:
    key = (from_id, to_id, cand.relation)
    if key in seen:
        return
    seen.add(key)
    graph_type = _graph_type(cand.relation)
    ref = EdgeRef(
        type=graph_type,
        suite=suite_name,
        source_node_id=from_id,
        target_node_id=to_id,
        properties={},
    )
    result.edges.append(MapEdge(ref=ref, relation=cand.relation, strength=cand.strength))


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
    "Pruned",
    "map_neighbourhood",
]

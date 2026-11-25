"""Successors of X: a tree of what comes after a skill, under policy learn-order-v1 (ADR-0013).

Children of a node, ESCO: *stated* are the skills ESCO says need it (essential-for); *policy*
are its narrower skills (the mapper's own parent/narrower reading of BROADER_THAN) and the
related skills of a later reuse level. SFIA: the next level of the skill, stated. Stated
children come first, then reuse level, label and node ID. A node is never repeated, hubs are
not expanded, depth, breadth and max_nodes are hard caps. A taxonomy with no order data says
``no_order_data``. No model call: same input and graph data give the same JSON.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from talent_angels.skills.connect.federated import _select
from talent_angels.skills.connect.neighbourhood import HUB_KINDS, _check_limits, _relation
from talent_angels.skills.evaluate.policy import LEARN_ORDER_V1 as P
from talent_angels.skills.roadmap.common import (
    MAX_NODES,
    Graph,
    Json,
    id_suite,
    ref,
    resolve_side,
)
from talent_angels.suites import SuiteRegistry

DEFAULT_MAX_NODES = 200
ORDERED = ("esco", "sfia")  # taxonomies that carry order data for a skill


@dataclass(frozen=True)
class Child:
    id: str
    label: str
    kind: str
    cite: str
    stated: bool
    reason: str
    reuse: str | None = None
    expand: bool = True

    @property
    def key(self) -> tuple[int, int, str, str]:
        rank = (
            P.reuse_order.index(self.reuse) if self.reuse in P.reuse_order else len(P.reuse_order)
        )
        return (0 if self.stated else 1, rank, self.label.casefold(), self.id)


def _rank(reuse: object) -> int | None:
    return P.reuse_order.index(reuse) if reuse in P.reuse_order else None  # type: ignore[arg-type]


def _esco_children(graph: Graph, node_id: str) -> list[Child]:
    res = graph.neighbors(node_id)
    labels = {n.id: n for n in res.nodes}
    mine = _rank((graph.props(node_id) or {}).get("reuse_level"))
    found: dict[str, tuple[bool, str]] = {}  # id -> (stated, reason)
    related: list[str] = []
    for e in res.edges:
        other = e.to_id if e.from_id == node_id else e.from_id
        if other not in labels or other == node_id:
            continue
        if e.type == "RELATED_TO":
            if e.properties.get("relation_type") == "essential":
                if e.to_id == node_id:  # `other` needs this skill
                    found[other] = (True, "esco:essential-for")
            else:
                related.append(other)
        elif e.type == "BROADER_THAN" and _relation(e, node_id) == "narrower":
            found.setdefault(other, (False, "esco:narrower"))
    pool = sorted(set(related) - found.keys(), key=lambda i: (labels[i].label.casefold(), i))
    if len(pool) > P.related_pool:
        graph.notes.add(f"related_pool:{P.related_pool}")
    kids: list[Child] = []
    for other in [*found, *pool[: P.related_pool]]:
        node = labels[other]
        hub = node.kind in HUB_KINDS
        reuse = None if hub else (graph.props(other) or {}).get("reuse_level")
        if other not in found:  # a related skill counts only when it comes later
            later = _rank(reuse)
            if later is None or mine is None or later <= mine:
                continue
        stated, reason = found.get(other, (False, "esco:related-later-reuse"))
        kids.append(
            Child(
                other, node.label, node.kind, node.source_id, stated, reason,
                reuse if isinstance(reuse, str) else None, not hub,
            )
        )  # fmt: skip
    return sorted(kids, key=lambda k: k.key)


def _level(node_id: str) -> int:
    last = node_id.rsplit(":", 1)[-1]
    return int(last) if last.isdigit() else 0


def _sfia_children(graph: Graph, node_id: str) -> list[Child]:
    node = graph.node(node_id)
    if node is None:
        return []
    skill = node_id
    if node.kind == "SkillLevel":  # its skill is the one that has this level
        skill = next(
            (e.from_id for e in graph.neighbors(node_id).edges if e.type == "HAS_LEVEL"), ""
        )
    elif node.kind != "Skill":
        return []
    levels = {n.id: n for n in graph.neighbors(skill).nodes if n.kind == "SkillLevel"}
    later = [i for i in levels if _level(i) > (_level(node_id) if node.kind == "SkillLevel" else 0)]
    if not later:
        return []
    nxt = levels[min(later, key=_level)]
    reason = "sfia:next-level" if node.kind == "SkillLevel" else "sfia:first-level"
    return [Child(nxt.id, nxt.label, nxt.kind, nxt.source_id, True, reason)]


_CHILDREN = {"esco": _esco_children, "sfia": _sfia_children}
_START_KINDS = {"esco": {"Skill"}, "sfia": {"Skill", "SkillLevel"}}


def _walk(graph: Graph, name: str, start: Any, depth: int, breadth: int, max_nodes: int) -> Json:
    children = _CHILDREN[name]
    reuse = (graph.props(start.id) or {}).get("reuse_level")
    nodes: list[Json] = [{**ref(start), "depth": 0, "reuse_level": reuse}]
    edges: list[Json] = []
    pruned: list[Json] = []
    warnings: list[str] = []
    seen = {start.id}
    frontier = [start.id]
    for level in range(1, depth + 1):
        following: list[str] = []
        for parent in frontier:
            fresh = [k for k in children(graph, parent) if k.id not in seen]
            kept = 0
            for kid in fresh[:breadth]:
                if len(nodes) >= max_nodes:
                    pruned.append({"node": parent, "considered": len(fresh), "kept": kept})
                    return _tree(nodes, edges, pruned, [*warnings, f"max_nodes:{max_nodes}"], graph)
                seen.add(kid.id)
                nodes.append(
                    {"id": kid.id, "label": kid.label, "kind": kid.kind, "cite": kid.cite,
                     "depth": level, "reuse_level": kid.reuse}
                )  # fmt: skip
                edges.append(
                    {"from_id": parent, "to_id": kid.id,
                     "kind": "stated" if kid.stated else "policy", "reason": kid.reason}
                )  # fmt: skip
                kept += 1
                if kid.expand:
                    following.append(kid.id)
            pruned.append({"node": parent, "considered": len(fresh), "kept": kept})
        frontier = following
        if not frontier:
            break
    return _tree(nodes, edges, pruned, warnings, graph)


def _tree(
    nodes: list[Json], edges: list[Json], pruned: list[Json], warn: list[str], g: Graph
) -> Json:
    return {"nodes": nodes, "edges": edges, "pruned": pruned, "warnings": warn + g.warnings}


def successors(
    registry: SuiteRegistry,
    x: str,
    *,
    suites: Iterable[str] | None = None,
    depth: int = 2,
    breadth: int = 4,
    max_nodes: int = DEFAULT_MAX_NODES,
) -> Json:
    """What comes after ``x`` in every selected taxonomy, as a cited tree."""
    _check_limits(depth, breadth)
    if not 1 <= max_nodes <= MAX_NODES:
        raise ValueError(f"max_nodes must be 1..{MAX_NODES}, got {max_nodes}")
    names = _select(registry, suites)
    owner = id_suite(registry, x)
    out: dict[str, Json] = {}
    for name in names:
        part: Json = {"status": "no_order_data", "start": None, "candidates": [], "warnings": []}
        if name in ORDERED:
            try:
                with registry.open(name) as runtime:
                    graph = Graph(runtime.suite)
                    side = resolve_side(graph, name, x, owner, "start")
                    part["status"] = side.status
                    part["candidates"] = side.candidates
                    if side.node is not None and side.node.kind not in _START_KINDS[name]:
                        part["status"] = "no_order_data"
                        part["warnings"] = [f"not_a_skill:{side.node.kind}"]
                    elif side.node is not None:
                        part["start"] = {"id": side.node.id, "label": side.node.label}
                        part |= _walk(graph, name, side.node, depth, breadth, max_nodes)
            except Exception as exc:  # one broken taxonomy must not hide the others
                part = {
                    "status": f"error:{type(exc).__name__}",
                    "start": None,
                    "candidates": [],
                    "warnings": [],
                }
        out[name] = part
    return {
        "policy": P.name,
        "meaning": P.meaning,
        "x": x,
        "depth": depth,
        "breadth": breadth,
        "max_nodes": max_nodes,
        "suites": out,
        "warnings": [f"{n}:{w}" for n, p in out.items() for w in p["warnings"]],
    }

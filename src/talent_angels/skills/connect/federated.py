"""All-taxonomy Graph-Mapper: one question, every suite mapped, one cited result.

Each suite is walked on its own (ids stay suite-scoped). A suite that cannot
resolve the text may be started from an official crosswalk of a start another
suite resolved. Official crosswalk links (``crosswalk.py``) are added between
mapped nodes; nothing else crosses suites. No model call: same input, same bytes.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

from talent_angels.skills.connect.crosswalk import (
    CrosswalkLink,
    crosswalk_links,
    links_from,
    suite_of,
)
from talent_angels.skills.connect.neighbourhood import (
    DEFAULT_MAX_NODES,
    MEANING,
    MapEdge,
    MapNode,
    Neighbourhood,
    Pruned,
    _check_limits,
    _edge_dict,
    _node_dict,
    _pruned_dict,
    map_from_text,
    map_neighbourhood,
    map_route,
)
from talent_angels.skills.evaluate import rank_routes
from talent_angels.skills.evaluate.policy import MAP_RANK_V1
from talent_angels.suites import SuiteRegistry, UnknownSuiteError

_STRENGTH_ORDER = {"exact": 0, "close": 1, "soc": 2}


@dataclass
class SuiteMap:
    """One suite's share of the merged map."""

    # mapped | ambiguous | not_found | not_searched | via_crosswalk | no_route | error
    status: str = "not_found"
    start: dict[str, str] | None = None
    candidates: list[dict[str, str]] = field(default_factory=list)
    via: str | None = None
    nodes: list[MapNode] = field(default_factory=list)
    edges: list[MapEdge] = field(default_factory=list)
    pruned: list[Pruned] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    xprops: dict[str, dict[str, object]] = field(default_factory=dict)
    route: dict[str, object] | None = None
    route_ids: list[str] = field(default_factory=list)
    side: str | None = None  # route mode: which end is ambiguous (start | target)

    def to_dict(self) -> dict[str, object]:
        out: dict[str, object] = {
            "status": self.status,
            "start": self.start,
            "candidates": self.candidates,
            "via": self.via,
            "nodes": len(self.nodes),
            "pruned": [_pruned_dict(p) for p in self.pruned],
            "warnings": self.warnings,
        }
        if self.route is not None:
            out["route"] = self.route
        if self.side is not None:
            out["side"] = self.side
        return out


@dataclass
class FederatedMap:
    query: str
    depth: int
    breadth: int
    suites: dict[str, SuiteMap] = field(default_factory=dict)

    @property
    def mapped(self) -> bool:
        return any(s.nodes for s in self.suites.values())

    def to_dict(self) -> dict[str, object]:
        mapped = {name: s for name, s in self.suites.items() if s.nodes}
        props = {n.ref.id: s.xprops.get(n.ref.id, {}) for s in mapped.values() for n in s.nodes}
        links = crosswalk_links(props)
        confirmed: dict[str, set[str]] = {}
        for link in links:
            confirmed.setdefault(link.from_id, set()).add(suite_of(link.to_id))
            confirmed.setdefault(link.to_id, set()).add(suite_of(link.from_id))
        nodes: list[dict[str, object]] = []
        edges: list[dict[str, object]] = []
        for name, s in mapped.items():
            stops = {i: k for k, i in enumerate(s.route_ids)}
            for n in s.nodes:
                item = {**_node_dict(n), "suite": name, "cite": n.ref.source_id}
                if s.route:
                    item["on_route"] = n.ref.id in stops
                    if n.ref.id in stops:
                        item["stop"] = stops[n.ref.id]
                if n.ref.id in confirmed:
                    item["confirmed_by"] = sorted(confirmed[n.ref.id])
                nodes.append(item)
            edges += [{**_edge_dict(e), "suite": name} for e in s.edges]
        edges += [link.to_dict() for link in links]
        return {
            "query": self.query,
            "policy": MAP_RANK_V1.name,
            "meaning": MEANING,
            "depth": self.depth,
            "breadth": self.breadth,
            "suites": {name: s.to_dict() for name, s in self.suites.items()},
            "nodes": nodes,
            "edges": edges,
            "summary": {
                "suites_mapped": len(mapped),
                "nodes": len(nodes),
                "edges": len(edges),
                "crosswalk_links": len(links),
                "confirmed_nodes": len(confirmed),
            },
            "warnings": [] if mapped else ["no_suite_mapped"],
        }


def _select(registry: SuiteRegistry, suites: Iterable[str] | None) -> list[str]:
    if suites is None:
        return list(registry.available)
    picked = [registry.resolve(s) for s in suites]
    for name in picked:
        if name not in registry.available:
            raise UnknownSuiteError(
                f"unknown suite {name!r}; available: {', '.join(registry.available)}"
            )
    return [n for n in registry.available if n in picked]


def _per_suite_cap(max_nodes: int, count: int) -> int:
    # Hard cap for the whole map: never more than max_nodes in total.
    return max(1, max_nodes // max(1, count))


def _from_walk(walk: Neighbourhood) -> SuiteMap:
    part = SuiteMap(
        nodes=walk.nodes,
        edges=walk.edges,
        pruned=walk.pruned,
        warnings=walk.warnings,
        xprops=walk.xprops,
    )
    if walk.nodes:
        part.status = "mapped"
        part.start = {"id": walk.start_id, "label": walk.start_label}
    elif "ambiguous_start" in walk.warnings:
        part.status = "ambiguous"
        part.candidates = [
            {"id": c.id, "label": c.pref_label, "kind": c.kind} for c in walk.candidates
        ]
    return part


def _seeds(target: str, parts: dict[str, SuiteMap]) -> list[CrosswalkLink]:
    """Official links from a resolved start in another suite to a node of ``target``."""
    found: list[CrosswalkLink] = []
    for part in parts.values():
        if part.status == "mapped" and part.start:
            sid = part.start["id"]
            found += [
                link
                for link in links_from(sid, part.xprops.get(sid))
                if suite_of(link.to_id if link.from_id == sid else link.from_id) == target
            ]
    return sorted(found, key=lambda x: (_STRENGTH_ORDER[x.strength], x.from_id, x.to_id))


def map_all(
    registry: SuiteRegistry,
    text: str,
    *,
    suites: Iterable[str] | None = None,
    depth: int = 1,
    breadth: int = 4,
    kinds: Iterable[str] | None = None,
    max_nodes: int = DEFAULT_MAX_NODES,
) -> FederatedMap:
    """Map ``text`` in every selected suite; a failing suite never stops the others."""
    _check_limits(depth, breadth)
    names = _select(registry, suites)
    cap = _per_suite_cap(max_nodes, len(names))
    kinds = list(kinds) if kinds is not None else None
    out = FederatedMap(text, depth, breadth)
    # A node ID belongs to one taxonomy; the others are never searched with its text.
    prefix = text.split(":", 1)[0] if ":" in text else ""
    id_suite = prefix if prefix in registry.available else None
    for name in names:
        if id_suite and name != id_suite:
            out.suites[name] = SuiteMap(
                status="not_searched", warnings=[f"start_is_id_of:{id_suite}"]
            )
            continue
        try:
            with registry.open(name) as rt:
                walk = map_from_text(
                    rt.suite,  # type: ignore[arg-type]
                    name,
                    text,
                    depth=depth,
                    breadth=breadth,
                    kinds=kinds,
                    max_nodes=cap,
                )
            out.suites[name] = _from_walk(walk)
        except Exception as exc:  # one broken suite must not hide the others
            out.suites[name] = SuiteMap(status="error", warnings=[f"error:{type(exc).__name__}"])
    for name in names:
        part = out.suites[name]
        if part.status not in ("not_found", "ambiguous", "not_searched"):
            continue
        for link in _seeds(name, out.suites):
            seed = link.to_id if suite_of(link.to_id) == name else link.from_id
            try:
                with registry.open(name) as rt:
                    walk = map_neighbourhood(
                        rt.suite,  # type: ignore[arg-type]
                        name,
                        seed,
                        depth=depth,
                        breadth=breadth,
                        kinds=kinds,
                        max_nodes=cap,
                    )
            except Exception:
                break
            if walk.nodes:
                seeded = _from_walk(walk)
                seeded.status, seeded.via = "via_crosswalk", link.source
                out.suites[name] = seeded
                break
    return out


def map_all_route(
    registry: SuiteRegistry,
    from_text: str,
    to_text: str,
    *,
    suites: Iterable[str] | None = None,
    depth: int = 1,
    breadth: int = 3,
    kinds: Iterable[str] | None = None,
    max_nodes: int = DEFAULT_MAX_NODES,
) -> FederatedMap:
    """Per suite, map around Pathfind's top-ranked route between the two names."""
    from talent_angels.skills.pathfind.dispatch import pathfind_one  # import cycle via mcp

    _check_limits(depth, breadth)
    names = _select(registry, suites)
    cap = _per_suite_cap(max_nodes, len(names))
    out = FederatedMap(f"{from_text} -> {to_text}", depth, breadth)
    for name in names:
        try:
            with registry.open(name) as rt:
                found, _ = pathfind_one(rt.suite, name, from_text, to_text)  # type: ignore[arg-type]
                # The scores ride in "route" below; the raw list is noise per suite.
                warns = [w for w in found.warnings if not w.startswith("route_scores:")]
                part = SuiteMap(warnings=warns)
                if found.routes:
                    # pathfind_one already ordered routes by route-rank-v1; this re-reads the score.
                    top = rank_routes(rt.suite, found)[0]  # type: ignore[arg-type]
                    walk = map_route(
                        rt.suite,  # type: ignore[arg-type]
                        name,
                        found.routes[0].node_ids,
                        depth=depth,
                        breadth=breadth,
                        kinds=kinds,
                        max_nodes=cap,
                    )
                    part = SuiteMap(
                        status="mapped",
                        nodes=walk.nodes,
                        edges=walk.edges,
                        pruned=walk.pruned,
                        warnings=[
                            *walk.warnings,
                            *(w for w in warns if w not in walk.warnings),
                        ],
                        xprops=walk.xprops,
                        route_ids=list(walk.route),
                        route={
                            "node_ids": list(walk.route),
                            "score": top.score,
                            "reasons": top.reasons,
                        },
                    )
                elif "ambiguous" in found.warnings:
                    part.status = "ambiguous"
                    part.side = next(
                        (
                            w.split(":", 1)[1]
                            for w in found.warnings
                            if w.startswith("pathfind_side:")
                        ),
                        None,
                    )
                    part.candidates = [
                        {"id": n.id, "label": n.pref_label, "kind": n.kind}
                        for n in found.nodes[:10]
                    ]
                elif any(w.startswith("pathfind_side:") for w in found.warnings):
                    part.status = "not_found"
                else:
                    part.status = "no_route"
            out.suites[name] = part
        except Exception as exc:
            out.suites[name] = SuiteMap(status="error", warnings=[f"error:{type(exc).__name__}"])
    return out


__all__ = ["FederatedMap", "SuiteMap", "map_all", "map_all_route"]

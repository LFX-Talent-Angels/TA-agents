"""Discovery map: how a list of roles, tasks and skills connects in the taxonomy graphs.

Every item is resolved with the Graph-Mapper's own start rule (a clear match only), mapped
with the Graph-Mapper, merged into one cited graph and classified in code: direct links,
connectors (nodes two or more items reach), routes (Pathfind + route-rank-v1), orphans,
clusters, and a verdict on each of the user's own guesses. No model call, no inferred edge.

Determinism: items are sorted, every list ends in a total order (node ID last), floats are
rounded, nothing time- or run-dependent is recorded. Same resolved node set and same graph
data give the same JSON, whatever the order of the input rows.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from contextlib import ExitStack
from itertools import combinations
from typing import Any

from talent_angels.contracts import AgentResult, Route
from talent_angels.llm.protocol import check_cancelled
from talent_angels.skills.connect.crosswalk import crosswalk_links, suite_of
from talent_angels.skills.connect.discover_io import MAX_ITEMS, Item
from talent_angels.skills.connect.federated import _select
from talent_angels.skills.connect.neighbourhood import (
    HUB_KINDS,
    MEANING,
    _candidates,
    _check_limits,
    _edge_dict,
    _edge_key,
    _graph_type,
    map_neighbourhood,
    resolve_start,
)
from talent_angels.skills.evaluate import ROUTE_RANK_V1, rank_routes
from talent_angels.skills.evaluate.policy import MAP_RANK_V1
from talent_angels.suites import SuiteRegistry

POLICY = "discover-v1"
DEFAULT_DISCOVER_MAX_NODES = 2000
MAX_DISCOVER_NODES = 5000
MAX_ROUTE_PAIRS = 30  # item pairs routed beyond the user's own guesses (those always are)
INDIRECT_MAX_EDGES = 3  # a guess counts as indirect through a route of at most this many edges
MAX_CANDIDATES = 3
TOOL_EDGES = frozenset({"USES_SOFTWARE"})  # a job "uses" a tool: weak evidence two items relate
MEANING_ONLY = "hybrid_rrf"  # the suite's match method when no label tier matched: a guess

Json = dict[str, Any]


def _round(value: float | None) -> float | None:
    return None if value is None else round(float(value), 4)


def _resolve(suite: Any, name: str, text: str) -> Json:
    match = resolve_start(suite, name, text)
    if match.id is not None:
        return {"status": "resolved", "id": match.id, "confidence": _round(match.confidence)}
    if not match.candidates:
        return {"status": "not_found"}
    scored = sorted(
        (-match.candidate_hits.get(c.id, (0.0, ""))[0], c.pref_label.casefold(), c.id, c)
        for c in match.candidates
    )
    top = [
        {
            "id": c.id,
            "label": c.pref_label,
            "kind": c.kind,
            "confidence": _round(match.candidate_hits.get(c.id, (None, ""))[0]),
            "method": match.candidate_hits.get(c.id, (None, None))[1],
        }
        for *_, c in scored[:MAX_CANDIDATES]
    ]
    return {"status": "unclear", "candidates": top}


def _recorded(entry: Mapping[str, Any] | None) -> Json:
    """One entry of a previous run's ``resolved`` block (or a hand-written one)."""
    if entry is None:
        return {"status": "not_found"}
    if entry.get("id") and entry.get("status", "resolved") == "resolved":
        picked = {"picked": True} if entry.get("picked") else {}
        return picked | {
            "status": "resolved",
            "id": str(entry["id"]),
            "confidence": entry.get("confidence"),
        }
    if entry.get("status") == "unclear":
        return {"status": "unclear", "candidates": list(entry.get("candidates") or [])}
    return {"status": "not_found"}


def _best_route(
    suites: Mapping[str, Any], nodes_of: Mapping[str, Mapping[str, str]], a: str, b: str
) -> Json | None:
    """The top route-rank-v1 route between two items in any suite that resolves both."""
    from talent_angels.skills.pathfind.routes import find_routes, supports_enumerate_paths

    best: tuple[Any, ...] | None = None
    for name, suite in suites.items():
        left, right = nodes_of[a].get(name), nodes_of[b].get(name)
        if not left or not right or left == right or not supports_enumerate_paths(suite):
            continue
        found = find_routes(suite, name, left, right)
        if not found.routes:
            continue
        labels = {n.id: n for r in found.routes for n in r.nodes}
        result = AgentResult(
            capability="pathfind",
            suite=name,
            nodes=list(labels.values()),
            routes=[
                Route(node_ids=[n.id for n in r.nodes], edges=list(r.edges)) for r in found.routes
            ],
        )
        scorer: Any = suite
        top = rank_routes(scorer, result)[0]
        rank = (-top.score, len(top.route.node_ids), name, tuple(top.route.node_ids))
        if best is None or rank < best[0]:
            best = (rank, name, top, labels)
    if best is None:
        return None
    _, name, top, labels = best
    return {
        "suite": name,
        "edges": len(top.route.node_ids) - 1,
        "score": _round(top.score),
        "reasons": list(top.reasons),
        "stops": [
            {"id": i, "label": labels[i].pref_label, "kind": labels[i].kind}
            for i in top.route.node_ids
        ],
    }


def discover(
    registry: SuiteRegistry,
    items: Sequence[Item],
    *,
    suites: Iterable[str] | None = None,
    depth: int = 2,
    breadth: int = 4,
    kinds: Iterable[str] | None = None,
    max_nodes: int = DEFAULT_DISCOVER_MAX_NODES,
    recorded: Sequence[Mapping[str, Any]] | None = None,
    picks: Sequence[str] = (),
    notes: Sequence[str] = (),
) -> Json:
    """Resolve, map, merge and classify ``items``; ``recorded`` replaces resolution."""
    _check_limits(depth, breadth)
    if not 1 <= max_nodes <= MAX_DISCOVER_NODES:
        raise ValueError(f"max_nodes must be 1..{MAX_DISCOVER_NODES}, got {max_nodes}")
    names = _select(registry, suites)
    kind_list = sorted(kinds) if kinds is not None else None
    items = sorted(items, key=lambda i: i.key)
    if len(items) > MAX_ITEMS:
        raise ValueError(f"at most {MAX_ITEMS} items, got {len(items)}")
    text = {i.key: i.text for i in items}
    by_record = {
        (str(e.get("suite")), str(e.get("item", "")).casefold()): e for e in recorded or []
    }

    chosen: dict[tuple[str, str], str] = {}  # (suite, item key) -> the user's node ID
    for pick in picks:
        item_text, sep, node_id = (part.strip() for part in pick.rpartition("="))
        key, suite_name = item_text.casefold(), suite_of(node_id)
        if not sep or key not in text or suite_name not in names:
            raise ValueError(
                f"bad pick {pick!r}: use ITEM=ID, with an item from the list and an ID from "
                f"one of {', '.join(names)}"
            )
        if chosen.setdefault((suite_name, key), node_id) != node_id:
            raise ValueError(f"two picks for {item_text!r} in {suite_name}")

    warnings: set[str] = set()
    state: dict[str, Json] = {}  # suite -> {"status": ok|error:<Type>}
    res: dict[tuple[str, str], Json] = {}  # (suite, item key) -> resolution
    walks: dict[tuple[str, str], Any] = {}
    direct: list[Json] = []
    versions: Json = {}

    with ExitStack() as stack:
        live: dict[str, Any] = {}
        for name in names:
            try:
                runtime = stack.enter_context(registry.open(name))
                live[name] = runtime.suite
                if version := getattr(runtime.suite, "data_version", None):
                    versions[name] = str(version)
                for item in items:
                    check_cancelled()
                    res[(name, item.key)] = (
                        _recorded(by_record.get((name, item.key)))
                        if recorded is not None
                        else _resolve(runtime.suite, name, item.text)
                    )
                for (suite_name, key), node_id in chosen.items():
                    if suite_name == name:
                        res[(name, key)] = {
                            "status": "resolved",
                            "id": node_id,
                            "confidence": None,
                            "picked": True,
                        }
            except Exception as exc:  # one broken suite must not hide the others
                live.pop(name, None)
                state[name] = {"status": f"error:{type(exc).__name__}"}
                for item in items:
                    res.pop((name, item.key), None)
        total = sum(1 for r in res.values() if r["status"] == "resolved")
        cap = max(1, max_nodes // max(1, total))
        for name, suite in list(live.items()):
            try:
                for item in items:
                    check_cancelled()
                    r = res[(name, item.key)]
                    if r["status"] != "resolved":
                        continue
                    walk = map_neighbourhood(
                        suite,
                        name,
                        r["id"],
                        depth=depth,
                        breadth=breadth,
                        kinds=kind_list,
                        max_nodes=cap,
                    )
                    if not walk.nodes:
                        res[(name, item.key)] = {"status": "not_found"}
                        warnings.add(f"{name}:{item.text}:node_not_found")
                        continue
                    r.update(label=walk.start_label, kind=walk.nodes[0].ref.kind)
                    r["cite"] = walk.nodes[0].ref.source_id
                    walks[(name, item.key)] = walk
                    warnings.update(f"{name}:{item.text}:{w}" for w in walk.warnings)
                # Every direct link between two items, however the walk's breadth cut it.
                owner = defaultdict(list)
                for item in items:
                    if (name, item.key) in walks:
                        owner[res[(name, item.key)]["id"]].append(item.key)
                for centre in sorted(owner):
                    for other, cand in sorted(
                        _candidates(suite.get_neighbors(centre), centre).items()
                    ):
                        if other in owner and other != centre:
                            direct.append(
                                {
                                    "from_id": centre,
                                    "to_id": other,
                                    "relation": cand.relation,
                                    "graph_edge": _graph_type(cand.relation),
                                    "strength": cand.strength,
                                    "suite": name,
                                }
                            )
                state[name] = {"status": "ok"}
            except Exception as exc:
                live.pop(name, None)
                state[name] = {"status": f"error:{type(exc).__name__}"}
                for item in items:
                    res.pop((name, item.key), None)
                    walks.pop((name, item.key), None)

        # ---- one merged graph
        nodes: dict[str, Json] = {}
        reach: dict[str, dict[str, int]] = defaultdict(dict)  # node -> item key -> level
        edges: dict[tuple[str, str, str], Json] = {}
        xprops: dict[str, dict[str, object]] = {}
        item_nodes: dict[str, list[str]] = defaultdict(list)  # node -> item keys
        nodes_of: dict[str, dict[str, str]] = defaultdict(dict)  # item key -> suite -> node
        for (name, key), walk in sorted(walks.items(), key=lambda kv: (kv[0][1], kv[0][0])):
            start = res[(name, key)]["id"]
            item_nodes[start].append(key)
            nodes_of[key][name] = start
            for n in walk.nodes:
                nodes.setdefault(
                    n.ref.id,
                    {
                        "id": n.ref.id,
                        "label": n.ref.pref_label,
                        "kind": n.ref.kind,
                        "suite": name,
                        "cite": n.ref.source_id,
                    },
                )
                reach[n.ref.id][key] = min(n.level, reach[n.ref.id].get(key, n.level))
            for e in walk.edges:
                edges.setdefault(
                    _edge_key(e.ref.source_node_id, e.ref.target_node_id, e.ref.type),
                    {**_edge_dict(e), "suite": name},
                )
            xprops |= walk.xprops
        for e in direct:
            edges.setdefault(_edge_key(e["from_id"], e["to_id"], e["graph_edge"]), e)
        cross = crosswalk_links(xprops)
        for link in cross:
            edges.setdefault(_edge_key(link.from_id, link.to_id, "CROSSWALK"), link.to_dict())

        # ---- links, connectors
        links: list[Json] = []

        def add_link(a: str, b: str, a_id: str, b_id: str, e: Json) -> None:
            if a > b:
                a, b, a_id, b_id = b, a, b_id, a_id
            links.append(
                {
                    "a": text[a],
                    "b": text[b],
                    "a_id": a_id,
                    "b_id": b_id,
                    "relation": e["relation"],
                    "graph_edge": e["graph_edge"],
                    "strength": e["strength"],
                    "suite": e.get("suite") or e.get("source"),
                    "_a": a,
                    "_b": b,
                }
            )

        for e in edges.values():
            for a in item_nodes.get(e["from_id"], ()):
                for b in item_nodes.get(e["to_id"], ()):
                    if a != b:
                        add_link(a, b, e["from_id"], e["to_id"], e)
        for node_id, owners in item_nodes.items():
            for a, b in combinations(sorted(owners), 2):  # two items, one node: the same thing
                same = {"relation": "same_node", "graph_edge": "SAME_NODE", "strength": None}
                add_link(a, b, node_id, node_id, {**same, "suite": suite_of(node_id)})
        links.sort(key=lambda x: (x["_a"], x["_b"], x["graph_edge"], x["a_id"], x["b_id"]))
        linked = {(x["_a"], x["_b"]) for x in links}

        by_tool: dict[tuple[str, str], set[str]] = {}

        def tool_reached(suite: str, key: str) -> set[str]:
            """Nodes this item's walk reaches only through a tool edge (O*NET USES_SOFTWARE: a
            job uses it), directly or by going on from such a node. One mixed way in, say a plain
            RELATED_TO, and the node is not here. Edges elsewhere in the walk do not matter."""
            if (suite, key) not in by_tool:
                done: set[str] = set()
                by_tool[(suite, key)] = done
                walk = walks[(suite, key)]
                level = {n.ref.id: n.level for n in walk.nodes}
                into: dict[str, list[tuple[str, str]]] = defaultdict(list)  # node -> (from, edge)
                for e in walk.edges:
                    a, b = e.ref.source_node_id, e.ref.target_node_id
                    if level.get(b, -9) == level.get(a, -9) + 1:
                        into[b].append((a, e.ref.type))
                    elif level.get(a, -9) == level.get(b, -9) + 1:
                        into[a].append((b, e.ref.type))
                for node in sorted(into, key=lambda i: (level[i], i)):
                    if all(t in TOOL_EDGES or src in done for src, t in into[node]):
                        done.add(node)
            return by_tool[(suite, key)]

        def tool_only(suite: str, node_id: str, keys: Iterable[str]) -> bool:
            """Every item reaches the node only through a tool: what the items share is "jobs use
            both tools", not a relation between the items."""
            return all((suite, k) in walks and node_id in tool_reached(suite, k) for k in keys)

        connectors: list[Json] = []
        shared: dict[tuple[str, str], list[str]] = defaultdict(list)
        for node_id, who in reach.items():
            if node_id in item_nodes or len(who) < 2:
                continue
            n = nodes[node_id]
            connectors.append(
                {
                    "id": node_id,
                    "label": n["label"],
                    "kind": n["kind"],
                    "suite": n["suite"],
                    "cite": n["cite"],
                    "broad": n["kind"] in HUB_KINDS,
                    "tool_only": tool_only(n["suite"], node_id, who),
                    "item_count": len(who),
                    "items": [{"item": text[k], "level": who[k]} for k in sorted(who)],
                    "_keys": sorted(who),
                }
            )
        # Specific connectors first; broad hubs (skill groups, ISCO groups) after them; those
        # linked only through a tool (O*NET USES_SOFTWARE) last of all.
        connectors.sort(
            key=lambda c: (
                c["tool_only"], c["broad"], -c["item_count"], c["label"].casefold(), c["id"],
            )
        )  # fmt: skip
        for c in connectors:
            for pair in combinations(c["_keys"], 2):
                shared[pair].append(c["id"])

        # ---- routes: guesses first (always), then other unconnected pairs up to the cap
        resolved_keys = [i.key for i in items if nodes_of.get(i.key)]
        guesses_in: list[tuple[str, str, str]] = []  # (item key, guess as typed, guess key)
        for i in items:
            guesses_in += [(i.key, g, g.casefold()) for g in i.related_to]
        wanted = {tuple(sorted((a, b))) for a, _, b in guesses_in}
        open_pairs = [
            p for p in combinations(resolved_keys, 2) if p not in linked and p not in shared
        ]
        first = [p for p in open_pairs if p in wanted]
        # Items with no link or connector yet come first: a route is their last chance.
        touched = {k for p in [*linked, *shared] for k in p}
        rest = sorted(
            (p for p in open_pairs if p not in wanted),
            key=lambda p: (p[0] in touched and p[1] in touched, p),
        )
        route_of: dict[tuple[str, str], Json] = {}
        failed: set[str] = set()  # items in a pair whose route search errored: not fully checked
        for a, b in [*first, *rest[:MAX_ROUTE_PAIRS]]:
            check_cancelled()
            try:
                route = _best_route(live, nodes_of, a, b)
            except Exception as exc:
                warnings.add(f"route:{text[a]}:{text[b]}:error:{type(exc).__name__}")
                failed.update((a, b))
                continue
            if route:
                route_of[(a, b)] = {
                    "a": text[a],
                    "b": text[b],
                    **route,
                    "policy": ROUTE_RANK_V1.name,
                }
        routes = list(route_of.values())

    # ---- who is connected to whom
    near: dict[str, set[str]] = defaultdict(set)
    for a, b in [*linked, *shared, *route_of]:
        near[a].add(b)
        near[b].add(a)
    unrouted = {k for p in rest[MAX_ROUTE_PAIRS:] for k in p} | failed  # not fully checked
    orphans = [
        {"item": text[k], "routes_fully_checked": k not in unrouted}
        for k in resolved_keys
        if not near[k]
    ]

    seen: set[str] = set()
    groups: list[list[str]] = []
    for k in resolved_keys:
        if k in seen or not near[k]:
            continue
        stack_, comp = [k], []
        while stack_:
            cur = stack_.pop()
            if cur not in seen:
                seen.add(cur)
                comp.append(cur)
                stack_ += sorted(near[cur] - seen)
        groups.append(sorted(comp))
    groups.sort(key=lambda g: (-len(g), g[0]))
    clusters = []
    for n, g in enumerate(groups, 1):
        inside = set(g)
        shown = sorted(
            (c for c in connectors if len(inside & set(c["_keys"])) >= 2),
            key=lambda c: (
                c["tool_only"],
                -len(inside & set(c["_keys"])),
                c["broad"],
                c["label"].casefold(),
                c["id"],
            ),
        )
        clusters.append(
            {
                "name": f"Cluster {n}",
                "size": len(g),
                "items": [text[k] for k in g],
                "connectors": [c["label"] for c in shown[:5]],
            }
        )

    # ---- the user's own guesses
    def why_unresolved(key: str, typed: str) -> str | None:
        if key not in text:
            return f"{typed!r} is not in the item list"
        if nodes_of.get(key):
            return None
        where = sorted(s for (s, k), r in res.items() if k == key and r["status"] == "unclear")
        return f"{text[key]!r} is not resolved" + (
            f" (unclear in {', '.join(where)})" if where else ""
        )

    verdicts: list[Json] = []
    for a, typed, b in sorted(set(guesses_in), key=lambda g: (g[0], g[2], g[1])):
        pair = (min(a, b), max(a, b))
        why = [w for w in (why_unresolved(a, text[a]), why_unresolved(b, typed)) if w]
        row: Json = {"item": text[a], "related_to": text.get(b, typed)}
        if why:
            row |= {"verdict": "unresolved", "detail": "; ".join(why)}
        elif pair in linked:
            hit = next(x for x in links if (x["_a"], x["_b"]) == pair)
            row |= {"verdict": "confirmed", "detail": f"{hit['relation']} link in {hit['suite']}"}
        elif pair in shared:
            names_ = [nodes[i]["label"] for i in shared[pair][:3]]
            row |= {"verdict": "indirect", "detail": "both reach: " + ", ".join(names_)}
        elif pair in route_of and route_of[pair]["edges"] <= INDIRECT_MAX_EDGES:
            row |= {"verdict": "indirect", "detail": f"route of {route_of[pair]['edges']} edges"}
        else:
            far = route_of.get(pair)
            row |= {
                "verdict": "not_in_graph",
                "detail": f"closest route has {far['edges']} edges" if far else "no route found",
            }
        verdicts.append(row)

    # ---- the record
    resolved_block: list[Json] = []
    gaps: dict[str, Json] = {n: {"unclear": [], "not_found": []} for n in names}
    items_out: list[Json] = []
    for i in items:
        mine = {s: res[(s, i.key)] for s in names if (s, i.key) in res}
        for s, r in mine.items():
            if r["status"] == "resolved":
                resolved_block.append(
                    {"item": i.text, "suite": s, "status": "resolved"}
                    | {k: r[k] for k in ("id", "label", "kind", "confidence", "cite")}
                    | ({"picked": True} if r.get("picked") else {})
                )
            elif r["status"] == "unclear":
                resolved_block.append(
                    {"item": i.text, "suite": s, "status": "unclear", "candidates": r["candidates"]}
                )
                gaps[s]["unclear"].append({"item": i.text, "candidates": r["candidates"]})
            else:
                gaps[s]["not_found"].append(i.text)
        status = {r["status"] for r in mine.values()}
        items_out.append(
            {
                "item": i.text,
                "kind": i.kind,
                "related_to": list(i.related_to),
                "status": next(
                    s for s in ("resolved", "unclear", "not_found") if s in status | {"not_found"}
                ),
                "resolved_in": [s for s, r in mine.items() if r["status"] == "resolved"],
                "unclear_in": [s for s, r in mine.items() if r["status"] == "unclear"],
                # No match anywhere, or only meaning guesses: no taxonomy has a label for it.
                "not_in_any_taxonomy": bool(mine)
                and "resolved" not in status
                and all(
                    c.get("method") == MEANING_ONLY
                    for r in mine.values()
                    for c in r.get("candidates", ())
                ),
                "connected": bool(near.get(i.key)),
            }
        )
    for s in names:
        gaps[s]["status"] = state.get(s, {}).get("status", "ok")

    verdict_counts = {
        v: sum(1 for r in verdicts if r["verdict"] == v)
        for v in ("confirmed", "indirect", "not_in_graph", "unresolved")
    }

    def clean(rows: Iterable[Json]) -> list[Json]:
        return [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]

    return {
        "policy": POLICY,
        "map_policy": MAP_RANK_V1.name,
        "route_policy": ROUTE_RANK_V1.name,
        "meaning": MEANING,
        "parameters": {
            "depth": depth,
            "breadth": breadth,
            "kinds": kind_list,
            "max_nodes": max_nodes,
            "nodes_per_item": cap,
            "suites": names,
            "max_route_pairs": MAX_ROUTE_PAIRS,
            "indirect_max_edges": INDIRECT_MAX_EDGES,
        },
        "versions": versions,
        "input_notes": list(notes),
        "items": items_out,
        "resolved": resolved_block,
        "summary": {
            "items": len(items),
            "resolved_items": len(resolved_keys),
            "not_in_any_taxonomy": sum(1 for i in items_out if i["not_in_any_taxonomy"]),
            "nodes": len(nodes),
            "edges": len(edges),
            "links": len(links),
            "connectors": len(connectors),
            "broad_connectors": sum(1 for c in connectors if c["broad"]),
            "routes": len(routes),
            "route_pairs_skipped": max(0, len(rest) - MAX_ROUTE_PAIRS),
            "orphans": len(orphans),
            "clusters": len(clusters),
            "guesses": verdict_counts,
        },
        "suites": {n: state.get(n, {"status": "ok"}) for n in names},
        "links": clean(links),
        "connectors": clean(connectors),
        "routes": routes,
        "orphans": orphans,
        "clusters": clusters,
        "guesses": verdicts,
        "gaps": gaps,
        "nodes": [
            nodes[i] | {"items": [text[k] for k in item_nodes.get(i, ())]} for i in sorted(nodes)
        ],
        "edges": sorted(
            edges.values(),
            key=lambda e: (
                e.get("suite") or "~",
                e["graph_edge"],
                *sorted((e["from_id"], e["to_id"])),
            ),
        ),
        "warnings": sorted(warnings),
    }


__all__ = [
    "DEFAULT_DISCOVER_MAX_NODES",
    "INDIRECT_MAX_EDGES",
    "MAX_DISCOVER_NODES",
    "MAX_ROUTE_PAIRS",
    "POLICY",
    "discover",
]

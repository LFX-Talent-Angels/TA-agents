"""The three result pages: map (and route map), successors, roadmap. Built from the JSON the
CLI prints, so a page can be rebuilt from a saved file; no graph or model call happens here.
"""

from __future__ import annotations

from typing import Any

from talent_angels.report.html import (
    TEdge,
    TNode,
    cite,
    esc,
    legend,
    note,
    page,
    section,
    table,
    tree_figure,
    trim,
)
from talent_angels.skills.evaluate.policy import LEARN_ORDER_V1

Json = dict[str, Any]

_KINDS = [
    ("n-skill", "skill"),
    ("n-job", "job"),
    ("n-group", "group"),
    ("n-other", "other"),
]
# Words for a link, read from the box it leaves towards the box it enters.
_RELATION = {
    "related": "related",
    "needs": "needs",
    "needed_by": "job needs it",
    "group": "in group",
    "parent": "broader",
    "narrower": "narrower",
    "level": "has level",
    "same_as": "same as",
}
_REASON = {
    "esco:essential-for": "needs it",
    "esco:narrower": "narrower",
    "esco:related-later-reuse": "related, later",
    "sfia:next-level": "next level",
    "sfia:first-level": "first level",
}


_QUIET = ("ranked:", "learning_order:", "route_scores:")  # route data, not something to act on


def _word(relation: str) -> str:
    return _RELATION.get(relation, relation.replace("_", " "))


def _ids(items: list[Json], limit: int = 5) -> str:
    return "; ".join(f"{c['label']} = {c['id']}" for c in items[:limit])


def _status_note(name: str, part: Json, *, side: dict[str, str] | None = None) -> str:
    status = str(part.get("status", ""))
    cands = part.get("candidates") or []
    if status == "ambiguous" and cands:
        which = f" ({side[part['side']]})" if side and part.get("side") in side else ""
        return f"{name}: the name{which} is ambiguous. Rerun with one of these ids: {_ids(cands)}"
    if status == "no_order_data":
        why = "; ".join(part.get("warnings") or [])
        return f"{name}: no order data for this question" + (f" ({why})" if why else "")
    return f"{name}: {status.replace('_', ' ')}" + (
        f" ({'; '.join(part['warnings'])})" if part.get("warnings") else ""
    )


# ---------------------------------------------------------------- map


def _flat(data: Json) -> tuple[Json, list[Json], list[Json]]:
    """Suites, nodes and edges in one shape, whether the map is one taxonomy or all of them."""
    if "suite" not in data:
        return data["suites"], data["nodes"], data["edges"]
    name = data["suite"]
    route = data.get("route")
    mapped = bool(data["nodes"])
    part: Json = {
        "status": "mapped" if mapped else "not_found",
        "start": data.get("start"),
        "candidates": data.get("candidates") or [],
        "warnings": data["warnings"],
    }
    if route is not None:
        part["route"] = {"node_ids": route}
    nodes = [{**n, "suite": name, "cite": n.get("cite", "")} for n in data["nodes"]]
    edges = [{**e, "suite": name} for e in data["edges"]]
    return {name: part}, nodes, edges


def _map_figure(
    name: str, nodes: list[Json], edges: list[Json], query: str, route: list[str]
) -> str:
    stop = {i: k for k, i in enumerate(route)}
    tnodes = [
        TNode(
            n["id"],
            n["label"],
            n["kind"],
            n["level"],
            f"stop {stop[n['id']] + 1} · {n['kind']}"
            if n["id"] in stop
            else f"{name} · {n['kind']}",
            n["id"] in stop,
        )
        for n in nodes
    ]
    tedges = [
        TEdge(e["from_id"], e["to_id"], _word(e["relation"]), "e",
              f"{_word(e['relation'])} ({e['graph_edge']}"
              + (f", {e['strength']}" if e.get("strength") else "") + ")")
        for e in edges
    ]  # fmt: skip
    items = [*_KINDS, ("e", "link the node was first reached by"), ("e-extra", "another link")]
    if route:
        items = [("e-spine", "the route"), ("n-route", "a stop on the route"), *items]
    steps = "route stops, left" if route else "start, left"
    aria = (
        f"Graph map of {query} in {name}: {len(nodes)} nodes, {len(edges)} links, "
        f"{steps}, further steps to the right"
    )
    titles = ["Route" if route else "Start", *(f"{k} step{'s' * (k > 1)} out" for k in range(1, 6))]
    return tree_figure(tnodes, tedges, items, aria=aria, column_titles=titles, spine=route)


def map_page(data: Json) -> str:
    """The neighbourhood map: one picture per taxonomy, crosswalks, then every node in a table."""
    suites, nodes, edges = _flat(data)
    query = data.get("query") or (
        " → ".join(n["label"] for n in nodes if n.get("on_route")) or data["start"]["label"]
        if nodes
        else "the map"
    )
    route_mode = any(n.get("on_route") for n in nodes)
    by_suite: dict[str, list[Json]] = {}
    for n in nodes:
        by_suite.setdefault(n["suite"], []).append(n)
    side = {"start": "the first name", "target": "the second name"}
    parts: list[str] = []
    for name, part in suites.items():
        mine = by_suite.get(name, [])
        if not mine:
            parts.append(note(_status_note(name, part, side=side), warn=True))
            continue
        here = [e for e in edges if e.get("suite") == name]
        route = list((part.get("route") or {}).get("node_ids") or [])
        found = f"{len(mine)} nodes" + (f", reached {part['via']}" if part.get("via") else "")
        extra = "".join(note(w) for w in part.get("warnings") or [] if not w.startswith(_QUIET))
        parts.append(
            section(
                f"{name}: {found}", extra + _map_figure(name, mine, here, query, route), tag=name
            )
        )
    labels = {n["id"]: f"{n['label']} ({n['suite']})" for n in nodes}
    cross = [e for e in edges if e.get("graph_edge") == "CROSSWALK"]
    if cross:
        rows = [
            [labels.get(e["from_id"], e["from_id"]), labels.get(e["to_id"], e["to_id"]),
             e["strength"], e.get("source", "")]
            for e in cross
        ]  # fmt: skip
        parts.append(
            section(
                "Same thing in another taxonomy (official crosswalk)",
                table(["One side", "Other side", "How close", "Source"], rows),
            )
        )
    parts.append(note(data["meaning"].capitalize() + "."))
    header = ["Step", "Label", "Kind", "Taxonomy", "Source"]
    rows = [[n["level"], n["label"], n["kind"], n["suite"], cite(n.get("cite"))] for n in nodes]
    if route_mode:
        header = ["Step", "Route stop", *header[1:]]
        rows = [
            [r[0], str(n["stop"] + 1) if "stop" in n else "", *r[1:]]
            for r, n in zip(rows, nodes, strict=True)
        ]
    parts.append(section("Every node", table(header, rows)))
    warns = [w for w in data.get("warnings") or [] if isinstance(w, str)]
    lead = note(
        f"{len(nodes)} nodes and {len(edges)} links in {len(by_suite)} taxonomies. "
        + ("Warnings: " + "; ".join(warns) if warns else "")
    )
    return page(f"Graph map: {trim(query, 80)}", "".join(parts), lead=lead)


# ---------------------------------------------------------------- successors


def successors_page(data: Json) -> str:
    """What comes after a skill: a tree, stated links solid and policy links dashed."""
    parts = [
        legend(
            [("e-stated", "stated by the taxonomy"), ("e-policy", "our reading (policy), dashed")]
        )
    ]
    shown = 0
    for name, part in data["suites"].items():
        nodes = part.get("nodes") or []
        if not nodes:
            parts.append(note(_status_note(name, part), warn=part["status"] != "no_order_data"))
            continue
        shown += 1
        tnodes = [
            TNode(n["id"], n["label"], n["kind"], n["depth"],
                  " · ".join(filter(None, [name, n["kind"], n.get("reuse_level")])))
            for n in nodes
        ]  # fmt: skip
        tedges = [
            TEdge(
                e["from_id"], e["to_id"], _REASON.get(e["reason"], e["reason"].split(":")[-1]),
                "e-stated" if e["kind"] == "stated" else "e-policy",
                f"{e['kind']}: {e['reason']}",
            )
            for e in part["edges"]
        ]  # fmt: skip
        aria = (
            f"What comes after {data['x']} in {name}: {len(nodes)} skills, "
            "start on the left, later steps to the right; solid lines are stated by the "
            "taxonomy, dashed lines are policy"
        )
        titles = ["Start", *(f"{k} step{'s' * (k > 1)} later" for k in range(1, 6))]
        fig = tree_figure(
            tnodes, tedges, [*_KINDS, ("e-stated", "stated"), ("e-policy", "policy")],
            aria=aria, column_titles=titles,
        )  # fmt: skip
        extra = "".join(note(w) for w in part.get("warnings") or [])
        rows = [
            [n["depth"], n["label"], n["kind"], n.get("reuse_level") or "", cite(n.get("cite"))]
            for n in nodes
        ]
        kinds = {e["to_id"]: e["kind"] for e in part["edges"]}
        rows = [r + [kinds.get(n["id"], "start")] for r, n in zip(rows, nodes, strict=True)]
        table_html = table(["Step", "Label", "Kind", "Reuse level", "Source", "Link"], rows)
        parts.append(section(f"{name}: {len(nodes)} skills", extra + fig + table_html, tag=name))
    parts.append(note(data["meaning"].capitalize() + "."))
    lead = note(f"After “{data['x']}”, under policy {data['policy']}.")
    if not shown:
        lead += note("No taxonomy returned a tree for this skill.", warn=True)
    return page(f"What comes after: {trim(data['x'], 80)}", "".join(parts), lead=lead)


# ---------------------------------------------------------------- roadmap


def _item(i: Json) -> str:
    levels = i.get("levels")
    jump = ""
    if levels:
        before = "n/a" if levels.get("from") is None else f"{levels['from']:.2f}"
        jump = f'<p class="muted">level {esc(before)}→{levels["to"]:.2f}</p>'
    badge = ' <span class="badge">stated prerequisite</span>' if i["stated_prerequisite"] else ""
    return (
        f'<div class="item"><p><strong>{esc(i["label"])}</strong>{badge}</p>'
        f'<p><span class="tag">{esc(i["suite"])}</span> <span class="muted">{esc(i["kind"])}'
        f"</span></p>{jump}<p>{cite(i['cite'])}</p></div>"
    )


def roadmap_page(data: Json) -> str:
    """From job A to job B: four step columns, then credentials, job zones and notes."""
    by_step = {s["step"]: s for s in data["steps"]}
    cols = []
    for number, title in enumerate(LEARN_ORDER_V1.step_names, start=1):
        step = by_step.get(number, {"items": [], "more": 0})
        body = "".join(_item(i) for i in step["items"]) or '<p class="muted">Nothing here.</p>'
        more = f'<p class="muted">+{step["more"]} more in the JSON</p>' if step["more"] else ""
        cols.append(f'<div class="step"><h3>{number}. {esc(title)}</h3>{body}{more}</div>')
    parts = [section("What to learn, in order", f'<div class="steps">{"".join(cols)}</div>')]
    creds = [
        [c["label"], c["suite"], c.get("entry_education") or "", c.get("work_experience") or "",
         c.get("job_training") or "", cite(c.get("cite"))]
        for c in data["credentials"]
    ]  # fmt: skip
    if creds:
        parts.append(
            section(
                "Credentials the target job asks for (BLS)",
                table(
                    ["Job", "Taxonomy", "Entry education", "Work experience", "Training", "Source"],
                    creds,
                ),
            )
        )
    zone = data.get("job_zone")
    if zone:

        def side(z: Json | None) -> str:
            return z["label"] if z else "not given"

        parts.append(
            section(
                "Job Zone (O*NET)",
                table(["From job", "To job"], [[side(zone.get("from")), side(zone.get("to"))]]),
            )
        )
    notes = [
        note(_status_note(n, p), warn=p["status"] == "ambiguous" or p["status"].startswith("error"))
        for n, p in data["suites"].items()
        if p["status"] not in ("ok", "no_order_data")
    ]
    if notes:
        parts.append(section("Taxonomies that could not answer", "".join(notes)))
    parts.append(note(data["meaning"].capitalize() + "."))
    done = ", ".join(f"{n} {p['items']}" for n, p in data["suites"].items() if p["status"] == "ok")
    lead = note(f"From “{data['from']}” to “{data['to']}” under policy {data['policy']}. " + done)
    return page(
        f"Roadmap: {trim(data['from'], 40)} → {trim(data['to'], 40)}", "".join(parts), lead=lead
    )

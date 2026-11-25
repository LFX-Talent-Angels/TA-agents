"""Discovery map outputs: discovery.json, discovery.xlsx (for non-engineers), discovery.html.

All three are written from the one dict ``discover()`` returns. The JSON and the HTML are
deterministic (no dates, no run ids); the HTML is one static page (shared renderer, light and
dark) with no script or CDN.
"""

from __future__ import annotations

import json
from io import BytesIO
from math import cos, pi, sin
from pathlib import Path
from typing import Any

from talent_angels.llm.protocol import check_cancelled
from talent_angels.report.html import cite, esc, legend, note, page, table, trim
from talent_angels.report.html import section as section_html

Json = dict[str, Any]

_STATUS = {"resolved": "found", "unclear": "unclear (pick one)", "not_found": "not found"}
_VERDICT = {
    "confirmed": "Yes, linked directly",
    "indirect": "Yes, connected indirectly",
    "not_in_graph": "Not in the graph",
    "unresolved": "Cannot check",
}


def _tables(data: Json) -> dict[str, tuple[list[str], list[list[Any]]]]:
    """Tab name -> (header, rows), worded for people who do not read JSON."""
    matched: dict[str, list[str]] = {}
    for r in data["resolved"]:
        if r["status"] == "resolved":
            matched.setdefault(r["item"], []).append(f"{r['suite']}: {r['label']}")
    items = [
        [
            i["item"],
            i["kind"] or "",
            _STATUS[i["status"]],
            "; ".join(matched.get(i["item"], [])),
            ", ".join(i["unclear_in"]),
            "yes" if i["not_in_any_taxonomy"] else "",
            "yes" if i["connected"] else ("no" if i["resolved_in"] else ""),
        ]
        for i in data["items"]
    ]
    gaps = [
        [
            suite,
            row["item"],
            "unclear",
            "; ".join(f"{c['label']} ({c['id']})" for c in row["candidates"]),
        ]
        for suite, g in data["gaps"].items()
        for row in g["unclear"]
    ] + [
        [suite, item, "not found", ""]
        for suite, g in data["gaps"].items()
        for item in g["not_found"]
    ]
    return {
        "Items": (
            [
                "Item",
                "Your kind",
                "Result",
                "Matched to",
                "Unclear in",
                "In no taxonomy",
                "Connected",
            ],
            items,
        ),
        "Connectors": (
            [
                "What connects them",
                "Type",
                "Taxonomy",
                "Too broad to be useful",
                "Only a tool link",
                "Items",
                "How many",
                "Source",
            ],
            [
                [
                    c["label"],
                    c["kind"],
                    c["suite"],
                    "yes" if c["broad"] else "",
                    "yes" if c["tool_only"] else "",
                    "; ".join(x["item"] for x in c["items"]),
                    c["item_count"],
                    c["cite"],
                ]
                for c in data["connectors"]
            ],
        ),  # fmt: skip
        "Links": (
            ["Item A", "Item B", "How they are linked", "Strength", "Taxonomy"],
            [
                [x["a"], x["b"], x["relation"], x["strength"] or "", x["suite"]]
                for x in data["links"]
            ],
        ),
        "Routes": (
            ["Item A", "Item B", "Taxonomy", "Steps", "Path", "Score"],
            [
                [
                    r["a"],
                    r["b"],
                    r["suite"],
                    r["edges"],
                    " > ".join(s["label"] for s in r["stops"]),
                    r["score"],
                ]
                for r in data["routes"]
            ],
        ),
        "Orphans": (
            ["Item that connects to nothing else", "Every possible route was checked"],
            [[o["item"], "yes" if o["routes_fully_checked"] else "no"] for o in data["orphans"]],
        ),
        "Clusters": (
            ["Cluster", "Size", "Items", "Main connectors"],
            [
                [c["name"], c["size"], "; ".join(c["items"]), "; ".join(c["connectors"])]
                for c in data["clusters"]
            ],
        ),
        "Guesses": (
            ["Item", "You expected it to go with", "Does the graph agree?", "Why"],
            [
                [g["item"], g["related_to"], _VERDICT[g["verdict"]], g["detail"]]
                for g in data["guesses"]
            ],
        ),
        "Gaps": (["Taxonomy", "Item", "Result", "Possible matches"], gaps),
    }


def write_xlsx(data: Json, path: Path | BytesIO) -> None:
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font

    book = Workbook()
    book.remove(book.active)  # type: ignore[arg-type]
    for title, (header, rows) in _tables(data).items():
        sheet = book.create_sheet(title)
        sheet.append(header)
        for row in rows:
            sheet.append(row)
        for cell in sheet[1]:
            cell.font = Font(bold=True)
        for col, head in enumerate(header, 1):
            width = max([len(head), *(len(str(r[col - 1])) for r in rows)])
            sheet.column_dimensions[sheet.cell(1, col).column_letter].width = min(60, width + 2)
        for row in sheet.iter_rows(min_row=2):
            for cell in row:
                cell.alignment = Alignment(wrap_text=True, vertical="top")
        sheet.freeze_panes = "A2"
    book.save(path)


_RING_LABEL = 24  # characters of an item name drawn beside its dot
_MAP_CONNECTORS = 12  # connectors drawn and numbered; the table lists them all


def _svg(data: Json) -> str:
    """Items on a ring, direct links as lines, the top connectors as numbered squares.

    Item names sit outside the ring and the connector names in a panel on the right, so no
    text is ever drawn on top of a line. Tool-only and broad connectors stay off the picture.
    """
    shown = [i["item"] for i in data["items"] if i["resolved_in"]]
    if len(shown) < 2:
        return ""
    cx, cy, radius = 370.0, 290.0, 190.0
    at = {
        item: (
            cx + radius * cos(2 * pi * k / len(shown) - pi / 2),
            cy + radius * sin(2 * pi * k / len(shown) - pi / 2),
        )
        for k, item in enumerate(shown)
    }
    top = [c for c in data["connectors"] if not c["broad"] and not c["tool_only"]]
    top = [c for c in top if any(x["item"] in at for x in c["items"])][:_MAP_CONNECTORS]
    aria = (
        f"Items and what connects them: {len(shown)} items on a ring, "
        f"{len(data['links'])} direct links, the top {len(top)} connectors numbered and named "
        "on the right"
    )
    out = [f'<svg class="ring" viewBox="0 0 1030 580" role="img" aria-label="{esc(aria)}">']
    for x in data["links"]:
        (x1, y1), (x2, y2) = at[x["a"]], at[x["b"]]
        out.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" class="link"/>')
    marks = []
    for n, c in enumerate(top, 1):
        pts = [at[x["item"]] for x in c["items"] if x["item"] in at]
        mx, my = (
            (cx + sum(p[0] for p in pts) / len(pts)) / 2,
            (cy + sum(p[1] for p in pts) / len(pts)) / 2,
        )
        for px, py in pts:
            out.append(
                f'<line x1="{mx:.1f}" y1="{my:.1f}" x2="{px:.1f}" y2="{py:.1f}" class="via"/>'
            )
        marks.append(
            f"<g><title>{n}. {esc(c['label'])} ({esc(c['suite'])})</title>"
            f'<rect x="{mx - 8:.1f}" y="{my - 8:.1f}" width="16" height="16" rx="3" class="conn"/>'
            f'<text x="{mx:.1f}" y="{my + 3.5:.1f}" text-anchor="middle" class="num">{n}</text></g>'
        )
    out += marks
    lonely = {o["item"] for o in data["orphans"]}
    for item, (px, py) in at.items():
        side = "start" if px > cx + 1 else ("end" if px < cx - 1 else "middle")
        dx = 10 if side == "start" else (-10 if side == "end" else 0)
        dy = 0 if side != "middle" else (-10 if py < cy else 18)
        look = "orphan-dot" if item in lonely else "item-dot"
        out.append(
            f"<g><title>{esc(item)}</title>"
            f'<circle cx="{px:.1f}" cy="{py:.1f}" r="5" class="{look}"/>'
            f'<text class="t" x="{px + dx:.1f}" y="{py + dy + 4:.1f}" text-anchor="{side}">'
            f"{esc(trim(item, _RING_LABEL))}</text></g>"
        )
    out.append(
        '<rect class="panel" x="760" y="20" width="262" height="540" rx="8"/>'
        '<text class="col" x="774" y="42">Top connectors</text>'
    )
    for n, c in enumerate(top, 1):
        y = 66 + (n - 1) * 40
        out.append(
            f"<g><title>{esc(c['label'])} ({esc(c['suite'])}): "
            f"{esc('; '.join(x['item'] for x in c['items']))}</title>"
            f'<rect x="774" y="{y - 10}" width="16" height="16" rx="3" class="conn"/>'
            f'<text x="782" y="{y + 2}" text-anchor="middle" class="num">{n}</text>'
            f'<text class="t" x="798" y="{y + 2}">{esc(trim(c["label"], 28))}</text>'
            f'<text class="t2" x="798" y="{y + 16}">{esc(c["suite"])} · '
            f"{c['item_count']} items</text></g>"
        )
    if not top:
        out.append('<text class="t2" x="774" y="66">No specific connector to draw.</text>')
    return "".join(out) + "</svg>"


def render_html(data: Json) -> str:
    tabs = _tables(data)
    s = data["summary"]
    g = s["guesses"]
    tool = sum(1 for c in data["connectors"] if c["tool_only"])
    facts = [
        f"{s['items']} items; {s['resolved_items']} found in at least one taxonomy",
        f"{s['links']} direct links, {s['connectors']} shared connectors "
        f"({s['broad_connectors']} too broad to be useful, {tool} linked only through a tool), "
        f"{s['routes']} longer routes",
        f"{s['orphans']} items connect to nothing else; "
        f"{s['not_in_any_taxonomy']} are in no taxonomy",
        f"Your guesses: {g['confirmed']} linked directly, {g['indirect']} connected indirectly, "
        f"{g['not_in_graph']} not in the graph, {g['unresolved']} could not be checked",
    ]
    clusters = (
        "".join(
            f"<h3>{esc(c['name'])} ({c['size']})</h3><p>{esc('; '.join(c['items']))}</p>"
            f'<p class="note">Main connectors: {esc("; ".join(c["connectors"]) or "none")}</p>'
            for c in data["clusters"]
        )
        or "<p>No group of two or more connected items.</p>"
    )
    shown = data["connectors"][:40]
    header, rows = tabs["Connectors"]
    picture = _svg(data)
    if picture:
        picture = (
            legend(
                [
                    ("n-job", "item (dot) on the ring"),
                    ("e-stated", "direct link"),
                    ("n-other", "numbered square: a shared connector"),
                ]
            )
            + f'<div class="scroll">{picture}</div>'
        )
    source = header.index("Source")
    linked = [[*r[:source], cite(r[source]), *r[source + 1 :]] for r in rows[:40]]
    connectors = table(header, linked, dim=[i for i, c in enumerate(shown) if c["tool_only"]])
    if tool:
        connectors = (
            note(
                "Greyed rows link the items only because a job lists both as software it "
                "uses (O*NET USES_SOFTWARE). They rank last and are not drawn on the map."
            )
            + connectors
        )
    sections = [
        ("Summary", "<ul>" + "".join(f"<li>{esc(f)}</li>" for f in facts) + "</ul>"),
        ("Map", picture or "<p>Not enough connected items to draw.</p>"),
        ("Clusters", clusters),
        ("What connects your items", connectors),
        ("Your guesses", table(*tabs["Guesses"])),
        ("Items that connect to nothing", table(*tabs["Orphans"])),
        ("Not found or unclear", table(*tabs["Gaps"])),
    ]
    body = "".join(section_html(t, h) for t, h in sections)
    lead = note(
        f"{data['meaning']} ({data['policy']}). Every link below comes from the official "
        "taxonomy graphs."
    )
    return page("Discovery map", body, lead=lead)


def write_outputs(data: Json, out_dir: str | Path) -> dict[str, Path]:
    out = Path(out_dir)
    paths = {k: out / f"discovery.{k}" for k in ("json", "xlsx", "html")}
    text = json.dumps(data, ensure_ascii=False, indent=2) + "\n"
    html = render_html(data)
    book = BytesIO()
    write_xlsx(data, book)  # everything is built first: a stop writes none of the three
    check_cancelled()
    out.mkdir(parents=True, exist_ok=True)
    paths["json"].write_text(text, encoding="utf-8")
    paths["xlsx"].write_bytes(book.getvalue())
    paths["html"].write_text(html, encoding="utf-8")
    return paths


def discover_summary(data: Json) -> str:
    """The three-line count summary the CLI and the chat command print."""
    s, g = data["summary"], data["summary"]["guesses"]
    return (
        f"{s['items']} items: {s['resolved_items']} found, "
        f"{s['not_in_any_taxonomy']} in no taxonomy\n"
        f"{s['links']} links, {s['connectors']} connectors, {s['routes']} routes, "
        f"{s['orphans']} orphans, {s['clusters']} clusters\n"
        f"guesses: {g['confirmed']} confirmed, {g['indirect']} indirect, "
        f"{g['not_in_graph']} not in graph, {g['unresolved']} unresolved"
    )


__all__ = ["discover_summary", "render_html", "write_outputs", "write_xlsx"]

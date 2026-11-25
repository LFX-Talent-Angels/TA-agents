"""Discovery map outputs: discovery.json, discovery.xlsx (for non-engineers), discovery.html.

All three are written from the one dict ``discover()`` returns. The JSON and the HTML are
deterministic (no dates, no run ids); the HTML is one static page with no script or CDN.
"""

from __future__ import annotations

import json
from html import escape
from math import cos, pi, sin
from pathlib import Path
from typing import Any

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


def write_xlsx(data: Json, path: Path) -> None:
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


def _svg(data: Json) -> str:
    """Resolved items on a ring, direct links as lines, the main connectors as small squares."""
    shown = [i["item"] for i in data["items"] if i["resolved_in"]]
    if len(shown) < 2:
        return ""
    cx, cy, radius = 340.0, 290.0, 210.0
    at = {
        item: (
            cx + radius * cos(2 * pi * k / len(shown) - pi / 2),
            cy + radius * sin(2 * pi * k / len(shown) - pi / 2),
        )
        for k, item in enumerate(shown)
    }
    out = ['<svg viewBox="0 0 680 580" role="img" aria-label="Items and what connects them">']
    for x in data["links"]:
        (x1, y1), (x2, y2) = at[x["a"]], at[x["b"]]
        out.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" class="link"/>')
    for c in [c for c in data["connectors"] if not c["broad"]][:12]:
        pts = [at[x["item"]] for x in c["items"] if x["item"] in at]
        mx, my = (
            (cx + sum(p[0] for p in pts) / len(pts)) / 2,
            (cy + sum(p[1] for p in pts) / len(pts)) / 2,
        )
        for px, py in pts:
            out.append(
                f'<line x1="{mx:.1f}" y1="{my:.1f}" x2="{px:.1f}" y2="{py:.1f}" class="via"/>'
            )
        out.append(
            f'<rect x="{mx - 4:.1f}" y="{my - 4:.1f}" width="8" height="8" class="conn">'
            f"<title>{escape(c['label'])}</title></rect>"
        )
    lonely = {o["item"] for o in data["orphans"]}
    for item, (px, py) in at.items():
        side = "start" if px > cx + 1 else ("end" if px < cx - 1 else "middle")
        dx = 10 if side == "start" else (-10 if side == "end" else 0)
        dy = 0 if side != "middle" else (-10 if py < cy else 18)
        look = "orphan" if item in lonely else "item"
        out.append(
            f'<circle cx="{px:.1f}" cy="{py:.1f}" r="5" class="{look}"/>'
            f'<text x="{px + dx:.1f}" y="{py + dy + 4:.1f}" text-anchor="{side}">'
            f"{escape(item)}</text>"
        )
    return "".join(out) + "</svg>"


def _table(header: list[str], rows: list[list[Any]]) -> str:
    if not rows:
        return "<p>None.</p>"
    head = "".join(f"<th>{escape(h)}</th>" for h in header)
    body = "".join(
        "<tr>" + "".join(f"<td>{escape(str(c))}</td>" for c in row) + "</tr>" for row in rows
    )
    return f"<table><tr>{head}</tr>{body}</table>"


def render_html(data: Json) -> str:
    tabs = _tables(data)
    s = data["summary"]
    g = s["guesses"]
    facts = [
        f"{s['items']} items; {s['resolved_items']} found in at least one taxonomy",
        f"{s['links']} direct links, {s['connectors']} shared connectors "
        f"({s['broad_connectors']} too broad to be useful), {s['routes']} longer routes",
        f"{s['orphans']} items connect to nothing else; "
        f"{s['not_in_any_taxonomy']} are in no taxonomy",
        f"Your guesses: {g['confirmed']} linked directly, {g['indirect']} connected indirectly, "
        f"{g['not_in_graph']} not in the graph, {g['unresolved']} could not be checked",
    ]
    clusters = (
        "".join(
            f"<h3>{escape(c['name'])} ({c['size']})</h3><p>{escape('; '.join(c['items']))}</p>"
            f'<p class="note">Main connectors: {escape("; ".join(c["connectors"]) or "none")}</p>'
            for c in data["clusters"]
        )
        or "<p>No group of two or more connected items.</p>"
    )
    connectors = tabs["Connectors"]
    sections = [
        ("Summary", "<ul>" + "".join(f"<li>{escape(f)}</li>" for f in facts) + "</ul>"),
        ("Map", _svg(data) or "<p>Not enough connected items to draw.</p>"),
        ("Clusters", clusters),
        ("What connects your items", _table(connectors[0], connectors[1][:40])),
        ("Your guesses", _table(*tabs["Guesses"])),
        ("Items that connect to nothing", _table(*tabs["Orphans"])),
        ("Not found or unclear", _table(*tabs["Gaps"])),
    ]
    body = "".join(f"<section><h2>{escape(t)}</h2>{h}</section>" for t, h in sections)
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        "<title>Discovery map</title><style>"
        "body{font:15px/1.5 system-ui,sans-serif;max-width:900px;margin:2rem auto;"
        "padding:0 1rem;color:#1b1b1f}"
        "table{border-collapse:collapse;width:100%;font-size:14px}"
        "th,td{border:1px solid #ccd;padding:.3rem .5rem;text-align:left;vertical-align:top}"
        "th{background:#eef}.note{color:#555}svg{width:100%;height:auto}"
        "text{font-size:12px;fill:#1b1b1f}.link{stroke:#223;stroke-width:1.5}"
        ".via{stroke:#99a;stroke-width:1}.conn{fill:#c60}.item{fill:#235}.orphan{fill:#999}"
        "</style></head><body>"
        f'<h1>Discovery map</h1><p class="note">{escape(data["meaning"])} '
        f"({escape(data['policy'])}). Every link below comes from the official taxonomy graphs.</p>"
        f"{body}</body></html>\n"
    )


def write_outputs(data: Json, out_dir: str | Path) -> dict[str, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {k: out / f"discovery.{k}" for k in ("json", "xlsx", "html")}
    paths["json"].write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_xlsx(data, paths["xlsx"])
    paths["html"].write_text(render_html(data), encoding="utf-8")
    return paths


__all__ = ["render_html", "write_outputs", "write_xlsx"]

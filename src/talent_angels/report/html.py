"""The one HTML renderer behind every report page (map, successors, roadmap, discovery).

A page is self-contained and static: inline CSS, no script, no external URL except the
citations a table links to. Colours are CSS variables with a dark-mode set; the body paints
its own background. Pictures are inline SVG that scales; tables scroll sideways inside a
wrapper, so a phone never scrolls the page. Same input gives the same bytes: no clock, no
random ids, coordinates are integers computed from the order of the data it is given.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from html import escape as esc
from typing import Any

CSS = """
:root{--bg:#fff;--text:#1b1b1f;--muted:#5a5a66;--panel:#f4f4f8;--border:#cfcfda;--line:#6c6c80;
--faint:#c9c9d6;--accent:#2b5fc2;--spine:#d9480f;--warn:#8a5a00;--warn-bg:#fff4d6;
--skill-f:#e3f4ec;--skill-s:#1c7c54;--job-f:#e5edfb;--job-s:#2b5fc2;
--group-f:#f3ecf9;--group-s:#7a45b0;--other-f:#eeeef2;--other-s:#70707c}
@media (prefers-color-scheme:dark){:root{--bg:#121216;--text:#e8e8ee;--muted:#a3a3b2;
--panel:#1c1c22;--border:#3a3a46;--line:#9a9aae;--faint:#40404d;--accent:#8ab4ff;
--spine:#ff9f64;--warn:#f5c26b;--warn-bg:#3a2e12;
--skill-f:#12342a;--skill-s:#4cc38a;--job-f:#14263f;--job-s:#6ea2ff;
--group-f:#2a1d3a;--group-s:#b98af0;--other-f:#26262e;--other-s:#9a9aa6}}
html{color-scheme:light dark}
body{background:var(--bg);color:var(--text);margin:0;
font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1100px;margin:0 auto;padding:1.25rem 1rem 3rem}
h1{font-size:1.5rem;margin:.2rem 0 .4rem}h2{font-size:1.15rem;margin:1.6rem 0 .5rem}
h3{font-size:1rem;margin:1rem 0 .3rem}
a{color:var(--accent)}.note,.muted{color:var(--muted)}
.callout{background:var(--warn-bg);color:var(--warn);border-radius:6px;padding:.4rem .7rem;
margin:.5rem 0}
.scroll{overflow-x:auto;-webkit-overflow-scrolling:touch}.scroll+.scroll,.scroll+h3{margin-top:1rem}
table{border-collapse:collapse;width:100%;font-size:14px}
th,td{border:1px solid var(--border);padding:.3rem .5rem;text-align:left;vertical-align:top}
th{background:var(--panel)}tr.dim td{color:var(--muted);font-style:italic}
.tag{display:inline-block;border:1px solid var(--border);border-radius:999px;padding:0 .55em;
font-size:12px;color:var(--muted)}
.badge{display:inline-block;border:1px solid var(--accent);color:var(--accent);
border-radius:4px;padding:0 .4em;font-size:12px}
.legend{list-style:none;display:flex;flex-wrap:wrap;gap:.3rem 1.1rem;padding:0;margin:.4rem 0;
font-size:13px;color:var(--muted)}
.legend li{display:flex;align-items:center;gap:.4rem}.sw{width:30px;height:14px}
.steps{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:.75rem}
.step{background:var(--panel);border:1px solid var(--border);border-radius:8px;padding:.6rem .75rem}
.step h3{margin:.1rem 0 .5rem}.item{border-top:1px solid var(--border);padding:.45rem 0}
.item:first-of-type{border-top:0}.item p{margin:.1rem 0}
svg.tree,svg.ring{display:block;width:100%;height:auto;min-width:560px}
svg text{font-family:inherit}.t{fill:var(--text);font-size:12px}
.t2{fill:var(--muted);font-size:10px}.col{fill:var(--muted);font-size:11px;font-weight:600}
.el{fill:var(--muted);font-size:10px;paint-order:stroke;stroke:var(--bg);stroke-width:3px}
.n-skill{fill:var(--skill-f);stroke:var(--skill-s);stroke-width:1.5}
.n-job{fill:var(--job-f);stroke:var(--job-s);stroke-width:1.5}
.n-group{fill:var(--group-f);stroke:var(--group-s);stroke-width:1.5}
.n-other{fill:var(--other-f);stroke:var(--other-s);stroke-width:1.5}
.n-route{stroke:var(--spine);stroke-width:3}
.e,.e-stated,.e-policy,.e-extra,.e-spine{fill:none;stroke:var(--line);stroke-width:1.4}
.e-policy{stroke-dasharray:6 4}.e-extra{stroke:var(--faint);stroke-width:1}
.e-spine{stroke:var(--spine);stroke-width:4}
.link{stroke:var(--line);stroke-width:1.5}.via{stroke:var(--faint);stroke-width:1}
.conn{fill:#c60;stroke:var(--bg)}.item-dot{fill:var(--job-s)}.orphan-dot{fill:var(--other-s)}
.num{fill:#fff;font-size:10px;font-weight:700}
.panel{fill:var(--panel);stroke:var(--border)}
"""

_SKILL = {"skill", "knowledge", "ability", "technology", "software", "tool", "competence"}
_JOB = {"occupation", "job"}
_GROUPISH = ("group", "category", "level", "headline", "headings")


class Raw(str):
    """A cell that is already HTML (a link): the table does not escape it."""


def kind_class(kind: str) -> str:
    """skill | job | group | other: what a colour stands for."""
    k = kind.casefold()
    if k in _JOB:
        return "job"
    if k in _SKILL:
        return "skill"
    return "group" if k.endswith(_GROUPISH) or k.startswith(_GROUPISH) else "other"


def trim(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def cite(value: object) -> Raw | str:
    """A link when the citation is a web address, the plain text otherwise."""
    text = str(value or "")
    if text.startswith(("http://", "https://")):
        return Raw(f'<a href="{esc(text)}" rel="noopener noreferrer">{esc(trim(text, 48))}</a>')
    return text


def _cell(value: Any) -> str:
    return str(value) if isinstance(value, Raw) else esc(str(value))


def table(header: Sequence[str], rows: Sequence[Sequence[Any]], *, dim: Iterable[int] = ()) -> str:
    """A table inside a sideways-scrolling wrapper; rows in ``dim`` are greyed."""
    if not rows:
        return "<p>None.</p>"
    grey = set(dim)
    head = "".join(f"<th>{esc(h)}</th>" for h in header)
    body = "".join(
        f'<tr{" class=dim" if i in grey else ""}>' + "".join(f"<td>{_cell(c)}</td>" for c in row)
        + "</tr>"
        for i, row in enumerate(rows)
    )  # fmt: skip
    return f'<div class="scroll"><table><tr>{head}</tr>{body}</table></div>'


def legend(items: Sequence[tuple[str, str]]) -> str:
    """``(css class, text)``: ``n-*`` is a box swatch, ``e-*`` a line swatch."""
    out = []
    for cls, text in items:
        if cls.startswith("n-"):
            mark = f'<rect class="{cls}" x="2" y="1" width="26" height="12" rx="3"/>'
        else:
            mark = f'<line class="{cls}" x1="1" y1="7" x2="29" y2="7"/>'
        out.append(
            f'<li><svg class="sw" viewBox="0 0 30 14" aria-hidden="true">{mark}</svg>'
            f"{esc(text)}</li>"
        )
    return f'<ul class="legend" aria-label="Legend">{"".join(out)}</ul>'


def section(title: str, body: str, *, tag: str | None = None) -> str:
    chip = f' <span class="tag">{esc(tag)}</span>' if tag else ""
    return f"<section><h2>{esc(title)}{chip}</h2>{body}</section>"


def note(text: str, *, warn: bool = False) -> str:
    return f'<p class="callout">{esc(text)}</p>' if warn else f'<p class="note">{esc(text)}</p>'


def page(title: str, body: str, *, lead: str = "") -> str:
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>{esc(title)}</title><style>{CSS}</style></head><body><main>"
        f"<h1>{esc(title)}</h1>{lead}{body}</main></body></html>\n"
    )


# ---- the layered tree: column = level, one row per node, left to right ----

_BOX_W, _BOX_H, _GUT, _PITCH, _PAD, _TOP = 190, 38, 120, 46, 40, 30


@dataclass(frozen=True)
class TNode:
    id: str
    label: str
    kind: str
    level: int
    sub: str = ""  # the small second line: taxonomy, kind, ...
    on_route: bool = False


@dataclass(frozen=True)
class TEdge:
    a: str
    b: str
    label: str = ""
    style: str = "e"  # e | e-stated | e-policy
    title: str = ""


def _layout(nodes: Sequence[TNode], edges: Sequence[TEdge]) -> tuple[list[str], dict[str, TEdge]]:
    """Preorder rows, and the edge each node was first reached by (the first one in the list)."""
    level = {n.id: n.level for n in nodes}
    parent: dict[str, TEdge] = {}
    for e in edges:
        if e.a not in level or e.b not in level:
            continue
        if e.b not in parent and level[e.b] == level[e.a] + 1:
            parent[e.b] = e
        elif e.a not in parent and level[e.a] == level[e.b] + 1:
            parent[e.a] = TEdge(e.b, e.a, e.label, e.style, e.title)
    kids: dict[str, list[str]] = {}
    for n in nodes:
        if n.id in parent:
            kids.setdefault(parent[n.id].a, []).append(n.id)
    rows: list[str] = []
    placed: set[str] = set()
    stack = [n.id for n in reversed(nodes) if n.id not in parent]
    while stack:
        nid = stack.pop()
        if nid in placed:
            continue
        placed.add(nid)
        rows.append(nid)
        stack.extend(reversed(kids.get(nid, [])))
    return rows, parent


def tree_svg(
    nodes: Sequence[TNode],
    edges: Sequence[TEdge],
    *,
    aria: str,
    column_titles: Sequence[str] = (),
    spine: Sequence[str] = (),
) -> str:
    """Boxes in columns by level; the first-reached edge drawn solid with its label, any other
    link faint behind the boxes; ``spine`` (node IDs in order) as one thick route line."""
    rows, parent = _layout(nodes, edges)
    by_id = {n.id: n for n in nodes}
    columns = max((n.level for n in nodes), default=0) + 1
    row_of = {nid: i for i, nid in enumerate(rows)}
    width = _PAD + columns * (_BOX_W + _GUT) - _GUT + 12
    height = _TOP + max(1, len(rows)) * _PITCH

    def x(n: TNode) -> int:
        return _PAD + n.level * (_BOX_W + _GUT)

    def y(n: TNode) -> int:
        return _TOP + row_of[n.id] * _PITCH

    mid = _BOX_H // 2
    out = [
        f'<svg class="tree" viewBox="0 0 {width} {height}" style="max-width:{width}px" '
        f'role="img" aria-label="{esc(aria)}">'
    ]
    for level in range(columns):
        title = column_titles[level] if level < len(column_titles) else f"{level} steps out"
        out.append(
            f'<text class="col" x="{_PAD + level * (_BOX_W + _GUT)}" y="16">{esc(title)}</text>'
        )
    drawn: set[str] = set()
    for nid, e in parent.items():  # solid first-reached edges, with their relation word
        if nid not in row_of or e.a not in row_of:
            continue
        a, b = by_id[e.a], by_id[nid]
        bus = x(a) + _BOX_W + 14
        ya, yb = y(a) + mid, y(b) + mid
        drawn.add(f"{e.a}|{nid}")
        out.append(
            f'<path class="{e.style}" d="M{x(a) + _BOX_W} {ya}H{bus}V{yb}H{x(b)}">'
            f"<title>{esc(e.title or e.label)}</title></path>"
        )
        if e.label:
            out.append(
                f'<text class="el" x="{bus + 6}" y="{yb - 4}">{esc(trim(e.label, 18))}</text>'
            )
    extras: list[str] = []
    for e in edges:  # every other link between two drawn nodes: faint, behind the boxes
        if e.a not in row_of or e.b not in row_of or f"{e.a}|{e.b}" in drawn:
            continue
        if f"{e.b}|{e.a}" in drawn:
            continue
        a, b = by_id[e.a], by_id[e.b]
        if a.level > b.level:
            a, b = b, a
        ya, yb = y(a) + mid, y(b) + mid
        if a.level == b.level:
            xl = x(a)
            d = f"M{xl} {ya}C{xl - 34} {ya} {xl - 34} {yb} {xl} {yb}"
        else:
            d = f"M{x(a) + _BOX_W} {ya}L{x(b)} {yb}"
        extras.append(
            f'<path class="e-extra" d="{d}"><title>{esc(e.title or e.label)}</title></path>'
        )
    out.extend(extras)
    stops = [by_id[s] for s in spine if s in row_of]
    for a, b in zip(stops, stops[1:], strict=False):
        ya, yb = y(a) + mid, y(b) + mid
        out.append(
            f'<path class="e-spine" d="M{x(a)} {ya}H{x(a) - 16}V{yb}H{x(b)}">'
            "<title>route</title></path>"
        )
    for nid in rows:
        n = by_id[nid]
        cls = f"n-{kind_class(n.kind)}" + (" n-route" if n.on_route else "")
        px, py = x(n), y(n)
        out.append(
            f"<g><title>{esc(n.label)} ({esc(n.kind)}; {esc(n.id)})</title>"
            f'<rect class="{cls}" x="{px}" y="{py}" width="{_BOX_W}" height="{_BOX_H}" rx="6"/>'
            f'<text class="t" x="{px + 8}" y="{py + 16}">{esc(trim(n.label, 27))}</text>'
            f'<text class="t2" x="{px + 8}" y="{py + 30}">{esc(trim(n.sub, 36))}</text></g>'
        )
    return "".join(out) + "</svg>"


def tree_figure(
    nodes: Sequence[TNode],
    edges: Sequence[TEdge],
    legend_items: Sequence[tuple[str, str]],
    *,
    aria: str,
    column_titles: Sequence[str] = (),
    spine: Sequence[str] = (),
) -> str:
    svg = tree_svg(nodes, edges, aria=aria, column_titles=column_titles, spine=spine)
    return f'{legend(legend_items)}<div class="scroll">{svg}</div>'

"""Colored Rich chrome for ta-agent. Local paint only — no extra LLM or graph."""

from __future__ import annotations

import re
from dataclasses import dataclass

from rich.console import Console, Group, RenderableType
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from talent_angels.session.kernel import ChatReply

_NUMBERED = re.compile(
    r"^(\d+)\.\s+(.+?)(?:\s+\((essential|optional)\))?\s*$",
    re.IGNORECASE,
)
_GROUP = re.compile(r"^\*\*(.+?)\*\*\s*$")
_SKILLS_HEAD = re.compile(r"—\s+\d+\s+skills", re.IGNORECASE)


@dataclass(frozen=True)
class ParsedReply:
    intro: str
    rows: tuple[tuple[str, str, str, str], ...]  # number, label, tag, group
    footer: str


def parse_list_reply(text: str) -> ParsedReply:
    """Split picker/skill markdown into intro, numbered rows, footer."""
    lines = text.splitlines()
    intro: list[str] = []
    rows: list[tuple[str, str, str, str]] = []
    footer: list[str] = []
    group = ""
    seen_row = False
    for line in lines:
        stripped = line.strip()
        numbered = _NUMBERED.match(stripped)
        heading = _GROUP.match(stripped)
        if numbered:
            seen_row = True
            rows.append(
                (
                    numbered.group(1),
                    numbered.group(2).strip(),
                    (numbered.group(3) or "").lower(),
                    group,
                )
            )
            continue
        if heading and not _SKILLS_HEAD.search(stripped):
            group = heading.group(1)
            if not seen_row:
                # skill-list title line is markdown bold + em dash, not an ISCO group
                if "skills on the map" in stripped.casefold():
                    intro.append(line)
                    continue
            if not seen_row and not rows:
                # group headings start the table; keep pre-heading prose as intro
                pass
            continue
        if not seen_row:
            intro.append(line)
        else:
            footer.append(line)
    return ParsedReply(
        intro="\n".join(intro).strip(),
        rows=tuple(rows),
        footer="\n".join(footer).strip(),
    )


def _tag_style(tag: str) -> str:
    if tag == "essential":
        return "green"
    if tag == "optional":
        return "yellow"
    return "dim"


def _list_table(rows: tuple[tuple[str, str, str, str], ...]) -> Table:
    show_group = any(row[3] for row in rows)
    show_tag = any(row[2] for row in rows)
    table = Table(
        show_header=bool(show_group or show_tag),
        box=None,
        pad_edge=False,
        expand=True,
        padding=(0, 1),
    )
    table.add_column("#", style="bold cyan", width=4, no_wrap=True)
    if show_group:
        table.add_column("Group", style="cyan", overflow="fold")
    table.add_column("Title", overflow="fold")
    if show_tag:
        table.add_column("Kind", width=10, no_wrap=True)
    last_group = None
    for number, label, tag, group in rows:
        cells: list[Text | str] = [number]
        if show_group:
            heading = group if group != last_group else ""
            last_group = group
            cells.append(Text(heading, style="bold cyan") if heading else "")
        cells.append(label)
        if show_tag:
            cells.append(Text(tag, style=_tag_style(tag)) if tag else "")
        table.add_row(*cells)
    return table


def _chunk_body(text: str) -> RenderableType:
    parsed = parse_list_reply(text)
    if len(parsed.rows) < 3:
        return Markdown(text)
    parts: list[RenderableType] = []
    if parsed.intro:
        parts.append(Markdown(parsed.intro))
    parts.append(_list_table(parsed.rows))
    if parsed.footer:
        parts.append(Markdown(parsed.footer))
    return Group(*parts)


def _assistant_body(text: str) -> RenderableType:
    """One table per `---` block so ESCO and O*NET pickers do not merge."""
    chunks = [chunk.strip() for chunk in re.split(r"\n-{3,}\n", text) if chunk.strip()]
    if len(chunks) <= 1:
        return _chunk_body(text)
    parts: list[RenderableType] = []
    for index, chunk in enumerate(chunks):
        if index:
            parts.append(Rule(style="bright_black"))
        parts.append(_chunk_body(chunk))
    return Group(*parts)


def render_welcome(console: Console, welcome: str) -> None:
    console.print(
        Panel(
            Markdown(welcome),
            title="[bold cyan]LFX Talent Angels[/bold cyan]",
            border_style="cyan",
            padding=(1, 2),
        )
    )


def render_assistant(console: Console, reply: ChatReply) -> None:
    if reply.text:
        console.print(
            Panel(
                _assistant_body(reply.text),
                border_style="bright_black",
                padding=(0, 1),
            )
        )
    if reply.source_note:
        console.print(Text(f"source: {reply.source_note}", style="italic dim green"))


def render_status(
    console: Console,
    *,
    suite: str,
    bound_label: str | None,
    session_name: str | None,
    model_label: str | None = None,
) -> None:
    bits = [
        Text("suite ", style="dim"),
        Text(suite, style="cyan"),
    ]
    # Which model answered is part of reading the answer: the same question
    # phrased by two models reads differently, and one of them may be the
    # zero-token stub. Switching is now a keystroke, so the status line has to
    # say where that keystroke left the session.
    if model_label:
        bits.extend(
            [
                Text("  ·  model ", style="dim"),
                Text(model_label, style="magenta"),
            ]
        )
    if bound_label:
        bits.extend(
            [
                Text("  ·  bound ", style="dim"),
                Text(bound_label, style="bold green"),
            ]
        )
    if session_name:
        bits.extend(
            [
                Text("  ·  session ", style="dim"),
                Text(session_name, style="dim"),
            ]
        )
    line = Text.assemble(*bits)
    console.print(Rule(style="bright_black"))
    console.print(line)

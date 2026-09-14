"""Numbered Locate picker: stable choices + bind by index (no shuffle)."""

from __future__ import annotations

from talent_angels.contracts import AgentResult, NodeRef
from talent_angels.session.models import PendingChoice

PICKER_LIMIT = 10


def choices_from_result(result: AgentResult, *, limit: int = PICKER_LIMIT) -> list[PendingChoice]:
    """Number 1..N in result.nodes order; cap at ``limit``; never shuffle."""
    groups = {
        edge.source_node_id: str(edge.properties.get("group_label") or "")
        for edge in result.edges
        if edge.type == "CLASSIFIED_UNDER"
    }
    nodes = result.nodes[:limit]
    return [
        PendingChoice(number=i, node=node, group_label=groups.get(node.id, ""))
        for i, node in enumerate(nodes, start=1)
    ]


def bind_pick(pending: list[PendingChoice], number: int) -> NodeRef:
    """Return the NodeRef for a 1-based pick number."""
    for choice in pending:
        if choice.number == number:
            return choice.node
    raise ValueError(f"no pending choice numbered {number}")


def render_picker(
    question: str,
    pending: list[PendingChoice],
    *,
    omitted: int,
    intro: str | None = None,
    include_source: bool = True,
) -> str:
    """Markdown numbered list of pref_label; quiet source line; no auto-pick."""
    heading = intro or f'I found several matches for "{question}". Which one did you mean?'
    if not pending:
        suite = "unknown"
        lines = [
            heading if intro else f'I found several matches for "{question}", but none to show.'
        ]
    else:
        suite = pending[0].node.suite
        lines = [
            heading,
            "",
        ]
        current_group = None
        for choice in pending:
            if choice.group_label and choice.group_label != current_group:
                current_group = choice.group_label
                lines.append(f"**{choice.group_label}**")
            lines.append(f"{choice.number}. {choice.node.pref_label}")
        lines.append("")
        lines.append("I won't pick #1 for you — reply with a number.")

    if omitted > 0:
        lines.append(
            f"Showing {len(pending)} of {len(pending) + omitted} "
            f"({omitted} more). The full list is in query details."
        )

    if include_source:
        lines.append(f"source: {suite}")
    return "\n".join(lines)

"""Write full query materials next to the repo, never to git."""

from __future__ import annotations

import os
from pathlib import Path

from talent_angels.assistant.turn import TurnOutcome

DEFAULT_DETAILS_DIR = "data/local/query-details"


def details_dir() -> Path:
    raw = os.environ.get("QUERY_DETAILS_DIR", DEFAULT_DETAILS_DIR).strip() or DEFAULT_DETAILS_DIR
    path = Path(raw)
    if not path.is_absolute():
        path = Path.cwd() / path
    return path


def write_query_details(outcome: TurnOutcome, *, question: str) -> Path:
    """Save a readable markdown dump of the full typed result."""
    target_dir = details_dir()
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / f"{outcome.record.run_id}.md"
    result = outcome.result
    lines = [
        f"# {outcome.record.run_id}",
        "",
        f"- **question:** {question}",
        f"- **capability:** {outcome.capability}",
        f"- **plan:** {', '.join(outcome.record.plan)}",
        f"- **confidence:** {result.confidence}",
        f"- **warnings:** {', '.join(result.warnings) or 'none'}",
        "",
        "## Answer",
        "",
        outcome.answer,
        "",
        f"## Nodes ({len(result.nodes)})",
        "",
    ]
    for index, node in enumerate(result.nodes, start=1):
        alts = ", ".join(node.alt_labels[:8])
        extra = f" — alt: {alts}" if alts else ""
        lines.append(f"{index}. **{node.pref_label}** (`{node.id}`, {node.kind}){extra}")
    if result.edges:
        lines.extend(["", f"## Edges ({len(result.edges)})", ""])
        for edge in result.edges:
            lines.append(f"- {edge.type}: `{edge.source_node_id}` → `{edge.target_node_id}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path

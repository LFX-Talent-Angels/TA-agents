"""Write full query materials next to the repo, never to git."""

from __future__ import annotations

import json
import os
from pathlib import Path

from talent_angels.assistant.turn import TurnOutcome
from talent_angels.runlog.view import economics, economics_markdown_lines

DEFAULT_DETAILS_DIR = "data/local/query-details"


def details_dir() -> Path:
    raw = os.environ.get("QUERY_DETAILS_DIR", DEFAULT_DETAILS_DIR).strip() or DEFAULT_DETAILS_DIR
    path = Path(raw)
    if not path.is_absolute():
        path = Path.cwd() / path
    return path


def write_query_details(outcome: TurnOutcome, *, question: str) -> Path:
    """Save a readable markdown dump plus a JSON sidecar of the full turn."""
    target_dir = details_dir()
    target_dir.mkdir(parents=True, exist_ok=True)
    record = outcome.record
    result = outcome.result
    path = target_dir / f"{record.run_id}.md"
    lines = [
        f"# {record.run_id}",
        "",
        f"- **question:** {question}",
        f"- **capability:** {outcome.capability}",
        f"- **plan:** {', '.join(record.plan)}",
        f"- **confidence:** {result.confidence}",
        f"- **warnings:** {', '.join(result.warnings) or 'none'}",
        f"- **ts:** {record.ts}",
        "",
        *economics_markdown_lines(record),
        "",
        "## Answer",
        "",
        outcome.answer,
        "",
        f"## Nodes ({len(result.nodes)})",
        "",
    ]
    if not result.nodes:
        lines.append("None.")
    for index, node in enumerate(result.nodes, start=1):
        alts = ", ".join(node.alt_labels[:8])
        extra = f" — alt: {alts}" if alts else ""
        lines.append(f"{index}. **{node.pref_label}** (`{node.id}`, {node.kind}){extra}")
    lines.extend(["", f"## Edges ({len(result.edges)})", ""])
    if not result.edges:
        lines.append("None.")
    else:
        for edge in result.edges:
            lines.append(f"- {edge.type}: `{edge.source_node_id}` → `{edge.target_node_id}`")
    if result.evidence:
        lines.extend(["", f"## Evidence ({len(result.evidence)})", ""])
        for pointer in result.evidence:
            lines.append(f"- `{pointer.pointer}` ({pointer.suite})")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    sidecar = target_dir / f"{record.run_id}.json"
    sidecar.write_text(
        json.dumps(
            {
                "question": question,
                "capability": outcome.capability,
                "plan": record.plan,
                "answer": outcome.answer,
                "warnings": result.warnings,
                "confidence": result.confidence,
                "nodes": [node.model_dump() for node in result.nodes],
                "edges": [edge.model_dump() for edge in result.edges],
                "evidence": [item.model_dump() for item in result.evidence],
                "economics": economics(record),
                "record": record.model_dump(),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return path

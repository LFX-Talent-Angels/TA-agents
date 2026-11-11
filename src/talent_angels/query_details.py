"""Write full query materials next to the repo, never to git."""

from __future__ import annotations

import json
import os
from pathlib import Path

from talent_angels.assistant.merge import suite_heading
from talent_angels.assistant.turn import TurnOutcome
from talent_angels.contracts import AgentResult
from talent_angels.memory.paths import default_details_dir
from talent_angels.runlog.view import economics, economics_markdown_lines


def details_dir() -> Path:
    """``QUERY_DETAILS_DIR`` or ``<memory home>/query-details``."""
    raw = os.environ.get("QUERY_DETAILS_DIR", "").strip()
    if not raw:
        return default_details_dir()
    path = Path(raw).expanduser()
    return path if path.is_absolute() else Path.cwd() / path


def _suite_section(result: AgentResult) -> list[str]:
    heading = suite_heading(result.suite) if result.suite else "map"
    lines = [
        f"## {heading} — {result.capability}",
        "",
        f"- **confidence:** {result.confidence}",
        f"- **warnings:** {', '.join(result.warnings) or 'none'}",
        "",
        f"### Nodes ({len(result.nodes)})",
        "",
    ]
    if not result.nodes:
        lines.append("None.")
    edge_for: dict[str, dict[str, object]] = {}
    for edge in result.edges:
        edge_for.setdefault(edge.target_node_id, {"type": edge.type, **edge.properties})
    for index, node in enumerate(result.nodes, start=1):
        alts = ", ".join(node.alt_labels[:8])
        extra = f" — alt: {alts}" if alts else ""
        props = edge_for.get(node.id)
        facts = ""
        if props:
            shown = {k: props[k] for k in _EDGE_FACTS if props.get(k) not in (None, "")}
            facts = " — " + ", ".join(f"{k}={v}" for k, v in shown.items()) if shown else ""
        lines.append(f"{index}. **{node.pref_label}** (`{node.id}`, {node.kind}){facts}{extra}")
    if result.evidence:
        lines.extend(["", f"### Evidence ({len(result.evidence)})", ""])
        lines.extend(f"- `{pointer.pointer}`" for pointer in result.evidence)
    lines.append("")
    return lines


#: Edge properties worth showing next to a neighbour (O*NET weights, ESCO type).
_EDGE_FACTS = ("type", "relation_type", "importance", "level", "hot_technology", "in_demand")


def write_query_details(outcome: TurnOutcome, *, question: str) -> Path:
    """Save a readable markdown dump plus a JSON sidecar of the full turn.

    Every suite's full result is written — the TUI's "the full list is in query
    details" used to be true only for the first suite.
    """
    target_dir = details_dir()
    target_dir.mkdir(parents=True, exist_ok=True)
    record = outcome.record
    path = target_dir / f"{record.run_id}.md"
    lines = [
        f"# {record.run_id}",
        "",
        f"- **question:** {question}",
        f"- **capability:** {outcome.capability}",
        f"- **plan:** {', '.join(record.plan)}",
        f"- **suites:** {', '.join(r.suite for r in outcome.results)}",
        f"- **ts:** {record.ts}",
        "",
        *economics_markdown_lines(record),
        "",
        "## Answer",
        "",
        outcome.answer,
        "",
    ]
    for result in outcome.results:
        lines.extend(_suite_section(result))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    primary = outcome.result
    sidecar = target_dir / f"{record.run_id}.json"
    sidecar.write_text(
        json.dumps(
            {
                "question": question,
                "capability": outcome.capability,
                "plan": record.plan,
                "answer": outcome.answer,
                "warnings": primary.warnings,
                "confidence": primary.confidence,
                "nodes": [node.model_dump() for node in primary.nodes],
                "edges": [edge.model_dump() for edge in primary.edges],
                "evidence": [item.model_dump() for item in primary.evidence],
                "results": [result.model_dump() for result in outcome.results],
                "economics": economics(record),
                "record": record.model_dump(),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return path

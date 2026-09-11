"""One user-facing answer from every attached suite's AgentResult.

Loops results; does not hardcode suite names. A new registry entry appears
in Sources used on the next turn.
"""

from __future__ import annotations

from collections.abc import Sequence

from talent_angels.assistant.merge import suite_heading
from talent_angels.contracts import AgentResult
from talent_angels.llm import LLMClient, Message
from talent_angels.session.phrase import uses_chat_phrasing

_SYNTH_SYSTEM = """You phrase taxonomy map facts for a terminal user.
Rules:
- Use only titles, ids, skills, and descriptions in the FACT CARD.
- Do not say two records are the same id or the same node.
- If two maps name similar titles, say they are separate official records.
- Do not invent occupations or skills.
- One short next-step at the end (skills, or pick a number if a map is ambiguous).
- Do not write LFX or Talent Angels.
- 3-6 sentences. Then stop; Sources used is printed in code."""


def sources_line(
    results: Sequence[AgentResult],
    *,
    extra_warnings: Sequence[str] = (),
) -> str:
    parts: list[str] = []
    for result in results:
        label = suite_heading(result.suite) if result.suite else "map"
        miss = (not result.nodes) or "not_found" in result.warnings
        if miss and "ambiguous" not in result.warnings:
            parts.append(f"{label}: no match")
        else:
            parts.append(label)
    line = "Sources used: " + " · ".join(parts) if parts else "Sources used: none"
    if extra_warnings:
        line += " [" + ", ".join(extra_warnings) + "]"
    return line


def _hit_phrase(result: AgentResult) -> str | None:
    heading = suite_heading(result.suite) if result.suite else "map"
    if "ambiguous" in result.warnings and result.nodes:
        return f"{heading} has several matches"
    if not result.nodes or "not_found" in result.warnings:
        return None
    top = result.nodes[0]
    extra = ""
    if result.capability == "connect" and len(result.nodes) > 1:
        extra = f", {len(result.nodes) - 1} listed skills"
    if result.capability == "pathfind":
        extra = ", no route found" if "no_path" in result.warnings else ", a route on that map"
    return f"{top.pref_label} ({heading}{extra})"


def synthesize_structured(
    results: Sequence[AgentResult],
    *,
    extra_warnings: Sequence[str] = (),
) -> str:
    """Deterministic one-paragraph answer. Safe with no LLM."""
    if not results:
        empty = sources_line((), extra_warnings=extra_warnings)
        return "No attached taxonomy was reachable. " + empty

    hits = [phrase for result in results if (phrase := _hit_phrase(result))]
    if not hits:
        body = "No node for that phrase with today's search. That's a miss, not a maybe."
    elif len(hits) == 1:
        body = f"On the attached maps this lines up with {hits[0]}."
    else:
        body = (
            "On the attached maps this lines up with "
            + "; ".join(hits)
            + ". These are separate official records, not one shared id."
        )
    return f"{body}\n\n{sources_line(results, extra_warnings=extra_warnings)}"


def _fact_card(results: Sequence[AgentResult]) -> str:
    lines: list[str] = []
    for result in results:
        heading = suite_heading(result.suite) if result.suite else "map"
        lines.append(f"SUITE {heading} ({result.suite})")
        lines.append(f"capability: {result.capability}")
        if result.warnings:
            lines.append("warnings: " + ", ".join(result.warnings))
        for node in result.nodes[:6]:
            lines.append(f"- {node.kind}: {node.pref_label} id={node.id}")
            if node.description:
                lines.append(f"  description: {node.description[:280]}")
        if len(result.nodes) > 6:
            lines.append(f"  ({len(result.nodes) - 6} more in query details)")
        lines.append("")
    return "\n".join(lines).strip()


def synthesize(
    results: Sequence[AgentResult],
    *,
    question: str = "",
    llm_client: LLMClient | None = None,
    extra_warnings: Sequence[str] = (),
) -> str:
    fallback = synthesize_structured(results, extra_warnings=extra_warnings)
    if not uses_chat_phrasing(llm_client) or not results:
        return fallback
    assert llm_client is not None
    messages = [
        Message(role="system", content=_SYNTH_SYSTEM),
        Message(
            role="user",
            content=f"User: {question}\n\nFACT CARD:\n{_fact_card(results)}",
        ),
    ]
    try:
        llm_result = llm_client.complete(messages)
    except (RuntimeError, OSError, ValueError):
        return fallback
    text = (llm_result.text or "").strip()
    if not text:
        return fallback
    sources = sources_line(results, extra_warnings=extra_warnings)
    if "Sources used:" not in text:
        text = f"{text}\n\n{sources}"
    return text

"""One user-facing answer from every attached suite's AgentResult.

Loops results; does not hardcode suite names. A new registry entry appears
in Sources used on the next turn.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from talent_angels.assistant.answer import NO_SUBJECT, PATHFIND_UNAVAILABLE, is_pathfind_unavailable
from talent_angels.assistant.llm_call import measure_complete
from talent_angels.assistant.merge import suite_heading
from talent_angels.assistant.prose import prose_only
from talent_angels.contracts import AgentResult
from talent_angels.env import episode_retriever
from talent_angels.llm import LLMClient, Message
from talent_angels.llm.protocol import uses_chat_phrasing
from talent_angels.memory.agent_notes import notes_prefix
from talent_angels.memory.profile import profile_prefix
from talent_angels.memory.retrieval import recall_prefix

_NODE_ID_RE = re.compile(
    r"\b(?:esco|onet|sfia|bls):[a-z0-9][a-z0-9_.:-]*",
    re.IGNORECASE,
)
_DUMMY_SUBJECT_RE = re.compile(
    r"\b(the subject|or the subject)\b",
    re.IGNORECASE,
)


def _phrasing_is_unsafe(text: str) -> bool:
    """Raw ids and dummy subjects belong in JSON, not the user sentence."""
    return bool(_NODE_ID_RE.search(text) or _DUMMY_SUBJECT_RE.search(text))


_SYNTH_SYSTEM = """You phrase taxonomy map facts for a terminal user.
Rules:
- First reply to the user's own words in one sentence; never assume a goal or
  a wish the user did not state.
- Use only titles, skills, and descriptions in the FACT CARD.
- Do not invent demand, pay, outlook, or study advice.
- Do not say two records are the same id or the same node.
- If two maps name similar titles, say once, in a short clause, that they are
  separate official records; do not repeat it.
- Do not invent occupations, skills, or people.
- Do not paste node ids (esco:…, onet:…). Titles only.
- Do not suggest pathfinding, filtering by category, or any interactive capability.
- Do not write LFX or Talent Angels.
- 2-4 plain sentences: no tables, lists, headings, or code. Then stop; Sources
  used is printed in code."""


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


#: How many names a deterministic answer shows per list before "+N more".
ANSWER_LIST_PREVIEW = 6


def _preview(labels: Sequence[str], total: int | None = None) -> str:
    shown = [label for label in labels if label][:ANSWER_LIST_PREVIEW]
    count = len(labels) if total is None else total
    extra = count - len(shown)
    text = ", ".join(shown)
    return f"{text} (+{extra} more)" if extra > 0 else text


def _connect_line(result: AgentResult) -> str:
    """Name the centre node and its top neighbours, split by relation type.

    Neighbours arrive ranked (skills first, O*NET importance, hot technology);
    this only groups and previews them.
    """
    center, neighbours = result.nodes[0], result.nodes[1:]
    type_of: dict[str, str] = {}
    for edge in result.edges:
        other = edge.target_node_id if edge.source_node_id == center.id else edge.source_node_id
        type_of.setdefault(other, edge.type)
    tools = [n.pref_label for n in neighbours if type_of.get(n.id) == "USES_SOFTWARE"]
    skills = [n.pref_label for n in neighbours if type_of.get(n.id) != "USES_SOFTWARE"]
    parts: list[str] = []
    if skills:
        parts.append(f"{len(skills)} skill(s): {_preview(skills)}")
    if tools:
        parts.append(f"{len(tools)} software/tool(s): {_preview(tools)}")
    if not parts:
        return f"{center.pref_label} — no linked skills on this map"
    return f"{center.pref_label} — " + "; ".join(parts)


def _suite_line(result: AgentResult, *, list_ambiguous: bool = True) -> str | None:
    """One line per suite, or None when the suite had nothing to say."""
    heading = suite_heading(result.suite) if result.suite else "map"
    if "ambiguous" in result.warnings and result.nodes:
        if not list_ambiguous:
            return f"{heading}: several matches"
        names = [node.pref_label for node in result.nodes]
        return f"{heading}: several matches — {_preview(names)}"
    if not result.nodes or "not_found" in result.warnings:
        return None
    if result.capability == "connect":
        return f"{heading}: {_connect_line(result)}"
    top = result.nodes[0]
    confidence = f", confidence {result.confidence:.0%}" if result.confidence is not None else ""
    return f"{heading}: {top.pref_label} ({top.kind}{confidence})"


def _hit_phrase(result: AgentResult) -> str | None:
    """Short label for a suite hit (kept for callers that want one phrase)."""
    heading = suite_heading(result.suite) if result.suite else "map"
    if "ambiguous" in result.warnings and result.nodes:
        return f"{heading} has several matches"
    if not result.nodes or "not_found" in result.warnings:
        return None
    top = result.nodes[0]
    extra = ""
    if result.capability == "connect" and len(result.nodes) > 1:
        extra = f", {len(result.nodes) - 1} listed skills"
    return f"{top.pref_label} ({heading}{extra})"


def synthesize_structured(
    results: Sequence[AgentResult],
    *,
    extra_warnings: Sequence[str] = (),
    list_ambiguous: bool = True,
) -> str:
    """Deterministic answer naming what each suite found. Safe with no LLM.

    ``list_ambiguous=False`` when a numbered picker is printed underneath (TUI),
    so the candidates are not listed twice.
    """
    if not results:
        empty = sources_line((), extra_warnings=extra_warnings)
        return "No attached taxonomy was reachable. " + empty

    sources = sources_line(results, extra_warnings=extra_warnings)
    if all("bind_required" in result.warnings for result in results):
        return f"Name or pick an occupation first, then ask for skills.\n\n{sources}"
    if all("no_subject" in result.warnings for result in results):
        return f"{NO_SUBJECT}\n\n{sources}"
    if all(is_pathfind_unavailable(result) for result in results):
        return f"{PATHFIND_UNAVAILABLE}\n\n{sources}"

    lines = [
        line for result in results if (line := _suite_line(result, list_ambiguous=list_ambiguous))
    ]
    if not lines:
        body = "No node for that phrase with today's search. That's a miss, not a maybe."
    elif len(lines) == 1:
        body = lines[0]
    else:
        body = "\n".join(lines) + "\nThese are separate official records, not one shared id."
    if list_ambiguous and any("ambiguous" in r.warnings and r.nodes for r in results):
        body += "\nWhich one do you mean?"
    return f"{body}\n\n{sources}"


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
    # The fourth site that renders `recall_prefix`. It belongs here for the same
    # reason it is in the agent loop and the two phrasing calls in
    # `session.phrase`: this is the prompt where the user's own words are
    # restated back to them, so it is the site where "you asked about this
    # before" is worth the most and costs the least. The user message already
    # carries `question`, so this is the only place the turn's own text is
    # available for retrieval — and it is the same string the loop recalls on,
    # not `result` or the fact card, so the two prompts cannot disagree about
    # what the user asked.
    #
    # `""` when no retriever is configured, so the prompt is byte-identical to
    # before for anyone not opted in.
    system_prompt = (
        profile_prefix()
        + notes_prefix()
        + recall_prefix(question, retriever=episode_retriever())
        + _SYNTH_SYSTEM
    )
    messages = [
        Message(role="system", content=system_prompt),
        Message(
            role="user",
            content=f"User: {question}\n\nFACT CARD:\n{_fact_card(results)}",
        ),
    ]
    try:
        llm_result, _ = measure_complete(llm_client, messages, stage="synthesize")
    except RuntimeError:
        return fallback
    text = prose_only(llm_result.text or "")
    if not text or _phrasing_is_unsafe(text):
        return fallback
    sources = sources_line(results, extra_warnings=extra_warnings)
    if "Sources used:" not in text:
        text = f"{text}\n\n{sources}"
    return text

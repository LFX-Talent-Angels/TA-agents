"""Package an AgentResult into user-facing text (ARCHITECTURE.md: answer step).

`structured` (default, `ANSWER_MODE` env) costs zero tokens. `natural` is the
only mode that calls the LLM, and only to *phrase* an already-resolved
result — never to decide facts (rule #2: only graph data is cited as fact).
"""

from __future__ import annotations

from talent_angels.assistant.llm_call import measure_complete
from talent_angels.contracts import AgentResult
from talent_angels.llm import LLMClient, Message
from talent_angels.runlog import StageUsage

AMBIGUOUS_CHOICE_LIMIT = 3
PATHFIND_UNIMPLEMENTED_WARNING = "capability_not_implemented:pathfind"
PATHFIND_UNAVAILABLE = (
    "Pathfind is not in this MVP, so I cannot compute a skill path or gap "
    "between two occupations. Ask me to locate one occupation, or to list "
    "the skills of one occupation."
)


def is_unimplemented_pathfind(result: AgentResult) -> bool:
    return PATHFIND_UNIMPLEMENTED_WARNING in result.warnings


def is_terminal_locate(result: AgentResult) -> bool:
    """Search finished: ask the user or stop. Do not search again."""
    return "ambiguous" in result.warnings or "not_found" in result.warnings


def summarize_result(result: AgentResult) -> str:
    """Deterministic user-facing text from a typed result (no LLM)."""
    if is_unimplemented_pathfind(result):
        return PATHFIND_UNAVAILABLE
    if not result.nodes:
        warning = result.warnings[0] if result.warnings else "not_found"
        return f"No match found for capability '{result.capability}' ({warning})."

    confidence_pct = f"{result.confidence:.0%}" if result.confidence is not None else "unknown"
    if "ambiguous" in result.warnings:
        choices = result.nodes[:AMBIGUOUS_CHOICE_LIMIT]
        rendered = "; ".join(f"{node.pref_label} ({node.kind}, id={node.id})" for node in choices)
        remaining = len(result.nodes) - len(choices)
        remainder = f"; {remaining} more candidate(s)" if remaining else ""
        return (
            f"Ambiguous locate result — confidence {confidence_pct}. "
            f"Candidates: {rendered}{remainder}. Please clarify which candidate you mean."
        )
    if result.capability == "pathfind":
        occupations = [node for node in result.nodes if node.kind.casefold() == "occupation"]
        start = (occupations[0] if occupations else result.nodes[0]).pref_label
        end = (occupations[-1] if occupations else result.nodes[-1]).pref_label
        summary = f"{start} → {end}"
        gap = next((item for item in result.warnings if item.startswith("derived_skill_gap:")), "")
        if gap:
            summary += "; " + gap.replace("derived_skill_gap:", "skill gap: ", 1)
        other = [item for item in result.warnings if not item.startswith("derived_skill_gap:")]
        if other:
            summary += f" [warnings: {', '.join(other)}]"
        return summary
    if result.capability == "connect":
        center = result.nodes[0]
        neighbors = result.nodes[1:]
        shown = neighbors[:5]
        rendered = "; ".join(node.pref_label for node in shown)
        extra = len(neighbors) - len(shown)
        remainder = f"; {extra} more" if extra else ""
        summary = (
            f"{center.pref_label} — confidence {confidence_pct}; "
            f"{len(result.edges)} direct connection(s): {rendered}{remainder}"
        )
        if extra:
            summary += " The full list is in the result payload."
        if result.warnings:
            summary += f" [warnings: {', '.join(result.warnings)}]"
        return summary

    top = result.nodes[0]
    summary = f"{top.pref_label} ({top.kind}, id={top.id}) — confidence {confidence_pct}"
    if len(result.nodes) > 1:
        summary += f"; {len(result.nodes) - 1} other candidate(s)"
    if result.warnings:
        summary += f" [warnings: {', '.join(result.warnings)}]"
    return summary


def build_answer(
    result: AgentResult, *, llm_client: LLMClient, mode: str = "structured"
) -> tuple[str, StageUsage | None]:
    """Returns (answer text, answer-stage usage). Stage is None when no LLM call ran."""
    summary = summarize_result(result)
    if is_unimplemented_pathfind(result) or is_terminal_locate(result) or not result.nodes:
        return summary, None
    # Many neighbors: keep the counted summary. Do not let the model invent pagination.
    if result.capability == "connect" and len(result.edges) > 5:
        return summary, None

    if mode != "natural":
        return summary, None

    messages = [
        Message(
            role="system",
            content=(
                "Rephrase the following taxonomy result as one plain sentence, "
                "citing only the given facts. Do not add occupations or skills "
                "that are not listed. If the result is ambiguous, ask the user "
                "to choose."
            ),
        ),
        Message(role="user", content=summary),
    ]
    try:
        llm_result, stage = measure_complete(llm_client, messages, stage="answer")
    except RuntimeError:
        return summary, None
    return llm_result.text, stage

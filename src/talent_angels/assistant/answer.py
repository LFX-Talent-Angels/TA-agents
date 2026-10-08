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

# GAP F: 5 undercounted real connect results (O*NET occupations routinely have
# 20-260+ skill edges). One constant so the summary (here) and the LLM-facing
# fact card (session.phrase.connect_card) preview the same number of skills.
CONNECT_PREVIEW_CAP = 15


def is_terminal_locate(result: AgentResult) -> bool:
    """Search finished: ask the user or stop. Do not search again."""
    return "ambiguous" in result.warnings or "not_found" in result.warnings


PATHFIND_UNAVAILABLE = (
    "Routes between two occupations (Pathfind) are not available yet, so no path "
    "was computed. Ask to compare the two occupations to see the skills they share "
    "and the ones only one of them needs."
)
NO_SUBJECT = (
    "Which occupation or skill do you mean? For example: "
    "“What skills does a nurse need?” or “Where is data scientist?”"
)


def is_pathfind_unavailable(result: AgentResult) -> bool:
    return any(w.startswith("capability_not_implemented") for w in result.warnings)


def summarize_result(result: AgentResult) -> str:
    """Deterministic user-facing text from a typed result (no LLM)."""
    if "bind_required" in result.warnings:
        return "Name or pick an occupation first, then ask for skills."
    if "no_subject" in result.warnings:
        return NO_SUBJECT
    if is_pathfind_unavailable(result):
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
    if result.capability == "connect":
        center = result.nodes[0]
        neighbors = result.nodes[1:]
        shown = neighbors[:CONNECT_PREVIEW_CAP]
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
    """Returns (answer text, answer-stage usage). Stage is None when no LLM call ran.

    **No recall block here, deliberately, and the signature is part of why.**
    Every other system-prompt site takes the question and appends the block built
    by ``memory.retrieval`` beside the profile and the notes. This one has no
    ``question`` parameter, so it cannot — and that is the design, not an
    omission to fix by passing one in. This is the single prompt that hands the
    user's own words back to the user: its whole job is to rephrase an
    already-resolved ``AgentResult``, and the recall block is a list of *past*
    questions. Feeding it here would put the user's earlier text in front of a
    model whose only instruction is "rephrase the following taxonomy result",
    where the cheapest way to be useful is to start answering the past questions
    instead. Intent classification omits recall for a different and equally
    deliberate reason (``llm_plan.interpret_question``): recall there would bias
    the plan towards the topics the user happened to ask about before.

    So the two omissions are not one omission, and neither is an oversight. The
    cost is that "we covered this last Thursday" has to be earned elsewhere.
    """
    summary = summarize_result(result)
    if is_terminal_locate(result) or not result.nodes:
        return summary, None
    # Many neighbors: keep the counted summary. Do not let the model invent
    # pagination. Deliberately a tighter gate than CONNECT_PREVIEW_CAP above —
    # this decides whether a natural-mode rephrase is safe at all, not how many
    # skills the (always-shown) summary previews.
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

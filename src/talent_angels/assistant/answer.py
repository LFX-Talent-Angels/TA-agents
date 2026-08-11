"""Package an AgentResult into user-facing text (ARCHITECTURE.md: answer step).

`structured` (default, `ANSWER_MODE` env) costs zero tokens. `natural` is the
only mode that calls the LLM, and only to *phrase* an already-resolved
result — never to decide facts (rule #2: only graph data is cited as fact).
"""

from __future__ import annotations

from talent_angels.contracts import AgentResult
from talent_angels.llm import LLMClient, LLMUsage, Message


def build_answer(
    result: AgentResult, *, llm_client: LLMClient, mode: str = "structured"
) -> tuple[str, LLMUsage | None]:
    """Returns (answer text, LLM usage). Usage is None when no LLM call was made."""
    if not result.nodes:
        warning = result.warnings[0] if result.warnings else "not_found"
        return f"No match found for capability '{result.capability}' ({warning}).", None

    top = result.nodes[0]
    confidence_pct = f"{result.confidence:.0%}" if result.confidence is not None else "unknown"
    summary = f"{top.pref_label} ({top.kind}, id={top.id}) — confidence {confidence_pct}"
    if len(result.nodes) > 1:
        summary += f"; {len(result.nodes) - 1} other candidate(s)"
    if result.warnings:
        summary += f" [warnings: {', '.join(result.warnings)}]"

    if mode != "natural":
        return summary, None

    messages = [
        Message(
            role="system",
            content=(
                "Rephrase the following Locate result as one plain sentence, "
                "citing only the given facts."
            ),
        ),
        Message(role="user", content=summary),
    ]
    llm_result = llm_client.complete(messages)
    return llm_result.text, llm_result.usage

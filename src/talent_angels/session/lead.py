"""The short answer above a pick list: what was understood, what was found, what next.

Written in code from the plan and the results, never by the model: a pick list
holds only candidate titles, and a model asked to introduce one has invented
skills for a title it picked itself. Every clause here comes from the planner's
reading (labelled as a reading) or from the graph results.
"""

from __future__ import annotations

from collections.abc import Sequence

from talent_angels.assistant.llm_plan import PlanDraft
from talent_angels.assistant.merge import suite_heading
from talent_angels.contracts import AgentResult

_NEXT = {
    "goal": "Pick the one you mean and I'll save it as your goal.",
    "standing": "Pick the one you mean and I'll save it as your current job.",
    "reject": "Pick the one you mean and I'll note that it is not your job.",
}


def _join(titles: Sequence[str]) -> str:
    if len(titles) == 1:
        return titles[0]
    return f"{', '.join(titles[:-1])} or {titles[-1]}"


def understood(question: str, draft: PlanDraft | None) -> str:
    """ "I read "SWE" as software engineer, software developer or web developer." """
    said = question.strip().rstrip("?.!")
    subject = (draft.subject or "").strip() if draft is not None else ""
    subject = subject or said
    if not subject:
        return ""
    quoted = f'"{subject}"' if subject.casefold() == said.casefold() else "your request"
    if draft is not None and draft.candidates:
        return f"I read {quoted} as {_join(list(draft.candidates[:3]))}."
    if subject.casefold() in said.casefold():
        return f'I looked up "{subject}".'
    return f'I read {quoted} as "{subject}".'


def _found_one(result: AgentResult) -> str:
    heading = suite_heading(result.suite)
    if result.nodes and "ambiguous" in result.warnings:
        more = "+" if "truncated" in result.warnings else ""
        noun = "title" if len(result.nodes) == 1 and not more else "titles"
        return f"{heading} has {len(result.nodes)}{more} matching {noun}"
    if result.nodes and result.capability in ("locate", "connect"):
        return f"{heading} matched {result.nodes[0].pref_label}"
    if "not_found" in result.warnings:
        return f"{heading} has no match"
    return ""


def found(results: Sequence[AgentResult]) -> str:
    """ "ESCO matched software developer and O*NET has 25+ matching titles." """
    parts = [part for part in (_found_one(result) for result in results) if part]
    if not parts:
        return ""
    if len(parts) == 1:
        return f"{parts[0]}."
    return f"{', '.join(parts[:-1])} and {parts[-1]}."


def lead(question: str, draft: PlanDraft | None, results: Sequence[AgentResult]) -> str:
    """Two or three short sentences shown before any list."""
    intent = draft.profile_intent if draft is not None else None
    next_step = _NEXT.get(
        intent or "", "Pick the one you mean, or tell me more and I'll narrow it."
    )
    return " ".join(
        part for part in (understood(question, draft), found(results), next_step) if part
    )

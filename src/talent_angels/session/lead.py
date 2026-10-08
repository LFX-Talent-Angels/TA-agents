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
from talent_angels.session.i18n import t

_NEXT = {"goal": "next_goal", "standing": "next_standing", "reject": "next_reject"}


def _join(titles: Sequence[str], word: str = "or") -> str:
    if len(titles) == 1:
        return titles[0]
    return f"{', '.join(titles[:-1])} {t(word)} {titles[-1]}"


def _on_the_map(candidate: str, results: Sequence[AgentResult]) -> bool:
    wanted = candidate.casefold().strip()
    for result in results:
        for node in result.nodes:
            for name in (node.pref_label, *node.alt_labels):
                name = name.casefold().strip()
                if name in (wanted, f"{wanted}s") or f"{name}s" == wanted:
                    return True
    return False


def understood(question: str, draft: PlanDraft | None, results: Sequence[AgentResult] = ()) -> str:
    """ "I read "SWE" as software engineer, software developer or web developer."

    Only planner titles the results confirm are named: a guess the map does
    not have is never shown to the user.
    """
    said = question.strip().rstrip("?.!")
    side = next(
        (
            w.split(":", 1)[1]
            for result in results
            for w in result.warnings
            if w.startswith("compare_side:")
        ),
        None,
    )
    if side and draft is not None and draft.subject and draft.secondary_subject:
        # "compare nurse and doctor": say it is a compare, and which side needs a pick.
        waiting = draft.subject if side == "1" else draft.secondary_subject
        return t(
            "compare_needs_pick",
            first=draft.subject,
            second=draft.secondary_subject,
            subject=waiting,
        )
    subject = (draft.subject or "").strip() if draft is not None else ""
    subject = subject or said
    if not subject:
        return ""
    quoted = f'"{subject}"' if subject.casefold() == said.casefold() else t("your_request")
    confirmed = [c for c in (draft.candidates if draft else ()) if _on_the_map(c, results)]
    if confirmed:
        return t("read_as_titles", quoted=quoted, titles=_join(confirmed[:3]))
    if subject.casefold() in said.casefold():
        return t("looked_up", subject=subject)
    return t("read_as_subject", quoted=quoted, subject=subject)


def _found_one(result: AgentResult) -> str:
    heading = suite_heading(result.suite)
    if result.nodes and "ambiguous" in result.warnings:
        more = "+" if "truncated" in result.warnings else ""
        many = len(result.nodes) > 1 or bool(more)
        count = f"{len(result.nodes)}{more}"
        pointers = [pointer.pointer for pointer in result.evidence]
        if pointers and all(":search:hybrid:" in pointer for pointer in pointers):
            # Found by meaning, not by name: close, not matching.
            noun = t("close_titles" if many else "close_title")
            return t("found_close", suite=heading, count=count, noun=noun)
        noun = t("matching_titles" if many else "matching_title")
        return t("found_list", suite=heading, count=count, noun=noun)
    if result.nodes and result.capability in ("locate", "connect"):
        return t("found_one", suite=heading, title=result.nodes[0].pref_label)
    if "not_found" in result.warnings:
        return t("found_none", suite=heading)
    return ""


def found(results: Sequence[AgentResult]) -> str:
    """ "ESCO matched software developer and O*NET has 25+ matching titles." """
    parts = [part for part in (_found_one(result) for result in results) if part]
    if not parts:
        return ""
    return f"{_join(parts, 'and')}."


def pick_intent(draft: PlanDraft | None) -> str | None:
    """What a pick from this list says about the user, if anything.

    Suggestions for "I'm a teacher and I like math, what suits me?" are not the
    user's current job: picking one must not save it as that.
    """
    if draft is None:
        return None
    if draft.profile_intent == "standing" and draft.candidates:
        return None
    return draft.profile_intent


def lead(question: str, draft: PlanDraft | None, results: Sequence[AgentResult]) -> str:
    """Two or three short sentences shown before any list."""
    next_step = t(_NEXT.get(pick_intent(draft) or "", "next_default"))
    return " ".join(
        part for part in (understood(question, draft, results), found(results), next_step) if part
    )

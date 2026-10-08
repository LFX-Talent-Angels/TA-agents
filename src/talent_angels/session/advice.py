"""Turn an advice question into a map question the profile can ground.

The assistant still does not choose for the user. "Should I be a nurse or a
midwife?" becomes a compare of the two; "what should I learn first?" with a
saved goal becomes a compare of the current job and the goal, whose
"only <goal>" group is what the map lists for the goal and not the job. With
nothing to ground it on, the caller keeps refusing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_EITHER = re.compile(
    r"\b(?:be(?:come)?|choose|pick|go\s+for|study)\s+(?:an?\s+)?(.+?)\s+or\s+(?:an?\s+)?(.+?)"
    r"\s*[?.!]*$",
    re.IGNORECASE,
)
_LEARN_TOWARD_GOAL = re.compile(
    r"\blearn\b.*\bfirst\b|\bwhat\s+should\s+i\s+learn\b|\bmissing\b|\bskill\s+gap\b",
    re.IGNORECASE,
)
_NOT_A_CHOICE = "I can't choose for you, but here is what the map lists for each."
#: "which one should I choose?" right after two titles were compared.
_WHICH_OF_PAIR = re.compile(
    r"\bwhich\s+(?:one|of\s+(?:them|the\s+two|these|those))\b"
    r"|\b(?:choose|pick)\s+(?:between\s+)?(?:them|the\s+two|one\s+of\s+them)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class AdvicePlan:
    question: str
    preface: str


def advice_plan(
    text: str,
    *,
    current: str | None,
    goal: str | None,
    compared: tuple[str, str] | None = None,
) -> AdvicePlan | None:
    """A grounded map question for an advice line, or None to keep refusing.

    ``compared`` is the pair the user just compared, so "which one should I
    choose?" has referents.
    """
    either = _EITHER.search(text)
    if either:
        return AdvicePlan(f"compare {either.group(1)} and {either.group(2)}", _NOT_A_CHOICE)
    if compared and _WHICH_OF_PAIR.search(text):
        first, second = compared
        return AdvicePlan(
            f"compare {first} and {second}",
            f"I can't choose between **{first}** and **{second}** for you. Here is what the "
            "map lists for each; tell me what you enjoy or want to avoid and I'll point to "
            "the skills that match.",
        )
    if not _LEARN_TOWARD_GOAL.search(text) or not goal:
        return None
    if current and current.casefold() != goal.casefold():
        return AdvicePlan(
            f"compare {current} and {goal}",
            f"Your profile says you are a **{current}** and your goal is **{goal}**. "
            f'The "Only {goal}" skills are listed for your goal and not for your current '
            "job on the map. It is a map, not a study plan, and I don't rank them for you.",
        )
    return AdvicePlan(
        f"what are the essential skills of a {goal}?",
        f"Your goal is **{goal}**. Here is what the map lists for it; "
        'tell me your current job ("I am a …") to see what is new for you.',
    )

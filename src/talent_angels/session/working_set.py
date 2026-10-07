"""The titles this conversation has looked at, so "compare the two" has referents.

``bindings`` is the current topic, one node per suite. The working set is the
last few titles the user resolved, as they searched them, newest last. It is
read in code to rewrite a pair follow-up into an explicit compare; the model
never guesses which two titles "they" are.
"""

from __future__ import annotations

import re

RECENT_LIMIT = 5

_PAIR = r"(?:the\s+two|both(?:\s+of\s+them)?|them|they|these\s+two|those\s+two)"
_PAIR_FOLLOWUPS = (
    re.compile(rf"^\s*compare\s+{_PAIR}\b", re.IGNORECASE),
    re.compile(rf"\bdifferences?\s+between\s+{_PAIR}\b", re.IGNORECASE),
    re.compile(rf"\bwhat\s+do\s+{_PAIR}\s+have\s+in\s+common\b", re.IGNORECASE),
    re.compile(rf"\bhow\s+do\s+{_PAIR}\s+(?:differ|compare)\b", re.IGNORECASE),
    re.compile(rf"\bwhat\s+do\s+{_PAIR}\s+share\b", re.IGNORECASE),
    re.compile(rf"\bchoose\s+between\s+{_PAIR}\b", re.IGNORECASE),
    re.compile(r"\bwhich\s+of\s+the\s+two\b", re.IGNORECASE),
)
_COMPARE_IT_WITH = re.compile(
    r"^\s*compare\s+(?:it|this|that)\s+(?:with|to|and)\s+(?:an?\s+)?(.+?)\s*[?.!]*$",
    re.IGNORECASE,
)


def remember(recent: list[str], label: str) -> list[str]:
    """``recent`` with ``label`` moved to the newest place, capped."""
    label = label.strip()
    if not label:
        return recent
    kept = [item for item in recent if item.casefold() != label.casefold()]
    return [*kept, label][-RECENT_LIMIT:]


def pair_followup(text: str, recent: list[str]) -> str | None:
    """An explicit compare question for a follow-up about the last titles, or None."""
    match = _COMPARE_IT_WITH.match(text)
    if match and recent:
        other = match.group(1).strip()
        if other.casefold() != recent[-1].casefold():
            return f"compare {recent[-1]} and {other}"
    if len(recent) >= 2 and any(pattern.search(text) for pattern in _PAIR_FOLLOWUPS):
        return f"compare {recent[-2]} and {recent[-1]}"
    return None

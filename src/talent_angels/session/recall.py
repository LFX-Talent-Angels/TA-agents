"""Answer "what did we talk about?" from session and episode memory, in code.

The titles this conversation resolved come from the working set; earlier
conversations come from recorded episodes. Nothing here is a taxonomy fact,
so the reply says where each list comes from and cites no graph data.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from talent_angels.memory.episodes import Episode

#: Earlier titles shown before the rest are left out.
EARLIER_LIMIT = 5

_FIRST = re.compile(r"\bfirst\b", re.IGNORECASE)
_LAST = re.compile(r"\b(?:last|previous|latest)\b", re.IGNORECASE)


def _earlier_titles(episodes: Sequence[Episode], skip: set[str]) -> list[str]:
    titles: list[str] = []
    for episode in episodes:  # newest first
        if not episode.satisfied:
            continue
        for label in episode.node_labels[:1]:
            key = label.casefold()
            if key not in skip:
                skip.add(key)
                titles.append(label)
    return titles[:EARLIER_LIMIT]


def recall_reply(text: str, recent: Sequence[str], episodes: Sequence[Episode]) -> str:
    """One deterministic reply naming what was looked at, newest context first."""
    if recent and _FIRST.search(text):
        return f"The first title in this conversation was **{recent[0]}**."
    if recent and _LAST.search(text):
        return f"The last title in this conversation was **{recent[-1]}**."
    lines: list[str] = []
    if recent:
        lines.append("In this conversation you looked at: " + ", ".join(recent) + ".")
    earlier = _earlier_titles(episodes, {title.casefold() for title in recent})
    if earlier:
        lines.append("Earlier you looked at: " + ", ".join(earlier) + ".")
    if not lines:
        return "We haven't looked at any occupations or skills yet. Name a job title to start."
    lines.append("Name one to look at it again.")
    return "\n".join(lines)

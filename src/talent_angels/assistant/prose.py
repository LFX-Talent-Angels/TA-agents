"""Keep model phrasing to plain sentences; lists, tables and ids are printed by code."""

from __future__ import annotations

import re

_FENCE = re.compile(r"```.*?(?:```|\Z)", re.DOTALL)
_BULLET = re.compile(r"^\s*[-*•]\s+\S")
#: Lines the model may write but code already prints: table rows, horizontal
#: rules, headings, a lone backticked key/value such as `occupation_id: x`,
#: and a bold record field such as **Confidence:** 95%.
_LAYOUT = re.compile(
    r"^\s*(?:\|.*|[-*_]{3,}\s*|#{1,6}\s.*|`[^`]*`\s*"
    r"|\*\*(?:title|confidence|kind|id|occupation_id)\b[^*]*\*\*.*)$",
    re.IGNORECASE,
)


def prose_only(text: str) -> str:
    """Drop tables, code blocks, headings and long bullet lists the model wrote.

    A short mention ("includes: - a - b") stays; three or more bullets repeat
    the list code prints under the text, so they go.
    """
    text = _FENCE.sub("", text)
    lines = [line for line in text.splitlines() if not _LAYOUT.match(line)]
    if sum(1 for line in lines if _BULLET.match(line)) >= 3:
        lines = [line for line in lines if not _BULLET.match(line)]
    kept = "\n".join(lines)
    return re.sub(r"\n{3,}", "\n\n", kept).strip()

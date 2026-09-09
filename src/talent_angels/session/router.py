"""Line router — classify user input before the model (rails)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

LineKind = Literal["command", "greet", "help_plain", "advice", "catalogue", "pick", "map"]

_GREET_RE = re.compile(
    r"^\s*(hi|hello|hey|thanks|thank you|"
    r"good\s+(morning|afternoon|evening))\s*[!.?]?\s*$",
    re.IGNORECASE,
)

_HELP_PLAIN_RE = re.compile(
    r"^\s*("
    r"what\s+can\s+you\s+do|"
    r"how\s+can\s+you\s+help(\s+me)?|"
    r"how\s+do\s+you\s+help|"
    r"what\s+do\s+you\s+do|"
    r"help"
    r")\s*[?.!]?\s*$",
    re.IGNORECASE,
)

_PICK_NUM_RE = re.compile(r"^\s*(\d+)\s*[.)]?\s*$")
_PICK_FIRST_RE = re.compile(
    r"^\s*(the\s+)?first(\s+one)?\s*[.!]?\s*$",
    re.IGNORECASE,
)

_ADVICE_PHRASES = (
    "should i",
    "is this a good career",
    "what should i do with my life",
)

_CATALOGUE_RE = re.compile(
    r"^\s*(?:please\s+)?(?:list|show|give(?:\s+me)?|what\s+are)\s+"
    r"(?:me\s+)?(?:all|every|the\s+full\s+list\s+of)\s+"
    r"(?:the\s+)?(?:jobs?|occupations?|roles?|titles?|careers?)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class RoutedLine:
    kind: LineKind
    text: str
    pick: int | None = None


def _is_advice(lowered: str) -> bool:
    if any(phrase in lowered for phrase in _ADVICE_PHRASES):
        return True
    if "learn" in lowered and "first" in lowered:
        return True
    return False


def route_line(text: str) -> RoutedLine:
    stripped = text.strip()
    if stripped.startswith("/"):
        return RoutedLine(kind="command", text=stripped)

    if _GREET_RE.match(stripped):
        return RoutedLine(kind="greet", text=stripped)

    if _HELP_PLAIN_RE.match(stripped):
        return RoutedLine(kind="help_plain", text=stripped)

    m = _PICK_NUM_RE.match(stripped)
    if m:
        return RoutedLine(kind="pick", text=stripped, pick=int(m.group(1)))

    if _PICK_FIRST_RE.match(stripped):
        return RoutedLine(kind="pick", text=stripped, pick=1)

    lowered = stripped.lower()
    if _is_advice(lowered):
        return RoutedLine(kind="advice", text=stripped)

    if _CATALOGUE_RE.search(stripped):
        return RoutedLine(kind="catalogue", text=stripped)

    return RoutedLine(kind="map", text=stripped)

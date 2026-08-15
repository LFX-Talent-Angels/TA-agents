"""Heuristic intent routing (MVP plan Sec 5) — zero tokens, keyword-based.

Only `locate` is implemented in Gate A. A question that heuristically routes
to `connect`/`pathfind` still returns a typed result — with an explicit
`capability_not_implemented` warning — rather than silently answering with
Locate instead (ARCHITECTURE.md: never invent; warnings are how "no" is said).
"""

from __future__ import annotations

from typing import Literal

Capability = Literal["locate", "connect", "pathfind"]
CAPABILITY_LOCATE: Capability = "locate"
CAPABILITY_CONNECT: Capability = "connect"
CAPABILITY_PATHFIND: Capability = "pathfind"

_PATHFIND_KEYWORDS = (
    "gap",
    "path between",
    "path from",
    "path to",
    "skill path",
    "career path",
    "learning path",
    "route from",
    "route to",
    "→",
    "->",
)
_CONNECT_KEYWORDS = ("skills for", "skills does", "neighbors", "essential skill", "optional skill")


def extract_locate_subject(question: str) -> str:
    """Strip common locate wrappers so lexical search sees the occupation/skill."""
    text = question.strip()
    lowered = text.lower()
    for prefix in ("where is ", "where are ", "what is ", "what are ", "find ", "locate "):
        if lowered.startswith(prefix):
            text = text[len(prefix) :]
            lowered = text.lower()
            break
    for suffix in (" in esco?", " in esco", "?"):
        if lowered.endswith(suffix):
            text = text[: -len(suffix)].rstrip()
            lowered = text.lower()
    return text.strip() or question.strip()


def classify_capability(question: str) -> Capability:
    q = question.lower()
    padded = f" {q} "
    if any(k in q for k in _PATHFIND_KEYWORDS):
        return CAPABILITY_PATHFIND
    if " from " in padded and " to " in padded:
        return CAPABILITY_PATHFIND
    if any(k in q for k in _CONNECT_KEYWORDS):
        return CAPABILITY_CONNECT
    return CAPABILITY_LOCATE

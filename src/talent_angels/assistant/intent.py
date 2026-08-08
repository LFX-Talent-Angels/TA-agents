"""Heuristic intent routing (MVP plan Sec 5) — zero tokens, keyword-based.

Only `locate` is implemented in Gate A. A question that heuristically routes
to `connect`/`pathfind` still returns a typed result — with an explicit
`capability_not_implemented` warning — rather than silently answering with
Locate instead (ARCHITECTURE.md: never invent; warnings are how "no" is said).
"""

from __future__ import annotations

CAPABILITY_LOCATE = "locate"
CAPABILITY_CONNECT = "connect"
CAPABILITY_PATHFIND = "pathfind"

_PATHFIND_KEYWORDS = ("gap", "path between", "path from", "route from", "route to")
_CONNECT_KEYWORDS = ("skills for", "skills does", "neighbors", "essential skill", "optional skill")


def classify_capability(question: str) -> str:
    q = question.lower()
    if any(k in q for k in _PATHFIND_KEYWORDS):
        return CAPABILITY_PATHFIND
    if any(k in q for k in _CONNECT_KEYWORDS):
        return CAPABILITY_CONNECT
    return CAPABILITY_LOCATE

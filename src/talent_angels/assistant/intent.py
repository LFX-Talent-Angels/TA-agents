"""Heuristic intent routing (MVP plan Sec 5) — zero tokens, keyword-based.

Only `locate` is implemented in Gate A. A question that heuristically routes
to `connect`/`pathfind` still returns a typed result — with an explicit
`capability_not_implemented` warning — rather than silently answering with
Locate instead (ARCHITECTURE.md: never invent; warnings are how "no" is said).
"""

from __future__ import annotations

import re
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
_CONNECT_KEYWORDS = (
    "skills for",
    "skills does",
    "skills i need",
    "skills do i",
    "need to become",
    "become a",
    "become an",
    "neighbors",
    "essential skill",
    "optional skill",
)

_SUBJECT_PATTERNS = (
    re.compile(r"^what essential skills does (?:a |an )?(.+?) need\b", re.I),
    re.compile(r"^what skills does (?:a |an )?(.+?) need\b", re.I),
    re.compile(
        r"^what skills (?:(?:do )?i |does one )?need to (?:become|be) (?:a |an |the )?(.+)$",
        re.I,
    ),
    re.compile(
        r"skills (?:(?:do )?i |does one )?need to (?:become|be) (?:a |an |the )?(.+)$",
        re.I,
    ),
    re.compile(
        r"^(?:i want to |i'd like to |to )(?:become|be) (?:a |an |the )?(.+)$",
        re.I,
    ),
)


def extract_locate_subject(question: str) -> str:
    """Strip common wrappers so lexical search sees the occupation/skill phrase."""
    text = question.strip()
    if text.endswith("?"):
        text = text[:-1].rstrip()
    lowered = text.lower()
    if lowered.endswith(" in esco"):
        text = text[: -len(" in esco")].rstrip()
        lowered = text.lower()

    for pattern in _SUBJECT_PATTERNS:
        match = pattern.match(text)
        if match:
            return match.group(1).strip() or question.strip()

    for prefix in ("where is ", "where are ", "what is ", "what are ", "find ", "locate "):
        if lowered.startswith(prefix):
            text = text[len(prefix) :].strip()
            break
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

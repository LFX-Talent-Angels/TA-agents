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
    "path between",
    "path from",
    "path to",
    "skill path",
    "career path",
    "learning path",
    "route from",
    "route to",
    "gap from",
    "gap between",
    "→",
    "->",
)
_COMPARE_RE = re.compile(r"\b(vs\.?|versus)\b", re.I)
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


def _in_suite_suffixes() -> tuple[str, ...]:
    """' in esco', ' in o*net', … from the alias table — not ESCO-only."""
    from talent_angels.assistant.merge import suite_heading
    from talent_angels.assistant.suite_select import SUITE_ALIASES

    suffixes: list[str] = []
    for name, aliases in SUITE_ALIASES.items():
        for alias in (name, suite_heading(name).casefold(), *aliases):
            token = f" in {alias.casefold()}"
            if token not in suffixes:
                suffixes.append(token)
    return tuple(suffixes)


_SUBJECT_PATTERNS = (
    re.compile(r"^what does (?:a |an |the )?(.+?) do\b", re.I),
    re.compile(r"^what do (?:a |an |the )?(.+?) do\b", re.I),
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
    for suffix in _in_suite_suffixes():
        if lowered.endswith(suffix):
            text = text[: -len(suffix)].rstrip()
            lowered = text.lower()
            break

    for pattern in _SUBJECT_PATTERNS:
        match = pattern.match(text)
        if match:
            return match.group(1).strip() or question.strip()

    for prefix in ("where is ", "where are ", "what is ", "what are ", "find ", "locate "):
        if lowered.startswith(prefix):
            text = text[len(prefix) :].strip()
            break
    return text.strip() or question.strip()


_PATH_ENDS = re.compile(
    r"(?:path|route|gap)\s+(?:from\s+)?(.+?)\s+to\s+(.+?)\s*$",
    re.I,
)
_FROM_TO = re.compile(r"\bfrom\s+(.+?)\s+to\s+(.+?)\s*$", re.I)


def extract_pathfind_endpoints(question: str) -> tuple[str, str] | None:
    """Best-effort 'from A to B' split. Planner draft wins when present."""
    text = question.strip().rstrip("?.!")
    if _COMPARE_RE.search(text):
        return None
    for pattern in (_PATH_ENDS, _FROM_TO):
        match = pattern.search(text)
        if match:
            left = match.group(1).strip()
            right = match.group(2).strip()
            if left and right:
                return left, right
    return None


def classify_capability(question: str) -> Capability:
    q = question.lower()
    padded = f" {q} "
    if _COMPARE_RE.search(q):
        if any(k in q for k in _CONNECT_KEYWORDS):
            return CAPABILITY_CONNECT
        return CAPABILITY_LOCATE
    if any(k in q for k in _PATHFIND_KEYWORDS):
        return CAPABILITY_PATHFIND
    if " from " in padded and " to " in padded:
        return CAPABILITY_PATHFIND
    if any(k in q for k in _CONNECT_KEYWORDS):
        return CAPABILITY_CONNECT
    return CAPABILITY_LOCATE

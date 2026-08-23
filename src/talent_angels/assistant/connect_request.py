"""Deterministic request extraction for the thin Connect MVP."""

from __future__ import annotations

import re

from talent_angels.skills.connect.models import ConnectRequest


class UnsupportedConnectQuery(ValueError):
    """Raised when the zero-token MVP cannot safely identify a subject."""


_SKILLS_DOES = re.compile(
    r"(?:essential\s+|optional\s+)?skills\s+does\s+"
    r"(?:(?:an?|the)\s+)?(.+?)\s+need[?.!]*$",
    re.IGNORECASE,
)
_SKILLS_FOR = re.compile(
    r"skills\s+for\s+(?:(?:an?|the)\s+)?(.+?)[?.!]*$",
    re.IGNORECASE,
)
_SKILLS_NEED_TO_BE = re.compile(
    r"skills\s+(?:(?:do\s+)?i\s+|does\s+one\s+)?need\s+to\s+(?:become|be)\s+"
    r"(?:(?:an?|the)\s+)?(.+?)[?.!]*$",
    re.IGNORECASE,
)
_NEIGHBORS_OF = re.compile(
    r"neighbou?rs\s+of\s+(?:(?:an?|the)\s+)?(.+?)[?.!]*$",
    re.IGNORECASE,
)


def extract_connect_request(question: str) -> ConnectRequest:
    """Extract only the explicitly supported zero-token Connect query forms."""
    cleaned = question.strip()
    lowered = cleaned.lower()
    relation_kind = None
    if "essential skill" in lowered:
        relation_kind = "essential"
    elif "optional skill" in lowered:
        relation_kind = "optional"

    match = (
        _SKILLS_DOES.search(cleaned)
        or _SKILLS_FOR.search(cleaned)
        or _SKILLS_NEED_TO_BE.search(cleaned)
    )
    if match:
        return ConnectRequest(
            subject=match.group(1).strip(),
            rel_types=("HAS_SKILL",),
            relation_kind=relation_kind,
        )

    match = _NEIGHBORS_OF.search(cleaned)
    if match:
        return ConnectRequest(subject=match.group(1).strip())

    raise UnsupportedConnectQuery("could not extract a Connect subject from the question")
